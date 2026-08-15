"""Coverage for the `lang` option on `report.write_markdown` / `cli.stage_report`.

The mapping in `report.MARKDOWN_STRINGS` is the single source of truth for
the translated prose; these tests assert against it rather than hardcoding a
second copy of the translation, so the two cannot drift apart.
"""

import pandas as pd
import pytest

from swmmbench import cli, report, schema, store

METRICS = ["avg_iterations_per_step", "continuity_error_flow"]


def _runs():
    """Two families, two statuses, four variants for EPA/m1."""
    return pd.DataFrame([
        {"model_id": "EPA/m1", "family": "EPA", "variant": "A", "status": "ok",
         "avg_iterations_per_step": 4.0, "continuity_error_flow": 0.10,
         "iteration_metric_kind": "picard"},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "B", "status": "ok",
         "avg_iterations_per_step": 2.0, "continuity_error_flow": 0.12,
         "iteration_metric_kind": "picard"},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "C", "status": "ok",
         "avg_iterations_per_step": 3.0, "continuity_error_flow": 0.11,
         "iteration_metric_kind": "picard"},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "REF", "status": "ok",
         "avg_iterations_per_step": 3.8, "continuity_error_flow": 0.09,
         "iteration_metric_kind": "picard"},
        {"model_id": "LID/m2", "family": "LID", "variant": "A", "status": "engine_error"},
    ])


def _write(tmp_path, lang):
    deltas = report.build_deltas(_runs(), METRICS)
    path = report.write_markdown(deltas, _runs(), tmp_path / f"summary_{lang}.md",
                                 lang=lang)
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Headings translate
# ---------------------------------------------------------------------------


def test_write_markdown_lang_es_produces_spanish_headings(tmp_path):
    text = _write(tmp_path, "es")
    es = report.MARKDOWN_STRINGS["es"]

    assert text.startswith(es["title"])
    for key in ("coverage_heading", "b_minus_a_heading", "c_minus_a_heading",
                "iter_shift_b_heading", "iter_shift_c_heading", "caveats_heading"):
        assert es[key] in text

    en = report.MARKDOWN_STRINGS["en"]
    assert en["title"] not in text
    assert en["coverage_heading"] not in text


def test_write_markdown_lang_en_produces_english_headings(tmp_path):
    text = _write(tmp_path, "en")
    en = report.MARKDOWN_STRINGS["en"]

    assert text.startswith(en["title"])
    for key in ("coverage_heading", "b_minus_a_heading", "c_minus_a_heading",
                "iter_shift_b_heading", "iter_shift_c_heading", "caveats_heading"):
        assert en[key] in text

    es = report.MARKDOWN_STRINGS["es"]
    assert es["title"] not in text
    assert es["coverage_heading"] not in text


def test_default_lang_is_es(tmp_path):
    deltas = report.build_deltas(_runs(), METRICS)
    path = report.write_markdown(deltas, _runs(), tmp_path / "summary.md")
    text = path.read_text(encoding="utf-8")

    assert text.startswith(report.MARKDOWN_STRINGS["es"]["title"])


# ---------------------------------------------------------------------------
# Structure and data are identical between languages -- only the prose differs
# ---------------------------------------------------------------------------


def _heading_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("## ")]


def _all_known_header_and_sep_lines() -> set[str]:
    """Every table header/separator line either language can produce.

    Used to strip prose lines from a document, leaving only data rows --
    without hardcoding a second copy of the header text.
    """
    keys = ("coverage_header", "coverage_sep", "metric_header", "metric_sep",
            "family_header", "family_sep")
    return {report.MARKDOWN_STRINGS[lang][key]
            for lang in report.MARKDOWN_STRINGS for key in keys}


def _data_rows(text: str) -> list[str]:
    """Table rows carrying actual values, not headings or header/sep lines."""
    known = _all_known_header_and_sep_lines()
    return [line for line in text.splitlines()
            if line.startswith("| ") and line not in known]


def test_both_languages_produce_the_same_section_count(tmp_path):
    en_text = _write(tmp_path, "en")
    es_text = _write(tmp_path, "es")

    assert len(_heading_lines(en_text)) == len(_heading_lines(es_text)) == 6


def test_both_languages_produce_identical_data_rows_and_numeric_values(tmp_path):
    # Data rows include the metric name, family, status and every computed
    # number. Since none of that is prose, the rows must be byte-identical
    # across languages -- only the surrounding headings/headers differ.
    en_text = _write(tmp_path, "en")
    es_text = _write(tmp_path, "es")

    en_rows = _data_rows(en_text)
    es_rows = _data_rows(es_text)

    assert en_rows == es_rows
    assert en_rows  # sanity: the fixture actually produces data rows
    assert any("avg_iterations_per_step" in row for row in en_rows)
    assert any("-2.0000" in row for row in en_rows)  # B - A delta for EPA/m1


# ---------------------------------------------------------------------------
# Data values stay verbatim -- they are identifiers, not prose
# ---------------------------------------------------------------------------


def test_metric_names_are_untranslated_in_both_languages(tmp_path):
    for lang in ("en", "es"):
        text = _write(tmp_path, lang)
        assert "avg_iterations_per_step" in text
        assert "continuity_error_flow" in text


def test_status_values_are_untranslated_in_both_languages(tmp_path):
    for lang in ("en", "es"):
        text = _write(tmp_path, lang)
        assert "| ok | " in text
        assert "| engine_error | " in text


def test_family_names_are_untranslated_in_both_languages(tmp_path):
    for lang in ("en", "es"):
        text = _write(tmp_path, lang)
        assert "| EPA | " in text


# ---------------------------------------------------------------------------
# The estimate caveat: present, and never softened, in both languages
# ---------------------------------------------------------------------------


def test_estimate_caveat_present_and_names_total_iterations_est_in_both_languages(
    tmp_path,
):
    for lang in ("en", "es"):
        text = _write(tmp_path, lang)
        assert "total_iterations_est" in text
        assert "\n".join(report.MARKDOWN_STRINGS[lang]["caveat_estimate"]) in text


# ---------------------------------------------------------------------------
# --lang reaches stage_report; default is es
# ---------------------------------------------------------------------------


@pytest.fixture
def store_dir(tmp_path):
    out = tmp_path / "out"
    store.write_table(_runs(), out, "runs", partition_by=["family"])
    return out


def test_stage_report_lang_parameter_reaches_write_markdown(store_dir):
    cli.stage_report(store_dir, lang="en")

    text = (store_dir / "summary.md").read_text(encoding="utf-8")
    assert text.startswith(report.MARKDOWN_STRINGS["en"]["title"])


def test_stage_report_default_lang_is_es(store_dir):
    cli.stage_report(store_dir)

    text = (store_dir / "summary.md").read_text(encoding="utf-8")
    assert text.startswith(report.MARKDOWN_STRINGS["es"]["title"])


def test_cli_main_lang_flag_reaches_stage_report(store_dir):
    code = cli.main(["report", "--out", str(store_dir), "--lang", "en"])

    assert code == 0
    text = (store_dir / "summary.md").read_text(encoding="utf-8")
    assert text.startswith(report.MARKDOWN_STRINGS["en"]["title"])


def test_cli_main_without_lang_flag_defaults_to_es(store_dir):
    code = cli.main(["report", "--out", str(store_dir)])

    assert code == 0
    text = (store_dir / "summary.md").read_text(encoding="utf-8")
    assert text.startswith(report.MARKDOWN_STRINGS["es"]["title"])


def test_cli_lang_flag_rejects_an_unsupported_language(store_dir, capsys):
    with pytest.raises(SystemExit):
        cli.main(["report", "--out", str(store_dir), "--lang", "fr"])
