"""Executed coverage for the `report` stage itself.

`report.py`'s pure functions are covered by test_report.py, but the stage that
drives them -- and enforces the replace-not-append rule for the derived tables,
the most load-bearing cross-stage rule in the harness -- was reachable only
through the engine-gated end-to-end test. These tests build the Parquet inputs
directly, so they need no engine.
"""

import pandas as pd
import pytest

from swmmbench import cli, report, schema, store

ENGINE_V1 = "openswmm 1.0.0 (build abc123)"
ENGINE_V2 = "openswmm 1.1.0 (build def456)"


def _runs(engine_version: str = ENGINE_V1) -> pd.DataFrame:
    """Two models x {A, B, C, D, E, REF}, with the scalars the report pivots on."""
    rows = []
    for model_id, base in (("EPA/m1", 4.0), ("LID/m2", 6.0)):
        family = model_id.split("/")[0]
        for variant, offset, version in (
            (schema.VARIANT_A, 0.0, engine_version),
            (schema.VARIANT_B, -1.0, engine_version),
            (schema.VARIANT_C, -0.4, engine_version),
            (schema.VARIANT_D, -1.6, engine_version),
            (schema.VARIANT_E, 0.5, engine_version),
            (schema.VARIANT_REF, 0.2, cli.REF_ENGINE_VERSION),
        ):
            rows.append({
                "model_id": model_id,
                "family": family,
                "variant": variant,
                "engine_version": version,
                "status": schema.Status.OK,
                "avg_iterations_per_step": base + offset,
                "continuity_error_flow": 0.1 + offset / 100.0,
                "avg_step": 10.0,
                "wall_ms": 100.0,
                "iteration_metric_kind": schema.ITER_PICARD,
                # An iteration delta is only formed when both sides are
                # known DYNWAVE, and an `X - REF` delta only when both echo
                # the same surcharge method.
                "reported_routing_model": schema.ROUTING_DYNWAVE,
                "reported_surcharge_method":
                    "DYNAMIC_SLOT" if variant == schema.VARIANT_E else "EXTRAN",
            })
    return pd.DataFrame(rows)


def _elements(engine_version: str = ENGINE_V1) -> pd.DataFrame:
    rows = []
    for model_id in ("EPA/m1", "LID/m2"):
        family = model_id.split("/")[0]
        for variant, value, version in (
            (schema.VARIANT_A, 1.0, engine_version),
            (schema.VARIANT_B, 1.5, engine_version),
            (schema.VARIANT_C, 1.2, engine_version),
            (schema.VARIANT_D, 0.8, engine_version),
            (schema.VARIANT_E, 1.9, engine_version),
            (schema.VARIANT_REF, 0.9, cli.REF_ENGINE_VERSION),
        ):
            rows.append({
                "model_id": model_id,
                "family": family,
                "variant": variant,
                "engine_version": version,
                "element_type": "NODE",
                "element_id": "n1",
                "metric": "max_depth",
                "value": value,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def store_dir(tmp_path):
    """A results store holding one engine build's runs and elements."""
    out = tmp_path / "out"
    store.write_table(_runs(), out, "runs", partition_by=["family"])
    store.write_table(_elements(), out, "elements", partition_by=["family"])
    return out


def test_report_writes_the_derived_tables(store_dir):
    code = cli.stage_report(store_dir)

    assert code == 0
    deltas = store.read_table(store_dir, "deltas")
    assert not deltas.empty
    assert set(deltas["model_id"]) == {"EPA/m1", "LID/m2"}
    assert "delta_b_minus_a" in deltas.columns

    element_deltas = store.read_table(store_dir, "element_deltas")
    assert not element_deltas.empty
    assert element_deltas["delta_b_minus_a"].iloc[0] == pytest.approx(0.5)


def test_every_value_and_delta_column_survives_the_parquet_round_trip(store_dir):
    # The D and E columns are what the five-variant sweep is paid for; a
    # column that build_deltas emits but the store drops (or that arrives
    # back as all-null) would publish nothing from 40% of the compute.
    cli.stage_report(store_dir)

    for name in ("deltas", "element_deltas"):
        table = store.read_table(store_dir, name)
        assert set(report.VALUE_COLUMNS) <= set(table.columns), name
        assert set(report.DELTA_COLUMNS) <= set(table.columns), name
        for column in report.DELTA_COLUMNS:
            assert table[column].notna().any(), f"{name}.{column}"


def test_the_d_minus_c_axis_is_published_by_the_stage(store_dir):
    cli.stage_report(store_dir)

    deltas = store.read_table(store_dir, "deltas")
    row = deltas[(deltas.model_id == "EPA/m1")
                 & (deltas.metric == "avg_iterations_per_step")].iloc[0]

    # A=4.0, C=3.6, D=2.4 -> D - C = -1.2 and D - A = -1.6.
    assert row["delta_d_minus_c"] == pytest.approx(-1.2)
    assert row["delta_d_minus_a"] == pytest.approx(-1.6)


def test_the_e_ref_surcharge_mismatch_does_not_cost_a_minus_ref(store_dir):
    # E echoes DYNAMIC_SLOT while REF echoes EXTRAN; the anchor guard is
    # per-delta, so A's parity debt is unaffected.
    cli.stage_report(store_dir)

    deltas = store.read_table(store_dir, "deltas")
    row = deltas[(deltas.model_id == "EPA/m1")
                 & (deltas.metric == "avg_iterations_per_step")].iloc[0]

    assert row["delta_a_minus_ref"] == pytest.approx(-0.2)
    assert row["delta_e_minus_a"] == pytest.approx(0.5)


def test_report_writes_a_summary_naming_the_estimate_caveat(store_dir):
    # stage_report's default language is `es`; assert via the same mapping
    # write_markdown draws from, not a hardcoded copy of the translation.
    cli.stage_report(store_dir)

    summary = store_dir / "summary.md"
    assert summary.is_file()
    text = summary.read_text(encoding="utf-8")

    assert "total_iterations_est" in text
    assert "\n".join(report.MARKDOWN_STRINGS["es"]["caveat_estimate"]) in text


def test_a_second_report_replaces_rather_than_doubles_the_derived_tables(store_dir):
    """The rule the whole derived-table design rests on.

    store.write_table appends (fresh UUID basename per call), so without
    _replace_table a second `report` would double every delta row and silently
    corrupt any aggregation computed over them.
    """
    cli.stage_report(store_dir)
    before_deltas = len(store.read_table(store_dir, "deltas"))
    before_elements = len(store.read_table(store_dir, "element_deltas"))

    cli.stage_report(store_dir)

    assert len(store.read_table(store_dir, "deltas")) == before_deltas
    assert len(store.read_table(store_dir, "element_deltas")) == before_elements


def test_report_on_an_empty_store_says_so_without_failing(tmp_path):
    assert cli.stage_report(tmp_path / "empty") == 0


def test_the_interaction_axis_is_published_by_the_stage(store_dir):
    cli.stage_report(store_dir)

    deltas = store.read_table(store_dir, "deltas")
    row = deltas[(deltas.model_id == "EPA/m1")
                 & (deltas.metric == "avg_iterations_per_step")].iloc[0]

    # A=4.0, B=3.0, C=3.6, D=2.4 -> (D - C) - (B - A) = -1.2 - (-1.0) = -0.2.
    assert row["delta_interaction"] == pytest.approx(-0.2)

    text = (store_dir / "summary.md").read_text(encoding="utf-8")
    assert report.MARKDOWN_STRINGS["es"]["interaction_heading"] in text


# ---------------------------------------------------------------------------
# Time-series coverage reaches the summary from the store
# ---------------------------------------------------------------------------


def _ts_diff(coverage: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "model_id": "EPA/m1", "family": "EPA", "comparison": "D_minus_C",
        "element_type": "NODE", "element_id": "n1",
        "attribute": "INVERT_DEPTH", "max_abs": 0.0, "max_rel": 0.0,
        "rmse": 0.0, "n_periods_left": 24,
        "n_periods_right": int(round(coverage * 24)),
        "n_common": int(round(coverage * 24)), "coverage_fraction": coverage,
        "start_time_match": True, "end_time_match": coverage == 1.0,
        "time_grid_match": coverage == 1.0,
    }])


def test_the_stage_carries_an_incomplete_overlap_into_the_summary(store_dir):
    # A near-perfect `max_abs` over half a run must not be published as a
    # full comparison; the stage reads `ts_diff` so the summary can say so.
    store.write_table(_ts_diff(0.5), store_dir, "ts_diff", partition_by=["family"])

    cli.stage_report(store_dir)

    text = (store_dir / "summary.md").read_text(encoding="utf-8")
    strings = report.MARKDOWN_STRINGS["es"]

    assert "| D_minus_C | 1 | 1 | 0.5000 |" in text
    assert "\n".join(strings["coverage_anomaly_none"]) not in text


def test_a_store_with_full_coverage_says_so(store_dir):
    store.write_table(_ts_diff(1.0), store_dir, "ts_diff", partition_by=["family"])

    cli.stage_report(store_dir)

    text = (store_dir / "summary.md").read_text(encoding="utf-8")
    strings = report.MARKDOWN_STRINGS["es"]

    assert "\n".join(strings["coverage_anomaly_none"]) in text
    assert "\n".join(strings["coverage_not_assessed"]) not in text


def test_a_store_with_no_ts_diff_reports_coverage_as_unassessed(store_dir):
    cli.stage_report(store_dir)

    text = (store_dir / "summary.md").read_text(encoding="utf-8")

    assert "\n".join(
        report.MARKDOWN_STRINGS["es"]["coverage_not_assessed"]) in text


# ---------------------------------------------------------------------------
# Mixed engine builds
# ---------------------------------------------------------------------------


def test_report_refuses_a_store_mixing_two_engine_builds(store_dir, capsys):
    # `runs` is keyed on engine_version, so two builds legitimately coexist.
    # The pivot is not -- it would arbitrarily pick one build's value per cell.
    store.write_table(_runs(ENGINE_V2), store_dir, "runs", partition_by=["family"])

    code = cli.stage_report(store_dir)
    out = capsys.readouterr().out

    assert code != 0
    assert ENGINE_V1 in out
    assert ENGINE_V2 in out


def test_a_refused_report_writes_nothing(store_dir):
    store.write_table(_runs(ENGINE_V2), store_dir, "runs", partition_by=["family"])

    cli.stage_report(store_dir)

    assert store.read_table(store_dir, "deltas").empty
    assert not (store_dir / "summary.md").exists()


def test_a_refusal_leaves_an_earlier_report_intact(store_dir):
    # Refusing must not destroy the previous, single-build report either.
    cli.stage_report(store_dir)
    before = len(store.read_table(store_dir, "deltas"))
    store.write_table(_runs(ENGINE_V2), store_dir, "runs", partition_by=["family"])

    cli.stage_report(store_dir)

    assert len(store.read_table(store_dir, "deltas")) == before


def test_main_surfaces_the_refusal_as_a_non_zero_exit(store_dir):
    store.write_table(_runs(ENGINE_V2), store_dir, "runs", partition_by=["family"])

    assert cli.main(["report", "--out", str(store_dir)]) != 0


def test_the_reference_sentinel_is_not_counted_as_an_engine_build(store_dir):
    # REF rows always carry `corpus-reference`; counting it would make every
    # single-build store look mixed.
    versions = cli.executed_engine_versions(store.read_table(store_dir, "runs"))

    assert versions == [ENGINE_V1]


# ---------------------------------------------------------------------------
# Engine-echoed option values vs. variant intent
# ---------------------------------------------------------------------------


def _option_echo_runs(c_reported_node_continuity: str = "SEMI_IMPLICIT") -> pd.DataFrame:
    """One model x {A, B, C}, each echoing its intended option value unless
    overridden -- e.g. a C run whose engine actually resolved EXPLICIT."""
    variant_echoes = {
        schema.VARIANT_A: ("EXPLICIT", "NO"),
        schema.VARIANT_B: ("EXPLICIT", "YES"),
        schema.VARIANT_C: (c_reported_node_continuity, "NO"),
    }
    return pd.DataFrame([
        {"model_id": "EPA/m1", "family": "EPA", "variant": variant,
         "engine_version": ENGINE_V1, "status": schema.Status.OK,
         "reported_node_continuity": node_continuity,
         "reported_anderson_accel": anderson}
        for variant, (node_continuity, anderson) in variant_echoes.items()
    ])


def test_report_warns_about_a_c_run_that_echoes_explicit_node_continuity(
    tmp_path, capsys,
):
    out = tmp_path / "out"
    store.write_table(_option_echo_runs(c_reported_node_continuity="EXPLICIT"),
                      out, "runs", partition_by=["family"])

    code = cli.stage_report(out)

    assert code == 0  # an anomaly is reported, not a hard failure
    captured = capsys.readouterr().out
    assert "EPA/m1" in captured
    assert "NODE_CONTINUITY" in captured


def test_report_stays_silent_when_every_run_echoes_its_intended_option(
    tmp_path, capsys,
):
    out = tmp_path / "out"
    store.write_table(_option_echo_runs(), out, "runs", partition_by=["family"])

    cli.stage_report(out)

    captured = capsys.readouterr().out
    assert "NODE_CONTINUITY" not in captured
    assert "ANDERSON_ACCEL" not in captured


def test_the_stage_carries_the_anomaly_into_the_summary_not_just_the_console(
    tmp_path,
):
    # The console line scrolls away; `summary.md` is what gets shared.
    out = tmp_path / "out"
    store.write_table(_option_echo_runs(c_reported_node_continuity="EXPLICIT"),
                      out, "runs", partition_by=["family"])

    cli.stage_report(out)

    text = (out / "summary.md").read_text(encoding="utf-8")
    strings = report.MARKDOWN_STRINGS["es"]

    assert strings["anomaly_heading"] in text
    assert f"| {schema.VARIANT_C} | NODE_CONTINUITY | 1 | 1 |" in text


def test_a_clean_store_states_the_absence_of_anomalies_in_the_summary(store_dir):
    cli.stage_report(store_dir)

    text = (store_dir / "summary.md").read_text(encoding="utf-8")

    assert "\n".join(report.MARKDOWN_STRINGS["es"]["anomaly_none"]) in text
