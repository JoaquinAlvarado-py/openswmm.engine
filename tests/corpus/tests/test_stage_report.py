"""Executed coverage for the `report` stage itself.

`report.py`'s pure functions are covered by test_report.py, but the stage that
drives them -- and enforces the replace-not-append rule for the derived tables,
the most load-bearing cross-stage rule in the harness -- was reachable only
through the engine-gated end-to-end test. These tests build the Parquet inputs
directly, so they need no engine.
"""

import pandas as pd
import pytest

from swmmbench import cli, schema, store

ENGINE_V1 = "openswmm 1.0.0 (build abc123)"
ENGINE_V2 = "openswmm 1.1.0 (build def456)"


def _runs(engine_version: str = ENGINE_V1) -> pd.DataFrame:
    """Two models x {A, B, REF}, with the scalars the report pivots on."""
    rows = []
    for model_id, base in (("EPA/m1", 4.0), ("LID/m2", 6.0)):
        family = model_id.split("/")[0]
        for variant, offset, version in (
            (schema.VARIANT_A, 0.0, engine_version),
            (schema.VARIANT_B, -1.0, engine_version),
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
            })
    return pd.DataFrame(rows)


def _elements(engine_version: str = ENGINE_V1) -> pd.DataFrame:
    rows = []
    for model_id in ("EPA/m1", "LID/m2"):
        family = model_id.split("/")[0]
        for variant, value, version in (
            (schema.VARIANT_A, 1.0, engine_version),
            (schema.VARIANT_B, 1.5, engine_version),
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


def test_report_writes_a_summary_naming_the_estimate_caveat(store_dir):
    cli.stage_report(store_dir)

    summary = store_dir / "summary.md"
    assert summary.is_file()
    text = summary.read_text(encoding="utf-8")

    assert "total_iterations_est" in text
    assert "estimate" in text.lower()


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


def test_the_reference_sentinel_is_not_counted_as_an_engine_build(store_dir):
    # REF rows always carry `corpus-reference`; counting it would make every
    # single-build store look mixed.
    versions = cli.executed_engine_versions(store.read_table(store_dir, "runs"))

    assert versions == [ENGINE_V1]
