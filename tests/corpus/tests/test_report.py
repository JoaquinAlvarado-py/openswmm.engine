import pandas as pd
import pytest

from swmmbench import report, schema


METRICS = ["avg_iterations_per_step", "continuity_error_flow"]


def _runs():
    return pd.DataFrame([
        {"model_id": "EPA/m1", "family": "EPA", "variant": "A",
         "status": "ok", "avg_iterations_per_step": 4.0,
         "continuity_error_flow": 0.10, "iteration_metric_kind": "picard"},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "B",
         "status": "ok", "avg_iterations_per_step": 2.0,
         "continuity_error_flow": 0.12, "iteration_metric_kind": "picard"},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "C",
         "status": "ok", "avg_iterations_per_step": 3.0,
         "continuity_error_flow": 0.11, "iteration_metric_kind": "picard"},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "REF",
         "status": "ok", "avg_iterations_per_step": 3.8,
         "continuity_error_flow": 0.09, "iteration_metric_kind": "picard"},
    ])


def test_b_minus_a_isolates_the_feature_effect():
    deltas = report.build_deltas(_runs(), METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert row["value_a"] == pytest.approx(4.0)
    assert row["value_b"] == pytest.approx(2.0)
    assert row["delta_b_minus_a"] == pytest.approx(-2.0)


def test_a_minus_ref_measures_parity_debt():
    deltas = report.build_deltas(_runs(), METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert row["delta_a_minus_ref"] == pytest.approx(0.2)


def test_a_model_without_a_reference_still_yields_the_feature_delta():
    runs = _runs()
    runs = runs[runs.variant != "REF"]

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "continuity_error_flow"].iloc[0]

    assert row["delta_b_minus_a"] == pytest.approx(0.02)
    assert pd.isna(row["delta_a_minus_ref"])


def test_kind_mismatch_nulls_only_the_b_delta_and_keeps_the_row():
    runs = _runs()
    runs.loc[runs.variant == "B", "iteration_metric_kind"] = schema.ITER_FV

    deltas = report.build_deltas(runs, METRICS)
    iterations = deltas[deltas.metric == "avg_iterations_per_step"]
    row = iterations.iloc[0]

    assert len(iterations) == 1
    assert pd.isna(row["delta_b_minus_a"])
    # A and REF still share a counter (both "picard"), so parity debt is
    # still measurable even though the A-B comparison is not.
    assert row["delta_a_minus_ref"] == pytest.approx(0.2)


def test_a_minus_ref_survives_a_missing_b_variant():
    runs = _runs()
    runs = runs[runs.variant != "B"]

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert pd.isna(row["delta_b_minus_a"])
    assert row["delta_a_minus_ref"] == pytest.approx(0.2)


def test_a_ref_kind_mismatch_drops_only_the_ref_delta():
    runs = _runs()
    runs.loc[runs.variant == "REF", "iteration_metric_kind"] = schema.ITER_FV

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert pd.isna(row["delta_a_minus_ref"])
    assert row["delta_b_minus_a"] == pytest.approx(-2.0)


def test_c_minus_a_isolates_the_continuity_effect():
    deltas = report.build_deltas(_runs(), METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert row["value_c"] == pytest.approx(3.0)
    assert row["delta_c_minus_a"] == pytest.approx(-1.0)


def test_a_model_without_a_c_variant_still_yields_the_b_and_ref_deltas():
    runs = _runs()
    runs = runs[runs.variant != "C"]

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert pd.isna(row["delta_c_minus_a"])
    assert row["delta_b_minus_a"] == pytest.approx(-2.0)
    assert row["delta_a_minus_ref"] == pytest.approx(0.2)


def test_c_kind_mismatch_nulls_only_the_c_delta_and_keeps_the_row():
    runs = _runs()
    runs.loc[runs.variant == "C", "iteration_metric_kind"] = schema.ITER_FV

    deltas = report.build_deltas(runs, METRICS)
    iterations = deltas[deltas.metric == "avg_iterations_per_step"]
    row = iterations.iloc[0]

    assert len(iterations) == 1
    assert pd.isna(row["delta_c_minus_a"])
    # A, B and REF still share a counter, so their deltas are unaffected.
    assert row["delta_b_minus_a"] == pytest.approx(-2.0)
    assert row["delta_a_minus_ref"] == pytest.approx(0.2)


def test_a_c_kind_mismatch_does_not_cost_the_a_ref_delta():
    runs = _runs()
    runs.loc[runs.variant == "A", "iteration_metric_kind"] = schema.ITER_FV

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    # A now mismatches both B and C in kind (both still "picard"), and REF
    # (also still "picard"), so every A-anchored delta nulls -- but the row
    # itself must survive.
    assert pd.isna(row["delta_b_minus_a"])
    assert pd.isna(row["delta_c_minus_a"])
    assert pd.isna(row["delta_a_minus_ref"])


def test_markdown_names_the_estimate_as_an_estimate(tmp_path):
    deltas = report.build_deltas(_runs(), METRICS)

    path = report.write_markdown(deltas, _runs(), tmp_path / "summary.md")
    text = path.read_text(encoding="utf-8")

    assert "estimate" in text.lower()
    assert "EPA" in text


def test_markdown_reports_both_b_minus_a_and_c_minus_a_axes(tmp_path):
    deltas = report.build_deltas(_runs(), METRICS)

    path = report.write_markdown(deltas, _runs(), tmp_path / "summary.md")
    text = path.read_text(encoding="utf-8")

    assert "B - A" in text
    assert "C - A" in text
    # The two axes are kept visible side by side, not collapsed into one.
    b_section = text.index("B - A")
    c_section = text.index("C - A")
    assert b_section != c_section


def test_empty_input_produces_an_empty_frame_not_an_error():
    assert report.build_deltas(pd.DataFrame(), METRICS).empty
