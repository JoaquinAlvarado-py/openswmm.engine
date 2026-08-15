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


def test_iteration_deltas_across_mismatched_metric_kinds_are_dropped():
    runs = _runs()
    runs.loc[runs.variant == "B", "iteration_metric_kind"] = schema.ITER_FV

    deltas = report.build_deltas(runs, METRICS)
    iterations = deltas[deltas.metric == "avg_iterations_per_step"]

    assert iterations.empty


def test_markdown_names_the_estimate_as_an_estimate(tmp_path):
    deltas = report.build_deltas(_runs(), METRICS)

    path = report.write_markdown(deltas, _runs(), tmp_path / "summary.md")
    text = path.read_text(encoding="utf-8")

    assert "estimate" in text.lower()
    assert "EPA" in text


def test_empty_input_produces_an_empty_frame_not_an_error():
    assert report.build_deltas(pd.DataFrame(), METRICS).empty
