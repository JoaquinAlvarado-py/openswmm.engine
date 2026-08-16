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

    for lang in ("en", "es"):
        path = report.write_markdown(deltas, _runs(), tmp_path / f"summary_{lang}.md",
                                     lang=lang)
        text = path.read_text(encoding="utf-8")

        assert "total_iterations_est" in text
        # Asserted via the same mapping write_markdown draws from, not a
        # hardcoded copy of the translation, so the two cannot drift apart.
        assert "\n".join(report.MARKDOWN_STRINGS[lang]["caveat_estimate"]) in text
        assert "EPA" in text


def _markdown_section(text: str, heading: str) -> str:
    """The body of a `## heading` section, up to the next `## ` heading."""
    start = text.index(heading) + len(heading)
    rest = text[start:]
    end = rest.find("\n## ")
    return rest[:end] if end != -1 else rest


def test_markdown_reports_both_b_minus_a_and_c_minus_a_axes_with_their_own_rows(
    tmp_path,
):
    # A test asserting only that the strings "B - A" and "C - A" appear
    # anywhere in the document would also pass with both axis sections
    # deleted, because the Caveats text mentions them. This instead checks
    # each axis's section is a real, separate table carrying that axis's own
    # computed values -- not the other axis's, and not an empty heading.
    deltas = report.build_deltas(_runs(), METRICS)
    strings = report.MARKDOWN_STRINGS["en"]

    path = report.write_markdown(deltas, _runs(), tmp_path / "summary.md", lang="en")
    text = path.read_text(encoding="utf-8")

    b_section = _markdown_section(text, strings["b_minus_a_heading"])
    c_section = _markdown_section(text, strings["c_minus_a_heading"])

    # _runs(): avg_iterations_per_step is A=4.0, B=2.0, C=3.0, so
    # delta_b_minus_a = -2.0 and delta_c_minus_a = -1.0 for the one model.
    assert "avg_iterations_per_step" in b_section
    assert "-2.0000" in b_section
    assert "-1.0000" not in b_section

    assert "avg_iterations_per_step" in c_section
    assert "-1.0000" in c_section
    assert "-2.0000" not in c_section


def test_empty_input_produces_an_empty_frame_not_an_error():
    assert report.build_deltas(pd.DataFrame(), METRICS).empty


# ---------------------------------------------------------------------------
# Engine-echoed option values vs. variant intent
# ---------------------------------------------------------------------------


def _option_runs(**overrides):
    """One executed run per variant, each echoing what its variant intends
    unless overridden by `overrides[variant] = {"reported_node_continuity": ...}`."""
    base = {
        schema.VARIANT_A: {"reported_node_continuity": "EXPLICIT",
                           "reported_anderson_accel": "NO"},
        schema.VARIANT_B: {"reported_node_continuity": "EXPLICIT",
                           "reported_anderson_accel": "YES"},
        schema.VARIANT_C: {"reported_node_continuity": "SEMI_IMPLICIT",
                           "reported_anderson_accel": "NO"},
    }
    for variant, patch in overrides.items():
        base[variant].update(patch)
    return pd.DataFrame([
        {"model_id": "EPA/m1", "family": "EPA", "variant": variant, **echoes}
        for variant, echoes in base.items()
    ])


def _option_runs_with_surcharge(**overrides):
    """One executed run per variant (A, B, C, D, E), each echoing what its
    variant intends -- including `reported_surcharge_method` -- unless
    overridden by `overrides[variant] = {"reported_surcharge_method": ...}`."""
    base = {
        schema.VARIANT_A: {"reported_node_continuity": "EXPLICIT",
                           "reported_anderson_accel": "NO",
                           "reported_surcharge_method": "EXTRAN"},
        schema.VARIANT_B: {"reported_node_continuity": "EXPLICIT",
                           "reported_anderson_accel": "YES",
                           "reported_surcharge_method": "EXTRAN"},
        schema.VARIANT_C: {"reported_node_continuity": "SEMI_IMPLICIT",
                           "reported_anderson_accel": "NO",
                           "reported_surcharge_method": "EXTRAN"},
        schema.VARIANT_D: {"reported_node_continuity": "SEMI_IMPLICIT",
                           "reported_anderson_accel": "YES",
                           "reported_surcharge_method": "EXTRAN"},
        schema.VARIANT_E: {"reported_node_continuity": "EXPLICIT",
                           "reported_anderson_accel": "NO",
                           "reported_surcharge_method": "DYNAMIC_SLOT"},
    }
    for variant, patch in overrides.items():
        base[variant].update(patch)
    return pd.DataFrame([
        {"model_id": "EPA/m1", "family": "EPA", "variant": variant, **echoes}
        for variant, echoes in base.items()
    ])


def test_option_anomalies_is_empty_across_all_five_variants_when_every_run_echoes_intent():
    anomalies = report.option_anomalies(_option_runs_with_surcharge())

    assert anomalies.empty


def test_option_anomalies_flags_a_d_run_that_echoes_anderson_off():
    # D's value comes from D - C, so its ANDERSON_ACCEL echo matters just
    # as much as B's.
    runs = _option_runs_with_surcharge(D={"reported_anderson_accel": "NO"})

    anomalies = report.option_anomalies(runs)

    assert len(anomalies) == 1
    row = anomalies.iloc[0]
    assert row["variant"] == schema.VARIANT_D
    assert row["option"] == "ANDERSON_ACCEL"
    assert row["expected"] == "YES"
    assert row["reported"] == "NO"


def test_option_anomalies_flags_a_d_run_that_echoes_explicit_node_continuity():
    runs = _option_runs_with_surcharge(D={"reported_node_continuity": "EXPLICIT"})

    anomalies = report.option_anomalies(runs)

    assert len(anomalies) == 1
    row = anomalies.iloc[0]
    assert row["variant"] == schema.VARIANT_D
    assert row["option"] == "NODE_CONTINUITY"
    assert row["expected"] == "SEMI_IMPLICIT"
    assert row["reported"] == "EXPLICIT"


def test_option_anomalies_flags_an_e_run_that_echoes_extran_surcharge_method():
    # E's whole point is the Dynamic Preissmann Slot -- an engine that
    # silently fell back to EXTRAN for E must be caught here.
    runs = _option_runs_with_surcharge(E={"reported_surcharge_method": "EXTRAN"})

    anomalies = report.option_anomalies(runs)

    assert len(anomalies) == 1
    row = anomalies.iloc[0]
    assert row["variant"] == schema.VARIANT_E
    assert row["option"] == "SURCHARGE_METHOD"
    assert row["expected"] == "DYNAMIC_SLOT"
    assert row["reported"] == "EXTRAN"


def test_option_anomalies_ignores_a_non_dynwave_run_missing_all_three_echoes():
    # A KINWAVE (or STEADY/FV) run never prints the DYNWAVE-only block at
    # all: no NODE_CONTINUITY, ANDERSON_ACCEL or SURCHARGE_METHOD echo.
    # Absence must be skipped, not flagged as a contradiction.
    runs = _option_runs_with_surcharge(E={"reported_node_continuity": None,
                                          "reported_anderson_accel": None,
                                          "reported_surcharge_method": None})

    anomalies = report.option_anomalies(runs)

    assert anomalies.empty


def test_option_anomalies_is_empty_when_every_run_echoes_its_intent():
    anomalies = report.option_anomalies(_option_runs())

    assert anomalies.empty


def test_option_anomalies_flags_a_c_run_that_echoes_explicit():
    runs = _option_runs(C={"reported_node_continuity": "EXPLICIT"})

    anomalies = report.option_anomalies(runs)

    assert len(anomalies) == 1
    row = anomalies.iloc[0]
    assert row["model_id"] == "EPA/m1"
    assert row["variant"] == schema.VARIANT_C
    assert row["option"] == "NODE_CONTINUITY"
    assert row["expected"] == "SEMI_IMPLICIT"
    assert row["reported"] == "EXPLICIT"


def test_option_anomalies_flags_a_b_run_that_echoes_anderson_off():
    runs = _option_runs(B={"reported_anderson_accel": "NO"})

    anomalies = report.option_anomalies(runs)

    assert len(anomalies) == 1
    row = anomalies.iloc[0]
    assert row["variant"] == schema.VARIANT_B
    assert row["option"] == "ANDERSON_ACCEL"
    assert row["expected"] == "YES"
    assert row["reported"] == "NO"


def test_option_anomalies_ignores_a_run_with_no_echo():
    # FV/STEADY/KINWAVE routing never prints the block at all -- absence is
    # not a contradiction, there is nothing to check.
    runs = _option_runs(C={"reported_node_continuity": None,
                           "reported_anderson_accel": None})

    anomalies = report.option_anomalies(runs)

    assert anomalies.empty


def test_option_anomalies_on_empty_input_is_an_empty_frame_not_an_error():
    assert report.option_anomalies(pd.DataFrame()).empty
