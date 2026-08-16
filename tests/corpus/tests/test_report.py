import pandas as pd
import pytest

from swmmbench import report, schema


METRICS = ["avg_iterations_per_step", "continuity_error_flow"]


#: Values every fixture run echoes unless a test overrides them. An
#: iteration delta is only formed when BOTH sides are known `DYNWAVE`, and
#: an `X - REF` delta only when both sides echo the same surcharge method,
#: so these echoes are part of the minimum a comparable run must carry.
_DYNWAVE_EXTRAN = {"reported_routing_model": "DYNWAVE",
                   "reported_surcharge_method": "EXTRAN"}


def _runs():
    return pd.DataFrame([
        {"model_id": "EPA/m1", "family": "EPA", "variant": "A",
         "status": "ok", "avg_iterations_per_step": 4.0,
         "continuity_error_flow": 0.10, "iteration_metric_kind": "picard",
         **_DYNWAVE_EXTRAN},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "B",
         "status": "ok", "avg_iterations_per_step": 2.0,
         "continuity_error_flow": 0.12, "iteration_metric_kind": "picard",
         **_DYNWAVE_EXTRAN},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "C",
         "status": "ok", "avg_iterations_per_step": 3.0,
         "continuity_error_flow": 0.11, "iteration_metric_kind": "picard",
         **_DYNWAVE_EXTRAN},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "D",
         "status": "ok", "avg_iterations_per_step": 1.5,
         "continuity_error_flow": 0.13, "iteration_metric_kind": "picard",
         **_DYNWAVE_EXTRAN},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "E",
         "status": "ok", "avg_iterations_per_step": 5.0,
         "continuity_error_flow": 0.14, "iteration_metric_kind": "picard",
         "reported_routing_model": "DYNWAVE",
         "reported_surcharge_method": "DYNAMIC_SLOT"},
        {"model_id": "EPA/m1", "family": "EPA", "variant": "REF",
         "status": "ok", "avg_iterations_per_step": 3.8,
         "continuity_error_flow": 0.09, "iteration_metric_kind": "picard",
         **_DYNWAVE_EXTRAN},
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


# ---------------------------------------------------------------------------
# The D and E axes
# ---------------------------------------------------------------------------


def test_d_minus_c_isolates_anderson_under_the_c1_smooth_operator():
    # The column the whole variant-D exercise exists to produce: C and D
    # differ in ANDERSON_ACCEL alone, so the difference is attributable.
    # Direction is D - C, and the operands are D and C -- not D and A.
    deltas = report.build_deltas(_runs(), METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert row["value_c"] == pytest.approx(3.0)
    assert row["value_d"] == pytest.approx(1.5)
    assert row["delta_d_minus_c"] == pytest.approx(-1.5)
    # The joint axis is kept alongside it and is a different number.
    assert row["delta_d_minus_a"] == pytest.approx(-2.5)


def test_e_minus_a_isolates_the_dynamic_preissmann_slot():
    deltas = report.build_deltas(_runs(), METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert row["value_e"] == pytest.approx(5.0)
    assert row["delta_e_minus_a"] == pytest.approx(1.0)


def test_e_minus_a_survives_the_surcharge_method_guard():
    # E echoes DYNAMIC_SLOT and A echoes EXTRAN, and that difference IS the
    # measurement -- the anchor guard must never reach a non-REF pairing.
    deltas = report.build_deltas(_runs(), METRICS)
    row = deltas[deltas.metric == "continuity_error_flow"].iloc[0]

    assert row["delta_e_minus_a"] == pytest.approx(0.04)


def test_a_model_without_a_d_variant_still_yields_every_other_delta():
    runs = _runs()
    runs = runs[runs.variant != "D"]

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert pd.isna(row["delta_d_minus_a"])
    assert pd.isna(row["delta_d_minus_c"])
    assert row["delta_b_minus_a"] == pytest.approx(-2.0)
    assert row["delta_c_minus_a"] == pytest.approx(-1.0)
    assert row["delta_e_minus_a"] == pytest.approx(1.0)
    assert row["delta_a_minus_ref"] == pytest.approx(0.2)


def test_d_minus_c_survives_a_missing_a_variant():
    # D - C reads C and D only; a crashed baseline must not cost it.
    runs = _runs()
    runs = runs[runs.variant != "A"]

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert row["delta_d_minus_c"] == pytest.approx(-1.5)
    assert pd.isna(row["delta_b_minus_a"])
    assert pd.isna(row["delta_a_minus_ref"])


# ---------------------------------------------------------------------------
# Commensurability: surcharge method against the anchor
# ---------------------------------------------------------------------------


def _guard(runs, metric, left, right, model_id="EPA/m1"):
    """Whether build_deltas would form `left - right` for `metric`.

    Built the way `build_deltas` builds it -- the option echoes come from
    `report.echo_pivots`, the same mapping `option_anomalies` reads -- so the
    helper cannot pass a pairing the real pipeline would reject.
    """
    return report._commensurable(
        metric, model_id, left, right,
        report._pivot(runs, "iteration_metric_kind"),
        report._pivot(runs, "reported_routing_model"),
        report.echo_pivots(runs),
    )


def test_an_e_minus_ref_pairing_is_auto_nulled_by_the_surcharge_guard():
    # There is no `delta_e_minus_ref` column -- parity debt is defined
    # against A -- but the anchor guard is written against the echoes rather
    # than against the letter E, so the pairing is rejected on its own
    # merits: EPA SWMM 5.2 has no Dynamic Preissmann Slot, so REF always
    # echoes EXTRAN while E echoes DYNAMIC_SLOT. Were the column ever added,
    # it would be null by construction rather than by special case.
    runs = _runs()

    assert _guard(runs, "continuity_error_flow",
                  schema.VARIANT_E, schema.VARIANT_REF) is False
    # A shares REF's surcharge method, so its parity debt is unaffected.
    assert _guard(runs, "continuity_error_flow",
                  schema.VARIANT_A, schema.VARIANT_REF) is True


def test_a_minus_ref_is_nulled_when_the_reference_used_another_surcharge_method():
    # Not an E-only rule: a corpus reference run under SLOT is equally
    # incommensurable against our EXTRAN baseline.
    runs = _runs()
    runs.loc[runs.variant == "REF", "reported_surcharge_method"] = "SLOT"

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "continuity_error_flow"].iloc[0]

    assert pd.isna(row["delta_a_minus_ref"])
    # Every delta that does not touch the anchor is untouched.
    assert row["delta_b_minus_a"] == pytest.approx(0.02)
    assert row["delta_d_minus_c"] == pytest.approx(0.02)


def test_a_missing_surcharge_echo_is_not_a_mismatch():
    runs = _runs()
    runs.loc[runs.variant == "REF", "reported_surcharge_method"] = None

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "continuity_error_flow"].iloc[0]

    assert row["delta_a_minus_ref"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Commensurability: the echoed option value against the variant's own intent
# ---------------------------------------------------------------------------


def _anderson_echoes(runs):
    """`_runs()` with every executed row echoing its variant's ANDERSON_ACCEL."""
    runs = runs.copy()
    runs["reported_anderson_accel"] = runs["variant"].map(
        {variant: options["ANDERSON_ACCEL"]
         for variant, options in report.variants.OPTIONS.items()}
    )
    return runs


def test_an_e_run_that_actually_executed_under_extran_publishes_no_delta():
    # The conclusion-inverting failure this screen exists for. The engine
    # silently ignores an unrecognised option value, so a broken E deck runs
    # as plain EXTRAN -- making `E - A` a subtraction of two IDENTICAL
    # configurations, published as ~0 and read as "the Dynamic Preissmann
    # Slot has no effect" when the slot was never enabled at all.
    runs = _runs()
    runs.loc[runs.variant == "E", "reported_surcharge_method"] = "EXTRAN"

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert pd.isna(row["delta_e_minus_a"])
    # Only E's pairings are affected; the rest of the row is untouched, and
    # the row itself survives with its values intact.
    assert row["value_e"] == pytest.approx(5.0)
    assert row["delta_b_minus_a"] == pytest.approx(-2.0)
    assert row["delta_d_minus_c"] == pytest.approx(-1.5)


def test_a_d_run_that_echoes_anderson_off_publishes_no_delta():
    # D's whole value is `D - C`: Anderson under the C1-smooth operator. A D
    # run that actually executed with ANDERSON_ACCEL off differs from C in
    # nothing, so `D - C ~ 0` would read as "Anderson does not help there".
    runs = _anderson_echoes(_runs())
    runs.loc[runs.variant == "D", "reported_anderson_accel"] = "NO"

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert pd.isna(row["delta_d_minus_c"])
    assert pd.isna(row["delta_d_minus_a"])
    # B - A still measures Anderson under EXPLICIT/EXTRAN, and C - A still
    # measures Crank-Nicolson; neither reads D's echo.
    assert row["delta_b_minus_a"] == pytest.approx(-2.0)
    assert row["delta_c_minus_a"] == pytest.approx(-1.0)


def test_the_intent_screen_applies_to_every_metric_not_just_iterations():
    # A misconfigured run corrupts its continuity error exactly as
    # thoroughly as its iteration count.
    runs = _runs()
    runs.loc[runs.variant == "E", "reported_surcharge_method"] = "EXTRAN"

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "continuity_error_flow"].iloc[0]

    assert pd.isna(row["delta_e_minus_a"])
    assert row["delta_b_minus_a"] == pytest.approx(0.02)


def test_a_missing_echo_is_not_a_contradiction():
    # A KINWAVE/STEADY/FV run never prints the DYNWAVE-only Analysis Options
    # block at all. Absence is no evidence, and must not cost a computable
    # delta -- the same rule the kind and anchor-surcharge screens follow.
    runs = _anderson_echoes(_runs())
    runs.loc[runs.variant == "E", "reported_surcharge_method"] = None
    runs.loc[runs.variant == "D", "reported_anderson_accel"] = None

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert row["delta_e_minus_a"] == pytest.approx(1.0)
    assert row["delta_d_minus_c"] == pytest.approx(-1.5)


def test_the_reference_anchor_is_not_screened_against_a_variant_intent():
    # REF is a third-party report the harness never configured, so it has no
    # intent to contradict. Its own confound is the anchor-surcharge screen,
    # which is what must catch a reference run under another method.
    runs = _runs()
    runs.loc[runs.variant == "REF", "reported_surcharge_method"] = "EXTRAN"

    assert _guard(runs, "continuity_error_flow",
                  schema.VARIANT_A, schema.VARIANT_REF) is True


def test_the_intent_screen_and_the_anomaly_report_read_one_mapping():
    # Defence in depth only holds if the two cannot drift: the same
    # OPTION_ECHO_COLUMNS mapping drives the warning and the withholding.
    runs = _runs()
    runs.loc[runs.variant == "E", "reported_surcharge_method"] = "EXTRAN"

    anomalies = report.option_anomalies(runs)

    assert set(anomalies["variant"]) == {schema.VARIANT_E}
    assert pd.isna(report.build_deltas(runs, METRICS)["delta_e_minus_a"]).all()


# ---------------------------------------------------------------------------
# Commensurability: routing model
# ---------------------------------------------------------------------------


def test_a_kinwave_model_contributes_no_iteration_delta():
    # The engine prints `Average Iterations per Step` for KINWAVE too, and
    # both sides would agree on any kind label -- so the kind guard alone
    # lets these straight through. The routing-model requirement is what
    # stops a counter with no Picard meaning being published as one.
    runs = _runs()
    runs["reported_routing_model"] = "KINWAVE"
    runs["iteration_metric_kind"] = None

    deltas = report.build_deltas(runs, METRICS)
    iterations = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]
    continuity = deltas[deltas.metric == "continuity_error_flow"].iloc[0]

    for column in ("delta_b_minus_a", "delta_c_minus_a", "delta_d_minus_a",
                   "delta_d_minus_c", "delta_e_minus_a", "delta_a_minus_ref"):
        assert pd.isna(iterations[column]), column
    # The row survives, and non-iteration metrics are unaffected: routing
    # model has no bearing on a continuity error.
    assert continuity["delta_b_minus_a"] == pytest.approx(0.02)


def test_one_kinwave_side_is_enough_to_null_the_iteration_delta():
    runs = _runs()
    runs.loc[runs.variant == "B", "reported_routing_model"] = "KINWAVE"

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert pd.isna(row["delta_b_minus_a"])
    assert row["delta_c_minus_a"] == pytest.approx(-1.0)


def test_an_unknown_routing_model_nulls_the_iteration_delta():
    # A missing kind is deliberately "not a mismatch"; a missing routing
    # model is NOT given the same benefit of the doubt, because there is
    # then no evidence the counter means Picard iterations at all.
    runs = _runs()
    runs["reported_routing_model"] = None

    deltas = report.build_deltas(runs, METRICS)
    row = deltas[deltas.metric == "avg_iterations_per_step"].iloc[0]

    assert pd.isna(row["delta_b_minus_a"])


def test_pct_steps_not_converging_is_guarded_like_the_iteration_count():
    runs = _runs()
    runs["pct_steps_not_converging"] = 1.0
    runs["reported_routing_model"] = "STEADY"

    deltas = report.build_deltas(runs, ["pct_steps_not_converging"])
    row = deltas.iloc[0]

    assert pd.isna(row["delta_b_minus_a"])


# ---------------------------------------------------------------------------
# Hardness buckets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (0.0, "<= 2"), (1.9999, "<= 2"), (2.0, "<= 2"),
    (2.0001, "2-4"), (3.0, "2-4"), (4.0, "2-4"),
    (4.0001, "> 4"), (12.5, "> 4"),
])
def test_hardness_bucket_boundaries_are_closed_on_the_right(value, expected):
    assert report.hardness_bucket(value) == expected


def test_hardness_buckets_partition_every_finite_value():
    # Exactly one bucket per value: no value in two, none in zero.
    for value in (0.0, 2.0, 2.5, 4.0, 4.5, 100.0):
        matches = [label for label, predicate in report.HARDNESS_BUCKETS
                   if predicate(value)]
        assert matches == [report.hardness_bucket(value)]
        assert len(matches) == 1


def test_a_missing_hardness_value_has_no_bucket():
    assert report.hardness_bucket(None) is None
    assert report.hardness_bucket(pd.NA) is None


# ---------------------------------------------------------------------------
# Stratified markdown sections
# ---------------------------------------------------------------------------


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


def _stratified_runs():
    """Three models: an easy and a hard DYNWAVE one, and a KINWAVE one.

    Each is its own family so a markdown row can be attributed by name.
    Variant A's `avg_iterations_per_step` places the model in its hardness
    bucket: 1.5 -> `<= 2`, 6.0 -> `> 4`, and the KINWAVE model's 3.0 would
    land in `2-4` if it were allowed to participate at all.
    """
    rows = []
    for model_id, family, routing, base in (
        ("EASY/m1", "EASY", "DYNWAVE", 1.5),
        ("HARD/m2", "HARD", "DYNWAVE", 6.0),
        ("KIN/m3", "KIN", "KINWAVE", 3.0),
    ):
        for variant, offset in (("A", 0.0), ("B", -0.5), ("C", -0.2),
                                ("D", -0.8), ("E", 0.3), ("REF", 0.1)):
            rows.append({
                "model_id": model_id, "family": family, "variant": variant,
                "status": "ok",
                "avg_iterations_per_step": base + offset,
                "continuity_error_flow": 0.1 + offset / 100.0,
                "pct_steps_not_converging": 0.0,
                "iteration_metric_kind": "picard" if routing == "DYNWAVE" else None,
                "reported_routing_model": routing,
                "reported_surcharge_method":
                    "DYNAMIC_SLOT" if variant == "E" else "EXTRAN",
            })
    return pd.DataFrame(rows)


def _write_stratified(tmp_path, lang="en", runs=None, elements=None):
    runs = _stratified_runs() if runs is None else runs
    deltas = report.build_deltas(runs, METRICS)
    path = report.write_markdown(deltas, runs, tmp_path / f"summary_{lang}.md",
                                 lang=lang, elements=elements)
    return path.read_text(encoding="utf-8")


def test_iteration_sections_carry_only_dynwave_rows(tmp_path):
    # KINWAVE's `Average Iterations per Step` counts nothing Picard-shaped;
    # pooling it into these medians makes them describe a quantity nobody
    # named. The other sections are unaffected -- KIN/m3's continuity
    # deltas are perfectly comparable.
    text = _write_stratified(tmp_path)
    strings = report.MARKDOWN_STRINGS["en"]

    for key in ("iter_shift_b_heading", "iter_shift_c_heading",
                "iter_shift_d_heading", "iter_shift_e_heading"):
        section = _markdown_section(text, strings[key])
        assert "| EASY | " in section
        assert "| HARD | " in section
        assert "KIN" not in section

    # ... but KIN/m3 still contributes to the per-metric tables, where its
    # iteration delta is null and its continuity delta is not.
    b_section = _markdown_section(text, strings["b_minus_a_heading"])
    assert "| continuity_error_flow | 3 |" in b_section
    assert "| avg_iterations_per_step | 2 |" in b_section


def test_iteration_sections_are_split_by_hardness_bucket(tmp_path):
    text = _write_stratified(tmp_path)
    section = _markdown_section(
        text, report.MARKDOWN_STRINGS["en"]["iter_shift_b_heading"])

    # EASY/m1's A run averages 1.5 iterations, HARD/m2's 6.0.
    assert "| EASY | <= 2 | 1 | 0.5000 |" in section
    assert "| HARD | > 4 | 1 | 0.5000 |" in section
    assert "2-4" not in section


def test_hardness_buckets_partition_the_models_in_the_report(tmp_path):
    text = _write_stratified(tmp_path)
    section = _markdown_section(
        text, report.MARKDOWN_STRINGS["en"]["iter_shift_b_heading"])

    counted = sum(int(line.split("|")[3]) for line in section.splitlines()
                  if line.startswith("| ") and "---" not in line
                  and "models" not in line)

    # Every DYNWAVE model appears exactly once across the buckets.
    assert counted == 2


# ---------------------------------------------------------------------------
# Withheld iteration deltas are counted, not silently absent
# ---------------------------------------------------------------------------


def _exclusions(runs):
    return report.iteration_exclusions(
        report.build_deltas(runs, METRICS), runs
    ).set_index("cause")


def test_a_store_with_no_routing_model_reports_unknown_provenance(tmp_path):
    # The strictness of the routing guard must not be silent: a store
    # predating `reported_routing_model` yields zero iteration deltas, and
    # empty sections read as "the feature has no effect" -- the exact
    # misreading the surcharge stratification exists to prevent.
    runs = _runs()
    runs["reported_routing_model"] = None

    counts = _exclusions(runs)

    assert report.EXCLUSION_NOT_DYNWAVE not in counts.index
    # Six axes for the one model, every one of them withheld.
    assert counts.loc[report.EXCLUSION_ROUTING_UNKNOWN, "pairings"] == 6
    assert counts.loc[report.EXCLUSION_ROUTING_UNKNOWN, "models"] == 1

    for lang in ("en", "es"):
        text = _write_stratified(tmp_path, lang=lang, runs=runs)
        strings = report.MARKDOWN_STRINGS[lang]
        section = _markdown_section(text, strings["iter_excluded_heading"])

        assert f"| {report.EXCLUSION_ROUTING_UNKNOWN} | 6 | 1 |" in section
        assert "\n".join(strings["iter_unknown_note"]) in section
        assert "\n".join(strings["iter_excluded_none"]) not in section


def test_a_kinwave_model_is_counted_as_a_healthy_exclusion(tmp_path):
    # `_stratified_runs` has two DYNWAVE models and one KINWAVE one. The
    # KINWAVE model loses all six axes; nothing is unknown.
    runs = _stratified_runs()

    counts = _exclusions(runs)

    assert counts.loc[report.EXCLUSION_NOT_DYNWAVE, "pairings"] == 6
    assert counts.loc[report.EXCLUSION_NOT_DYNWAVE, "models"] == 1
    assert report.EXCLUSION_ROUTING_UNKNOWN not in counts.index

    text = _write_stratified(tmp_path, runs=runs)
    strings = report.MARKDOWN_STRINGS["en"]
    section = _markdown_section(text, strings["iter_excluded_heading"])

    # An expected exclusion must not raise the actionable alarm.
    assert f"| {report.EXCLUSION_NOT_DYNWAVE} | 6 | 1 |" in section
    assert "\n".join(strings["iter_unknown_note"]) not in section
    assert "\n".join(strings["iter_empty_note"]) not in section


def test_the_two_exclusion_causes_are_reported_separately(tmp_path):
    # One KINWAVE model (healthy) and one model with no routing echo at all
    # (actionable). Collapsing them into a single count would hide the
    # second inside the first.
    runs = _stratified_runs()
    runs.loc[runs.model_id == "EASY/m1", "reported_routing_model"] = None

    counts = _exclusions(runs)

    assert counts.loc[report.EXCLUSION_NOT_DYNWAVE, "pairings"] == 6
    assert counts.loc[report.EXCLUSION_ROUTING_UNKNOWN, "pairings"] == 6
    assert set(counts["models"]) == {1}

    for lang in ("en", "es"):
        text = _write_stratified(tmp_path, lang=lang, runs=runs)
        strings = report.MARKDOWN_STRINGS[lang]
        section = _markdown_section(text, strings["iter_excluded_heading"])

        assert f"| {report.EXCLUSION_NOT_DYNWAVE} | 6 | 1 |" in section
        assert f"| {report.EXCLUSION_ROUTING_UNKNOWN} | 6 | 1 |" in section
        assert "\n".join(strings["iter_unknown_note"]) in section


def test_entirely_empty_iteration_sections_say_so(tmp_path):
    runs = _runs()
    runs["reported_routing_model"] = None

    for lang in ("en", "es"):
        text = _write_stratified(tmp_path, lang=lang, runs=runs)
        strings = report.MARKDOWN_STRINGS[lang]
        section = _markdown_section(text, strings["iter_excluded_heading"])

        assert "\n".join(strings["iter_empty_note"]) in section
        # The iteration sections themselves carry no data row.
        for key in ("iter_shift_b_heading", "iter_shift_d_heading"):
            body = _markdown_section(text, strings[key])
            rows = [line for line in body.splitlines()
                    if line.startswith("| ") and "---" not in line]
            assert rows == [strings["family_header"]]


def test_a_fully_measurable_store_reports_no_withheld_pairings(tmp_path):
    text = _write_stratified(tmp_path, runs=_runs())
    strings = report.MARKDOWN_STRINGS["en"]
    section = _markdown_section(text, strings["iter_excluded_heading"])

    assert _exclusions(_runs()).empty
    assert "\n".join(strings["iter_excluded_none"]) in section
    assert "\n".join(strings["iter_empty_note"]) not in section


def test_a_missing_operand_is_not_counted_as_a_routing_exclusion():
    # A crashed variant was never withheld by the routing guard; counting it
    # would blame provenance for an absence with a different cause.
    runs = _runs()
    runs = runs[runs.variant != "D"]

    assert _exclusions(runs).empty


def test_a_kind_mismatch_between_two_dynwave_runs_is_not_a_routing_exclusion():
    # It has its own caveat; attributing it to routing would double-count.
    runs = _runs()
    runs.loc[runs.variant == "B", "iteration_metric_kind"] = schema.ITER_FV

    assert _exclusions(runs).empty


def test_exclusions_on_an_empty_frame_are_an_empty_table_not_an_error():
    assert report.iteration_exclusions(pd.DataFrame(), pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# A computed delta with no baseline to stratify it by
# ---------------------------------------------------------------------------


def test_a_d_minus_c_whose_a_run_crashed_is_neither_dropped_nor_unaccounted(tmp_path):
    # `D - C` reads C and D only, so a crashed baseline still leaves a valid,
    # non-null delta. Bucketing by `value_a` alone used to drop it from the
    # D - C table AND from the exclusion ledger (which counts null deltas
    # only), so a real measurement vanished from both.
    runs = _runs()
    runs = runs[runs.variant != "A"]

    counts = _exclusions(runs)

    assert counts.loc[report.EXCLUSION_NO_BASELINE, "pairings"] == 1
    assert counts.loc[report.EXCLUSION_NO_BASELINE, "models"] == 1

    for lang in ("en", "es"):
        text = _write_stratified(tmp_path, lang=lang, runs=runs)
        strings = report.MARKDOWN_STRINGS[lang]

        # Kept in the table too, under an explicit `unknown` hardness bucket
        # rather than filed under a stratum it did not earn.
        section = _markdown_section(text, strings["iter_shift_d_heading"])
        assert f"| EPA | {report.HARDNESS_UNKNOWN} | 1 | 1.5000 |" in section

        ledger = _markdown_section(text, strings["iter_excluded_heading"])
        assert f"| {report.EXCLUSION_NO_BASELINE} | 1 | 1 |" in ledger
        # Something WAS measured, so the nothing-survived note must not fire.
        assert "\n".join(strings["iter_empty_note"]) not in ledger


def test_a_kinwave_baseline_does_not_lend_its_iteration_count_as_a_bucket():
    # A KINWAVE A run still prints an `Average Iterations per Step` line;
    # stratifying on it would sort the model by a number that counts nothing
    # Picard-shaped. Such a model is `unknown`, not `2-4`.
    runs = _runs()
    runs.loc[runs.variant == "A", "reported_routing_model"] = "KINWAVE"
    deltas = report.build_deltas(runs, METRICS)
    iterations = deltas[deltas.metric == "avg_iterations_per_step"]

    hardness = report.reported_hardness(iterations, report.dynwave_models(runs))

    assert list(hardness) == [report.HARDNESS_UNKNOWN]


def test_a_stratifiable_model_is_not_counted_as_missing_a_baseline():
    assert report.EXCLUSION_NO_BASELINE not in _exclusions(_runs()).index


# ---------------------------------------------------------------------------
# Surcharge activity
# ---------------------------------------------------------------------------


def _flooding_elements(model_id="HARD/m2", value=12.0):
    return pd.DataFrame([{
        "model_id": model_id, "family": model_id.split("/")[0],
        "variant": "A", "element_type": "NODE", "element_id": "J1",
        "metric": "node_total_flood_volume", "value": value,
    }])


def test_no_surcharge_active_model_is_stated_rather_than_averaged_to_zero(tmp_path):
    # The distinction this whole stratification exists for: "the feature was
    # not exercised" is not "the feature has no effect", and a near-zero
    # mean in a markdown table cannot tell a reader which one happened.
    for lang in ("en", "es"):
        text = _write_stratified(tmp_path, lang=lang)
        strings = report.MARKDOWN_STRINGS[lang]
        section = _markdown_section(text, strings["surcharge_heading"])

        assert "\n".join(strings["surcharge_none"]) in section
        assert f"| {report.SURCHARGE_ACTIVE} |" not in section
        assert f"| {report.SURCHARGE_INACTIVE} |" in section


def test_a_tiny_surcharge_active_subset_is_named_as_anecdote(tmp_path):
    text = _write_stratified(tmp_path, elements=_flooding_elements())
    strings = report.MARKDOWN_STRINGS["en"]
    section = _markdown_section(text, strings["surcharge_heading"])

    assert "\n".join(strings["surcharge_few"]).format(n=1) in section
    assert "\n".join(strings["surcharge_none"]) not in section
    assert f"| D - C | {report.SURCHARGE_ACTIVE} |" in section


def test_surcharge_activity_is_read_from_the_baseline_run(tmp_path):
    runs = _stratified_runs()
    runs.loc[(runs.model_id == "HARD/m2") & (runs.variant == "A"),
             "pct_steps_not_converging"] = 4.2

    assert report.surcharge_active_models(runs) == {"HARD/m2"}

    # A non-baseline variant's non-convergence must not move the stratum:
    # the stratum would then depend on the very feature being measured.
    other = _stratified_runs()
    other.loc[(other.model_id == "HARD/m2") & (other.variant == "D"),
              "pct_steps_not_converging"] = 4.2

    assert report.surcharge_active_models(other) == set()


def _one_active_model_without_a_c_run():
    """A surcharge-active model whose C run crashed.

    `pct_steps_not_converging > 0` on variant A puts it in the `active`
    stratum, but with no C row there is no `D - C` delta to put in it -- so
    that axis is empty while `D - A` and `E - A` are fully populated.
    """
    rows = []
    for variant, offset in (("A", 0.0), ("B", -0.5), ("D", -0.8), ("E", 0.3)):
        rows.append({
            "model_id": "SUR/m1", "family": "SUR", "variant": variant,
            "status": "ok",
            "avg_iterations_per_step": 6.0 + offset,
            "continuity_error_flow": 0.1 + offset / 100.0,
            "pct_steps_not_converging": 2.0 if variant == "A" else 0.0,
            "iteration_metric_kind": "picard",
            "reported_routing_model": "DYNWAVE",
            "reported_surcharge_method":
                "DYNAMIC_SLOT" if variant == "E" else "EXTRAN",
        })
    return pd.DataFrame(rows)


def test_an_axis_with_no_active_row_says_so_even_when_another_axis_has_rows(
    tmp_path,
):
    # The global note fires only when the active set itself is empty. A store
    # with surcharge-active models can still produce no `active` row for one
    # axis -- and that axis would then print `inactive` rows alone, which is
    # the exact misreading the announcement exists to block.
    runs = _one_active_model_without_a_c_run()

    for lang in ("en", "es"):
        text = _write_stratified(tmp_path, lang=lang, runs=runs)
        strings = report.MARKDOWN_STRINGS[lang]
        section = _markdown_section(text, strings["surcharge_heading"])
        note = "\n".join(strings["surcharge_axis_empty"])

        assert note.format(axis="D - C") in section
        # The other two axes are populated, so their notes must NOT fire.
        assert f"| D - A | {report.SURCHARGE_ACTIVE} |" in section
        assert f"| E - A | {report.SURCHARGE_ACTIVE} |" in section
        assert note.format(axis="D - A") not in section
        assert note.format(axis="E - A") not in section
        # The store HAS active models, so the global "none" note is wrong here.
        assert "\n".join(strings["surcharge_none"]) not in section


def test_the_per_axis_note_does_not_repeat_the_global_one(tmp_path):
    # With no surcharge-active model at all, the global note already says it
    # once; three per-axis repetitions would bury it.
    text = _write_stratified(tmp_path)
    strings = report.MARKDOWN_STRINGS["en"]
    section = _markdown_section(text, strings["surcharge_heading"])

    assert "\n".join(strings["surcharge_none"]) in section
    for axis, _column in report.SURCHARGE_SECTIONS:
        assert "\n".join(strings["surcharge_axis_empty"]).format(
            axis=axis) not in section


def test_flooding_elements_widen_the_surcharge_proxy():
    runs = _stratified_runs()

    assert report.surcharge_active_models(runs) == set()
    assert report.surcharge_active_models(
        runs, _flooding_elements()) == {"HARD/m2"}
    # A zero flood volume is not activity.
    assert report.surcharge_active_models(
        runs, _flooding_elements(value=0.0)) == set()


def test_empty_input_produces_an_empty_frame_not_an_error():
    assert report.build_deltas(pd.DataFrame(), METRICS).empty


# ---------------------------------------------------------------------------
# The anomaly section reaches the deliverable, not just the console
# ---------------------------------------------------------------------------


def test_the_summary_counts_the_runs_whose_echo_contradicts_intent(tmp_path):
    # `summary.md` is the artifact people keep and share; a console warning
    # is seen once, by one operator. The one signal that catches a silently
    # ignored option must survive into the document.
    runs = _runs()
    runs.loc[runs.variant == "E", "reported_surcharge_method"] = "EXTRAN"

    for lang in ("en", "es"):
        text = _write_stratified(tmp_path, lang=lang, runs=runs)
        strings = report.MARKDOWN_STRINGS[lang]
        section = _markdown_section(text, strings["anomaly_heading"])

        # Broken down by variant AND option: which deck failed, and which
        # option the engine actually resolved differently.
        assert f"| {schema.VARIANT_E} | SURCHARGE_METHOD | 1 | 1 |" in section
        assert "\n".join(strings["anomaly_total"]).format(n=1, n_models=1) in section
        assert "\n".join(strings["anomaly_none"]) not in section


def test_a_store_with_no_anomaly_says_so_rather_than_omitting_the_section(tmp_path):
    # An absent section is indistinguishable from a harness that never
    # looked, so the clean case is stated outright.
    for lang in ("en", "es"):
        text = _write_stratified(tmp_path, lang=lang, runs=_runs())
        strings = report.MARKDOWN_STRINGS[lang]
        section = _markdown_section(text, strings["anomaly_heading"])

        assert "\n".join(strings["anomaly_none"]) in section
        assert "SURCHARGE_METHOD" not in section


def test_the_anomaly_summary_folds_many_models_into_one_broken_deck():
    anomalies = pd.DataFrame([
        {"model_id": f"EPA/m{n}", "variant": schema.VARIANT_E,
         "option": "SURCHARGE_METHOD", "expected": "DYNAMIC_SLOT",
         "reported": "EXTRAN"}
        for n in range(3)
    ])

    summary = report.anomaly_summary(anomalies)

    assert len(summary) == 1
    assert summary.iloc[0]["runs"] == 3
    assert summary.iloc[0]["models"] == 3


def test_the_anomaly_summary_of_an_empty_frame_is_empty_not_an_error():
    assert report.anomaly_summary(pd.DataFrame()).empty


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
