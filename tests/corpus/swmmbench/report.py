"""Materialise the comparison axes and a human-readable summary.

`B - A` isolates Anderson acceleration under EXPLICIT/EXTRAN. `C - A`
isolates semi-implicit (Crank-Nicolson) node continuity. `D - C` isolates
Anderson acceleration *under* that C1-smooth operator -- the regime B cannot
exercise, because `DWSolver::computeAASkipFlags` disables Anderson at every
surcharged node under EXPLICIT/EXTRAN -- and `D - A` is the joint
Anderson + Crank-Nicolson effect. `E - A` isolates the Dynamic Preissmann
Slot. `A - REF` measures parity debt against EPA SWMM 5.2. Keeping all of
them side by side rather than collapsed is the reason the A/B/C/D/E matrix
exists: folding two feature changes into one variant would make an observed
shift unattributable between their two causes.

A/B/C/D is additionally a full 2x2 factorial over ANDERSON_ACCEL and
NODE_CONTINUITY, so it supports one measurement none of the simple effects
above can give: the INTERACTION, `(D - C) - (B - A)`. See
`INTERACTION_SPEC`.

Unlike the time-series stage (`cli.stage_diff`), `D - C` here is a plain
linear subtraction of two scalars, so it is formed from the same pivot as
every other axis rather than needing its own comparison pass.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema, variants

DEFAULT_METRICS = [
    "avg_iterations_per_step",
    "pct_steps_not_converging",
    "continuity_error_runoff",
    "continuity_error_flow",
    "continuity_error_quality",
    "avg_step",
    "wall_ms",
]

#: Metrics whose comparability depends on iteration_metric_kind matching AND
#: on both operands having been routed by the dynamic wave solver.
ITERATION_METRICS = {"avg_iterations_per_step", "pct_steps_not_converging"}

#: The only routing model under which `Average Iterations per Step` counts
#: Picard iterations of the dynamic-wave solver. The engine prints that line
#: for KINWAVE and STEADY too (DefaultReportPlugin.cpp), where the counter
#: has no Picard meaning at all -- so an iteration delta is only formed when
#: BOTH operands are known to be DYNWAVE.
DYNWAVE = schema.ROUTING_DYNWAVE

#: Every variant that can carry a value on a delta row, in report order. The
#: emitted column is `value_<lowercased variant>` -- `value_a` .. `value_e`
#: and `value_ref`.
DELTA_VARIANTS = (
    schema.VARIANT_A, schema.VARIANT_B, schema.VARIANT_C,
    schema.VARIANT_D, schema.VARIANT_E, schema.VARIANT_REF,
)

#: `(column, left, right)` for every delta emitted, computed as
#: `value_left - value_right`.
#:
#: `delta_d_minus_c` is the column the whole variant-D exercise exists to
#: produce: with NODE_CONTINUITY held at SEMI_IMPLICIT on both sides, the
#: only key that moves between C and D is ANDERSON_ACCEL, so the difference
#: is attributable to Anderson alone -- in the surcharge regime B cannot
#: reach. `delta_d_minus_a` is deliberately kept alongside it and is NOT a
#: substitute: it moves two keys at once and is a joint effect.
DELTA_SPECS = (
    ("delta_b_minus_a", schema.VARIANT_B, schema.VARIANT_A),
    ("delta_c_minus_a", schema.VARIANT_C, schema.VARIANT_A),
    ("delta_d_minus_a", schema.VARIANT_D, schema.VARIANT_A),
    ("delta_d_minus_c", schema.VARIANT_D, schema.VARIANT_C),
    ("delta_e_minus_a", schema.VARIANT_E, schema.VARIANT_A),
    ("delta_a_minus_ref", schema.VARIANT_A, schema.VARIANT_REF),
)

#: The 2x2 factorial interaction between the two factors A/B/C/D crosses --
#: Anderson acceleration and node continuity -- and the two simple effects it
#: is formed from, as `(column, anderson_under_semi_implicit,
#: anderson_under_explicit)`:
#:
#:     delta_interaction = (D - C) - (B - A)  ==  D - C - B + A
#:
#: A/B/C/D is a full 2x2 design (ANDERSON_ACCEL x NODE_CONTINUITY), and the
#: five simple effects above are all it publishes. None of them answers the
#: question the design exists to ask: *does switching to semi-implicit
#: continuity change how effective Anderson acceleration is?* `B - A` measures
#: Anderson under EXPLICIT, `D - C` measures it under SEMI_IMPLICIT, and only
#: their difference measures whether the operator changed the answer.
#:
#: It is expected to be non-zero, which is exactly why it must be quantified
#: rather than assumed: `DWSolver::computeAASkipFlags` disables Anderson at
#: every surcharged node under EXPLICIT/EXTRAN and not under SEMI_IMPLICIT, so
#: the two simple effects are measurements of two different regimes. A
#: near-zero interaction would be evidence that the skip flags do not matter
#: on this corpus -- a finding in its own right, and one no single axis can
#: report.
#:
#: Derived from the two DELTA_SPECS columns rather than from the four raw
#: values, deliberately: each of those columns is already null when its own
#: operand is missing or when `_commensurable` rejects its pairing, so the
#: interaction inherits both rules by construction. A fifth screen written
#: here could drift from the four it is supposed to agree with; a case the
#: screens reject cannot slip in through the interaction because the
#: interaction never sees the raw values at all.
INTERACTION_COLUMN = "delta_interaction"
INTERACTION_SPEC = (INTERACTION_COLUMN, "delta_d_minus_c", "delta_b_minus_a")

#: Axis notation for the interaction, in the same register as `B - A` and
#: `D - C`: data, not prose, and verbatim in both languages.
INTERACTION_AXIS = "(D - C) - (B - A)"

DELTA_COLUMNS = [column for column, _, _ in DELTA_SPECS] + [INTERACTION_COLUMN]
VALUE_COLUMNS = [f"value_{variant.lower()}" for variant in DELTA_VARIANTS]

#: The `case_id` of each variant's source run, carried through to `deltas`
#: and `element_deltas` -- one column per variant, exactly parallel to
#: `VALUE_COLUMNS`.
#:
#: These tables are recomputed and replaced on every `report`, so without the
#: provenance a published number is traceable only as far as "some row of
#: `runs`". With it, every `value_e` and every `delta_e_minus_a` names the
#: exact executable and the exact corpus state that produced it. One column
#: per variant rather than one per row because a delta row is a JOIN across
#: six runs: a single `case_id` would have to pick a side, and the side it
#: picked would be the one nobody was asking about.
CASE_ID_COLUMNS = [f"case_id_{variant.lower()}" for variant in DELTA_VARIANTS]


def _pivot(runs: pd.DataFrame, column: str) -> pd.DataFrame:
    """model_id x variant table of `column`, or an empty frame if absent."""
    if column not in runs.columns:
        return pd.DataFrame()
    try:
        return runs.pivot_table(index="model_id", columns="variant",
                                values=column, aggfunc="first")
    except (KeyError, ValueError):
        return pd.DataFrame()


def _lookup(table: pd.DataFrame, model_id, variant: str):
    """`table[model_id, variant]`, or `pd.NA` when either axis is missing."""
    if table.empty or variant not in table.columns:
        return pd.NA
    if model_id not in table.index:
        return pd.NA
    return table.loc[model_id, variant]


def _same(left, right) -> bool | None:
    """Case-insensitive echo comparison. None when either side is unknown.

    A missing echo is deliberately NOT a mismatch: a variant that crashed,
    or a report that never printed the line, must not cost the model an
    otherwise-computable delta.
    """
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return None
    return str(left).strip().upper() == str(right).strip().upper()


#: Options whose engine-reported value can be checked against the variant's
#: stated intent, keyed by the `runs` column that carries the echo
#: (DefaultReportPlugin.cpp prints all three under the DYNWAVE-only Analysis
#: Options block; rptparse.parse_scalars reads them into these columns).
#: OptionsHandler.cpp silently ignores an unrecognised value for any of them
#: -- no `else`, no warning -- so `options_applied` (what the harness's deck
#: asked for) cannot alone prove the engine actually did it. Without this
#: check, a typo in an option name or value would read as "the feature has
#: no effect" across the whole corpus instead of as a broken deck.
#:
#: Defined here rather than beside `option_anomalies` because BOTH consume
#: it: `option_anomalies` reports the contradiction to the operator, and
#: `_commensurable` withholds the deltas computed from a contradicting run.
#: One mapping, so a column renamed on one side cannot drift from the other.
OPTION_ECHO_COLUMNS = {
    "NODE_CONTINUITY": "reported_node_continuity",
    "ANDERSON_ACCEL": "reported_anderson_accel",
    "SURCHARGE_METHOD": "reported_surcharge_method",
}

#: The echo column the anchor-surcharge screen reads. Named through the
#: mapping rather than spelled a second time.
SURCHARGE_ECHO_COLUMN = OPTION_ECHO_COLUMNS["SURCHARGE_METHOD"]


def echo_pivots(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """model_id x variant tables for every option echo, keyed by column."""
    return {column: _pivot(runs, column)
            for column in OPTION_ECHO_COLUMNS.values()}


def _contradicts_intent(model_id, variant: str,
                        echoes: dict[str, pd.DataFrame]) -> bool:
    """True when `variant`'s run echoed a value its own deck did not ask for.

    The same mapping and the same comparison `option_anomalies` uses, so the
    warning it prints and the deltas withheld here can never disagree.

    REF is not screened: it is a third-party report the harness never
    configured, so it has no intent to contradict. Its own confound -- a
    reference run under a different surcharge method -- is what the
    anchor-surcharge screen exists for.

    A MISSING echo is never a contradiction: a KINWAVE, STEADY or FV run
    never prints the DYNWAVE-only Analysis Options block at all, and a run
    that failed before the block was written prints nothing either. Absence
    is no evidence, and must not cost a computable delta.
    """
    intended = variants.OPTIONS.get(variant)
    if not intended:
        return False
    for option, column in OPTION_ECHO_COLUMNS.items():
        expected = intended.get(option)
        if expected is None:
            continue
        reported = _lookup(echoes.get(column, pd.DataFrame()), model_id, variant)
        if reported is None or pd.isna(reported):
            continue
        if str(reported).strip().upper() != str(expected).strip().upper():
            return True
    return False


def _commensurable(
    metric: str,
    model_id,
    left: str,
    right: str,
    kinds: pd.DataFrame,
    routing: pd.DataFrame,
    echoes: dict[str, pd.DataFrame],
) -> bool:
    """True when `left - right` is a subtraction of two like quantities.

    Four confounds are screened, each per-delta rather than per-row so one
    bad pairing never costs an unrelated computable one:

    1. **Iteration counter kind.** Picard iterations and FV substeps are
       different counters; subtracting one from the other is meaningless.
       Only trips when BOTH kinds are known and differ.
    2. **Routing model.** The engine prints `Average Iterations per Step`
       for EVERY routing model and relabels it only under FV, so a KINWAVE
       or STEADY run carries a counter with no Picard meaning while still
       looking like one. Checked EXPLICITLY rather than left to (1): a
       missing kind is deliberately treated as "not a mismatch", and two
       KINWAVE runs agree on their (meaningless) kind anyway. Both operands
       must be known DYNWAVE.
    3. **Surcharge method, against the anchor.** Applied only when REF is
       one of the operands. E is DYNAMIC_SLOT and EPA SWMM 5.2 has no
       Dynamic Preissmann Slot, so its reports are always EXTRAN -- but the
       rule is written against the echoes, not against E, so it also catches
       a corpus reference that happens to have been run under a different
       surcharge method than ours. It must NOT apply to `E - A`: that delta
       moving the surcharge method is the entire point of variant E.
    4. **Echoed option value against the variant's own intent.** Defence in
       depth on top of the `option_anomalies` warning, which only prints.
       OptionsHandler.cpp silently ignores an unrecognised option value, so a
       broken E deck runs as plain EXTRAN -- and `delta_e_minus_a` would then
       be a subtraction of two IDENTICAL configurations, published as ~0 and
       read as "the Dynamic Preissmann Slot has no effect" when the truth is
       "the slot was never enabled". A conclusion-inverting failure, so the
       delta is withheld rather than published. Applies to every metric, not
       only the iteration ones: a misconfigured run corrupts its continuity
       error just as thoroughly. A missing echo is not a contradiction, and
       REF has no intent to contradict (see `_contradicts_intent`).
    """
    if metric in ITERATION_METRICS:
        if _same(_lookup(kinds, model_id, left),
                 _lookup(kinds, model_id, right)) is False:
            return False
        for variant in (left, right):
            model = _lookup(routing, model_id, variant)
            if model is None or pd.isna(model):
                return False
            if str(model).strip().upper() != DYNWAVE:
                return False

    if schema.VARIANT_REF in (left, right):
        surcharge = echoes.get(SURCHARGE_ECHO_COLUMN, pd.DataFrame())
        if _same(_lookup(surcharge, model_id, left),
                 _lookup(surcharge, model_id, right)) is False:
            return False

    for variant in (left, right):
        if _contradicts_intent(model_id, variant, echoes):
            return False

    return True


def _interaction(row: dict) -> object:
    """`(D - C) - (B - A)` for one delta row, or `pd.NA`.

    Reads the two already-computed simple effects rather than the four raw
    values, so it is null exactly when either constituent pairing is -- an
    operand missing, or the pairing rejected by a commensurability screen --
    with no screen of its own to drift out of step with them. See
    `INTERACTION_SPEC`.
    """
    _column, minuend, subtrahend = INTERACTION_SPEC
    left, right = row.get(minuend), row.get(subtrahend)
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return pd.NA
    return left - right


def build_deltas(runs: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """One row per (model, metric) with A..E and REF values and all deltas.

    A row is always emitted, never dropped; an individual delta is null when
    either operand is missing or when `_commensurable` rejects the pairing.
    `delta_interaction` follows the same two rules over all four of its
    operands, by being derived from the two deltas that already enforce them.
    """
    if runs.empty:
        return pd.DataFrame()

    present = [m for m in metrics if m in runs.columns]
    if not present:
        return pd.DataFrame()

    kinds = _pivot(runs, "iteration_metric_kind")
    routing = _pivot(runs, "reported_routing_model")
    echoes = echo_pivots(runs)
    # Empty for a store written before `case_id` existed; `_lookup` then
    # yields NA and the columns are still emitted, so the schema does not
    # depend on the age of the store.
    cases = _pivot(runs, "case_id")

    families = runs.groupby("model_id")["family"].first()
    rows: list[dict] = []

    for metric in present:
        wide = runs.pivot_table(index="model_id", columns="variant",
                                values=metric, aggfunc="first")
        for model_id, record in wide.iterrows():
            values = {variant: record.get(variant) for variant in DELTA_VARIANTS}

            row = {
                "model_id": model_id,
                "family": families.get(model_id),
                "metric": metric,
            }
            row.update({f"value_{variant.lower()}": values[variant]
                        for variant in DELTA_VARIANTS})
            row.update({f"case_id_{variant.lower()}":
                        _lookup(cases, model_id, variant)
                        for variant in DELTA_VARIANTS})

            for column, left, right in DELTA_SPECS:
                value_left, value_right = values[left], values[right]
                usable = (pd.notna(value_left) and pd.notna(value_right)
                          and _commensurable(metric, model_id, left, right,
                                             kinds, routing, echoes))
                row[column] = (value_left - value_right) if usable else pd.NA

            row[INTERACTION_COLUMN] = _interaction(row)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Engine-echoed option values vs. variant intent
# ---------------------------------------------------------------------------
#
# `OPTION_ECHO_COLUMNS` is defined above, next to `_commensurable`: the same
# mapping drives both the operator-facing report below and the withholding of
# any delta computed from a contradicting run.

ANOMALY_COLUMNS = ["model_id", "variant", "option", "expected", "reported"]


def option_anomalies(runs: pd.DataFrame) -> pd.DataFrame:
    """Executed runs whose engine-reported option value contradicts intent.

    Every executed variant is checked against `variants.OPTIONS`, so D's
    `ANDERSON_ACCEL YES` / `NODE_CONTINUITY SEMI_IMPLICIT` and E's
    `SURCHARGE_METHOD DYNAMIC_SLOT` are covered by the same loop as A/B/C --
    which matters most for E, whose entire value is the Dynamic Preissmann
    Slot: an engine that silently fell back to EXTRAN would otherwise
    publish `E - A ~ 0` as "the slot has no effect".

    A row with no echo (FV/STEADY/KINWAVE routing never prints the block at
    all) is not an anomaly: there is nothing to contradict, so it is silently
    skipped rather than flagged.

    Reporting is only half the defence: `_commensurable` reads the same
    `OPTION_ECHO_COLUMNS` mapping and withholds every delta computed from a
    run listed here, so a silently-ignored option cannot be published as a
    near-zero effect while this table names it.
    """
    if runs.empty:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    executed = runs[runs["variant"].isin(schema.VARIANTS)]
    rows: list[dict] = []

    for option, echo_column in OPTION_ECHO_COLUMNS.items():
        if echo_column not in executed.columns:
            continue
        for _, row in executed.iterrows():
            reported = row.get(echo_column)
            # `pd.isna` alone covers None, float NaN, NaT and `pd.NA` -- the
            # narrower `isinstance(reported, float)` guard missed `pd.NA`,
            # which would stringify to "<NA>" and flag a false anomaly on
            # every row whose echo column happened to hold it.
            if reported is None or pd.isna(reported):
                continue
            expected = variants.OPTIONS.get(row["variant"], {}).get(option)
            if expected is None:
                continue
            if str(reported).strip().upper() != str(expected).strip().upper():
                rows.append({
                    "model_id": row["model_id"], "variant": row["variant"],
                    "option": option, "expected": expected, "reported": reported,
                })

    return pd.DataFrame(rows, columns=ANOMALY_COLUMNS)


ANOMALY_SUMMARY_COLUMNS = ["variant", "option", "runs", "models"]


def anomaly_summary(anomalies: pd.DataFrame) -> pd.DataFrame:
    """`option_anomalies` folded to one row per (variant, option).

    Broken down by variant AND option because the two answer different
    questions: which variant's deck failed to take effect (E alone points at
    the Dynamic Preissmann Slot; every variant at once points at the deck
    writer), and which option the engine actually resolved differently. A
    single total would answer neither.

    Counted per run and per distinct model, for the same reason the exclusion
    ledger separates pairings from models: one broken deck applied to four
    hundred models is one bug, and a reader must be able to see that.
    """
    if anomalies.empty:
        return pd.DataFrame(columns=ANOMALY_SUMMARY_COLUMNS)

    grouped = anomalies.groupby(["variant", "option"], observed=True)
    frame = grouped.agg(runs=("model_id", "size"),
                        models=("model_id", "nunique")).reset_index()
    return frame.sort_values(["variant", "option"], ignore_index=True)


# ---------------------------------------------------------------------------
# Time-series comparisons computed over an incomplete overlap
# ---------------------------------------------------------------------------
#
# `outdiff.diff_series` compares only the timestamps two runs SHARE. If one
# run stopped at hour 12 of a 24-hour simulation and the two agreed over the
# first 12, its row reports `max_abs ~ 0` -- an excellent result, computed
# over half a run that silently failed. The numbers are still published (the
# agreement over the compared span is a real finding, and it is the evidence
# that identifies the truncation as the whole story), but a published number
# that cannot be read as a full comparison must be COUNTED and ATTRIBUTED,
# not merely accompanied by a column somebody might notice -- the same
# standard the option-echo anomalies and the withheld-iteration ledger meet.

#: `ts_diff` columns this section reads. A store whose `ts_diff` predates them
#: has no coverage evidence at all, which is reported as *not assessed* rather
#: than as *complete*: absence of evidence is not evidence of completeness.
COVERAGE_FRACTION_COLUMN = "coverage_fraction"
TIME_GRID_MATCH_COLUMN = "time_grid_match"

COVERAGE_ANOMALY_COLUMNS = ["comparison", "models", "series", "min_coverage"]


def _incomplete_coverage(ts_diff: pd.DataFrame) -> pd.Series:
    """Per-row mask of comparisons that did NOT cover both grids in full.

    A row counts as incomplete when its `time_grid_match` is known False, or
    when its `coverage_fraction` is known and below 1.0. Either alone would
    miss a store carrying only the other column; neither treats a MISSING
    value as incomplete, for the same reason a missing option echo is not a
    contradiction -- absence is no evidence, and a store written before these
    columns existed must not be reported as a corpus-wide anomaly.
    """
    incomplete = pd.Series(False, index=ts_diff.index)
    if TIME_GRID_MATCH_COLUMN in ts_diff.columns:
        matched = ts_diff[TIME_GRID_MATCH_COLUMN]
        incomplete |= matched.notna() & ~matched.fillna(False).astype(bool)
    if COVERAGE_FRACTION_COLUMN in ts_diff.columns:
        fraction = pd.to_numeric(ts_diff[COVERAGE_FRACTION_COLUMN],
                                 errors="coerce")
        incomplete |= fraction.notna() & (fraction < 1.0)
    return incomplete


def coverage_anomalies(ts_diff: pd.DataFrame) -> pd.DataFrame:
    """Time-series comparisons whose two runs did not cover the same grid.

    One row per `comparison`, because that is what an operator acts on:
    `D_minus_C` alone points at the C or D run, every comparison at once
    points at A.

    `models` is the count of distinct models affected -- equivalently, of
    affected (model, comparison) pairings, since a comparison contributes one
    pairing per model. `series` is the number of `ts_diff` rows behind them,
    i.e. per-element per-attribute series. Both are given for the reason the
    exclusion ledger separates pairings from models: a truncated run
    truncates every element and attribute it wrote, so `series` alone would
    report one stopped simulation as thousands of failures, while `models`
    alone would hide how much of the comparison is affected.

    `min_coverage` is the worst `coverage_fraction` in the group, so a reader
    can tell a comparison that missed one reporting period from one that
    missed half the simulation without opening the Parquet store.
    """
    if ts_diff.empty or not {"comparison", "model_id"} <= set(ts_diff.columns):
        return pd.DataFrame(columns=COVERAGE_ANOMALY_COLUMNS)

    flagged = ts_diff[_incomplete_coverage(ts_diff)]
    if flagged.empty:
        return pd.DataFrame(columns=COVERAGE_ANOMALY_COLUMNS)

    fraction = (pd.to_numeric(flagged[COVERAGE_FRACTION_COLUMN],
                              errors="coerce")
                if COVERAGE_FRACTION_COLUMN in flagged.columns
                else pd.Series(float("nan"), index=flagged.index,
                               dtype="float64"))
    frame = flagged.assign(_coverage=fraction)

    grouped = frame.groupby("comparison", observed=True)
    rows = [{"comparison": comparison,
             "models": group["model_id"].nunique(),
             "series": len(group),
             "min_coverage": group["_coverage"].min()}
            for comparison, group in grouped]
    return (pd.DataFrame(rows, columns=COVERAGE_ANOMALY_COLUMNS)
            .sort_values("comparison", ignore_index=True))


def coverage_anomaly_models(ts_diff: pd.DataFrame) -> int:
    """Distinct models with at least one incomplete-overlap comparison.

    NOT the sum of `coverage_anomalies`'s `models` column. That column is a
    per-comparison count, and one truncated run truncates every comparison it
    takes part in, so summing it reports a single stopped simulation as
    "5 model(s)" -- inflating the headline by up to the number of comparisons
    while the table beneath it says 1 five times over. Counted across
    comparisons here, from the same mask the table is built from, so the
    headline and the table are two views of one set rather than two
    computations that can disagree.
    """
    if ts_diff is None or ts_diff.empty:
        return 0
    if not {"comparison", "model_id"} <= set(ts_diff.columns):
        return 0
    return int(ts_diff[_incomplete_coverage(ts_diff)]["model_id"].nunique())


def coverage_was_assessed(ts_diff: pd.DataFrame) -> bool:
    """True when `ts_diff` carries coverage evidence at all.

    False for an absent `ts_diff` and for one written before the coverage
    columns existed. The summary states that case as *not assessed*: an
    unqualified "every comparison covered the full run" would be a claim the
    store cannot support.
    """
    if ts_diff is None or ts_diff.empty:
        return False
    return bool({COVERAGE_FRACTION_COLUMN, TIME_GRID_MATCH_COLUMN}
                & set(ts_diff.columns))


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

#: Hardness buckets over variant A's `avg_iterations_per_step`, as
#: `(label, predicate)` in report order. Anderson acceleration cannot help a
#: model that already converges in two Picard iterations, and this corpus is
#: dominated by small decks -- so an unstratified mean over every model
#: understates the feature exactly where it is supposed to pay off. The
#: labels are numeric notation, not prose: they stay verbatim in both
#: languages, like the `B - A` axis notation.
HARDNESS_BUCKETS = (
    ("<= 2", lambda value: value <= 2.0),
    ("2-4", lambda value: 2.0 < value <= 4.0),
    ("> 4", lambda value: value > 4.0),
)

HARDNESS_LABELS = [label for label, _ in HARDNESS_BUCKETS]

#: The bucket for a pairing that is computable but cannot be stratified: no
#: variant A row at all (a crashed baseline), an A run that was not routed by
#: DYNWAVE, or an A run with no iteration count. `D - C` needs neither an A
#: value nor an A run to be valid -- it reads C and D only -- so bucketing it
#: by `value_a` alone would silently drop a real measurement from the table.
#: A status-like data value, verbatim in both languages.
HARDNESS_UNKNOWN = "unknown"

#: Bucket order in the iteration-shift tables: the three hardness strata,
#: then the unstratifiable remainder last so it never displaces a real one.
REPORTED_HARDNESS_LABELS = HARDNESS_LABELS + [HARDNESS_UNKNOWN]


def hardness_bucket(value) -> str | None:
    """The bucket `value` falls in, or None when it is missing.

    The buckets partition the reals: `<= 2` is closed at 2, `2-4` is
    half-open `(2, 4]`, and `> 4` takes the rest, so no value lands in two
    buckets and no finite value lands in none.
    """
    if value is None or pd.isna(value):
        return None
    number = float(value)
    for label, predicate in HARDNESS_BUCKETS:
        if predicate(number):
            return label
    return None


def dynwave_models(runs: pd.DataFrame) -> set:
    """Models whose baseline (variant A) run was routed by DYNWAVE.

    The iteration-shift sections are restricted to these. A KINWAVE or
    STEADY run still prints an `Average Iterations per Step` line, but the
    number counts nothing Picard-shaped; pooling those into a median makes
    the median describe a quantity nobody named.
    """
    if runs.empty or "reported_routing_model" not in runs.columns:
        return set()
    baseline = runs[runs["variant"] == schema.VARIANT_A]
    routed = baseline[
        baseline["reported_routing_model"].astype("string").str.strip().str.upper()
        == DYNWAVE
    ]
    return set(routed["model_id"])


def reported_hardness(iterations: pd.DataFrame, routed: set) -> pd.Series:
    """The hardness bucket each iteration-delta row is filed under.

    `hardness_bucket(value_a)` when the model HAS a usable baseline --
    a variant A run this store knows was routed by DYNWAVE, carrying an
    iteration count -- and `HARDNESS_UNKNOWN` otherwise.

    The `routed` gate is not redundant with a present `value_a`: a KINWAVE
    A run still prints an `Average Iterations per Step` line, so bucketing on
    its value would sort a model by a number that counts nothing Picard-
    shaped. Such a model lands in `unknown` rather than in a hardness stratum
    it did not earn.

    One function so the tables and the exclusion ledger agree by
    construction: every pairing the tables file under `unknown` is exactly
    the set the ledger reports as `no_baseline_row`.
    """
    if iterations.empty:
        return pd.Series(dtype="object")
    values = (iterations["value_a"] if "value_a" in iterations.columns
              else pd.Series(pd.NA, index=iterations.index))
    buckets = [
        (hardness_bucket(value) if model_id in routed else None) or HARDNESS_UNKNOWN
        for model_id, value in zip(iterations["model_id"], values)
    ]
    return pd.Series(buckets, index=iterations.index, dtype="object")


#: Why a pairing that had BOTH operand values still produced no iteration
#: delta -- plus one cause (`no_baseline_row`) for a pairing that WAS
#: computed but could not be stratified. Data values, verbatim in both
#: languages.
#:
#: The three are kept apart because they mean different things to an
#: operator. `not_dynwave` is expected and healthy -- a KINWAVE, STEADY or FV
#: run has no Picard iteration count to compare, and excluding it is the
#: harness working. `routing_unknown` is actionable: nothing recorded what
#: routed the run, so the delta was withheld for lack of provenance rather
#: than because the feature did nothing. Collapsing them into one "excluded"
#: number would hide the second inside the first.
#:
#: `no_baseline_row` is neither: the delta EXISTS and is published. `D - C`
#: reads C and D only, so a model whose A run crashed still yields a valid
#: one -- it simply cannot be sorted into a hardness stratum defined by
#: variant A. It is listed here so a pairing that is real but unstratifiable
#: is visible in the accounting instead of merely appearing in an `unknown`
#: row someone has to notice.
EXCLUSION_NOT_DYNWAVE = "not_dynwave"
EXCLUSION_ROUTING_UNKNOWN = "routing_unknown"
EXCLUSION_NO_BASELINE = "no_baseline_row"

EXCLUSION_COLUMNS = ["cause", "pairings", "models"]


def iteration_exclusions(
    deltas: pd.DataFrame,
    runs: pd.DataFrame,
    metric: str = "avg_iterations_per_step",
) -> pd.DataFrame:
    """Iteration deltas withheld by the routing-model guard, by cause, plus
    the computed ones that could not be stratified.

    Counted per **(model, comparison) pairing**, not per model: the thing
    that was not produced is a delta, and one model contributes six of them.
    A per-model count would report `1` where six headline numbers went
    missing. The distinct model count is carried alongside so a reader can
    still see how concentrated the loss is.

    Only pairings that WOULD otherwise have been computed are counted -- both
    operand values present, and the delta null. A pairing whose operand is
    simply missing (a crashed variant) was never withheld by this guard and
    is not evidence of anything about routing provenance.

    A pairing where every *known* routing model is `DYNWAVE` but one is
    absent counts as `routing_unknown`; one where any known routing model is
    something else counts as `not_dynwave`, because that pairing would have
    been excluded even with complete provenance. Pairings whose two sides are
    both known `DYNWAVE` are not counted at all: they were dropped by the
    iteration-kind, anchor-surcharge or option-echo guard, which have their
    own caveats.

    `no_baseline_row` is the one cause counted over deltas that WERE
    computed: the pairing produced a number, but the model has no DYNWAVE
    variant A baseline to stratify it by, so the iteration-shift tables file
    it under the `unknown` hardness bucket. Without this line a `D - C` from
    a model whose A run crashed would be published in a bucket named
    `unknown` and appear in no accounting at all.
    """
    if deltas.empty or "metric" not in deltas.columns:
        return pd.DataFrame(columns=EXCLUSION_COLUMNS)

    routing = _pivot(runs, "reported_routing_model")
    frame = deltas[deltas["metric"] == metric]
    # Read positionally, not by label: `deltas` can arrive with a
    # non-unique index (a frame re-read from its partitioned dataset and
    # concatenated), where a label lookup would return a Series.
    hardness = list(reported_hardness(frame, dynwave_models(runs)))
    hits: dict[str, list] = {EXCLUSION_NOT_DYNWAVE: [],
                             EXCLUSION_ROUTING_UNKNOWN: [],
                             EXCLUSION_NO_BASELINE: []}

    for position, (_, row) in enumerate(frame.iterrows()):
        for column, left, right in DELTA_SPECS:
            if column not in row.index:
                continue
            values = [row.get(f"value_{side.lower()}") for side in (left, right)]
            if any(value is None or pd.isna(value) for value in values):
                continue
            if pd.notna(row.get(column)):
                # Computed, so nothing was withheld -- but a pairing with no
                # baseline to stratify against still has to be accounted for
                # somewhere, or it is only ever visible as an `unknown` row.
                if hardness[position] == HARDNESS_UNKNOWN:
                    hits[EXCLUSION_NO_BASELINE].append(row["model_id"])
                continue

            models = [_lookup(routing, row["model_id"], side)
                      for side in (left, right)]
            known = [str(model).strip().upper() for model in models
                     if model is not None and not pd.isna(model)]
            if any(model != DYNWAVE for model in known):
                hits[EXCLUSION_NOT_DYNWAVE].append(row["model_id"])
            elif len(known) < 2:
                hits[EXCLUSION_ROUTING_UNKNOWN].append(row["model_id"])

    rows = [{"cause": cause, "pairings": len(models),
             "models": len(set(models))}
            for cause, models in hits.items() if models]
    return pd.DataFrame(rows, columns=EXCLUSION_COLUMNS)


#: Per-element metrics that evidence flooding, i.e. a node whose hydraulic
#: grade rose above the ground and spilled. Either being positive for any
#: node of a model is taken as surcharge activity.
FLOODING_METRICS = ("node_total_flood_volume", "node_hours_flooded")

#: Below this many surcharge-active models, the D/E surcharge strata are
#: reported as anecdote rather than as a measurement. A mean over one or two
#: models reads exactly like a mean over four hundred in a markdown table,
#: and the reader cannot tell them apart without being told.
SURCHARGE_ACTIVE_MIN = 3

#: Strata labels for the surcharge-activity split. Status-like data values,
#: verbatim in both languages.
SURCHARGE_ACTIVE = "active"
SURCHARGE_INACTIVE = "inactive"


def surcharge_active_models(
    runs: pd.DataFrame, elements: pd.DataFrame | None = None,
) -> set:
    """Models whose baseline run shows evidence of surcharging.

    The proxy, and why this combination:

    * `pct_steps_not_converging > 0` on variant A. Under EXTRAN the
      free-surface/surcharge transition is the dominant source of
      non-convergence in the dynamic-wave solver, so a model that never
      fails to converge is very unlikely to be pressurising anything.
    * any node with `node_total_flood_volume > 0` or
      `node_hours_flooded > 0` on variant A. This catches the pressurised
      models that converge anyway, which the first signal alone would miss.

    `node_max_hgl` is parsed by the harness but is deliberately NOT used:
    an HGL is an elevation, and telling surcharge from ordinary free-surface
    depth needs each node's crown elevation, which the harness does not
    parse. There is no threshold to compare a bare HGL against, so including
    it would add noise rather than signal.

    Both signals are read from variant A only -- the baseline every D and E
    delta is anchored to -- so the stratum a model lands in does not depend
    on the very feature being measured.
    """
    active: set = set()

    if not runs.empty and "variant" in runs.columns:
        baseline = runs[runs["variant"] == schema.VARIANT_A]
        if "pct_steps_not_converging" in baseline.columns:
            values = pd.to_numeric(baseline["pct_steps_not_converging"],
                                   errors="coerce")
            active |= set(baseline.loc[values > 0, "model_id"])

    if elements is not None and not elements.empty:
        needed = {"variant", "metric", "value", "model_id"}
        if needed <= set(elements.columns):
            flooding = elements[
                (elements["variant"] == schema.VARIANT_A)
                & (elements["metric"].isin(FLOODING_METRICS))
            ]
            values = pd.to_numeric(flooding["value"], errors="coerce")
            active |= set(flooding.loc[values > 0, "model_id"])

    return active


#: Default report language. The harness's own author works in Spanish, but
#: the harness is being submitted upstream to an English-language project --
#: so the *code* stays English (identifiers, comments, docstrings) while only
#: the generated prose is switchable. `es` is the default because that is
#: this harness's primary operator; `en` exists for the upstream project.
DEFAULT_LANG = "es"

#: All prose that lands in the generated `summary.md`, keyed by language so
#: the two versions sit side by side and cannot drift apart -- deliberately
#: not a scatter of `if lang == "es"` branches through write_markdown's body.
#: Metric names, status values, family names, variant letters, the hardness
#: bucket labels (`<= 2`, `2-4`, `> 4`), the surcharge strata (`active`,
#: `inactive`) and the `B - A` / `C - A` / `D - C` / `D - A` / `E - A` /
#: `A - REF` axis notation are NOT here: they are Parquet column values and
#: identifiers a reader cross-references against the data and the code, and
#: must stay verbatim in both languages.
MARKDOWN_STRINGS = {
    "en": {
        "title": "# Corpus Benchmark Summary",
        "coverage_heading": "## Coverage",
        "coverage_header": "| status | runs |",
        "coverage_sep": "| --- | --- |",
        "anomaly_heading": "## Option echoes contradicting variant intent",
        "anomaly_header": "| variant | option | runs | models |",
        "anomaly_sep": "| --- | --- | --- | --- |",
        "anomaly_intro": [
            "The engine silently ignores an unrecognised option value, so a",
            "deck that failed to take effect would otherwise read as *the",
            "feature has no effect* across the whole corpus. Every run below",
            "echoed a value its own deck did not ask for; every delta with",
            "such a run on either side is withheld rather than published.",
        ],
        "anomaly_total": [
            "**{n} run(s) across {n_models} model(s) echoed an option value",
            "contradicting their variant's intent.** Treat the affected axes",
            "as unmeasured, not as measured-and-flat, and fix the deck before",
            "reading anything else here.",
        ],
        "anomaly_none": [
            "No run echoed an option value contradicting its variant's intent.",
        ],
        "coverage_anomaly_heading": "## Time-series comparisons over an incomplete overlap",
        "coverage_anomaly_header": "| comparison | models | series | min coverage |",
        "coverage_anomaly_sep": "| --- | --- | --- | --- |",
        "coverage_anomaly_intro": [
            "A time-series comparison reduces only the timestamps the two runs",
            "SHARE. If one run stopped at hour 12 of a 24-hour simulation and",
            "the two agreed over those 12, its row reports a near-zero",
            "`max_abs` -- an excellent-looking result computed over half a run",
            "that failed. `coverage_fraction` is `n_common` divided by the",
            "number of distinct timestamps appearing in EITHER series, so it",
            "is 1.0 exactly when the two reporting grids are identical.",
            "`start_time_match`, `end_time_match` and `time_grid_match`",
            "separate a truncated run (start matches, end does not) from the",
            "same span sampled differently (both ends match, the grid does",
            "not).",
        ],
        "coverage_anomaly_total": [
            "**{n_models} model(s) across {n_comparisons} comparison(s)",
            "produced a time-series comparison over an incomplete overlap.**",
            "Their `max_abs`, `max_rel` and `rmse` ARE still written to",
            "`ts_diff` -- an exact agreement up to the point one run stopped",
            "is a real finding, and deleting it would destroy the evidence",
            "that identifies the truncation -- but they are statistics about a",
            "sub-span, and must never be read as a full comparison. Check",
            "`coverage_fraction` on any row before quoting its numbers.",
        ],
        "coverage_anomaly_none": [
            "Every time-series comparison in this store covered both runs'",
            "reporting grids in full.",
        ],
        "coverage_not_assessed": [
            "**Coverage was not assessed.** This store has no `ts_diff` rows",
            "carrying `coverage_fraction` or `time_grid_match` -- either the",
            "`diff` stage has not been run, or it was run before those columns",
            "existed. Read that as *not checked*, never as *complete*: a run",
            "that stopped halfway would be invisible here.",
        ],
        "interaction_heading": "## Anderson × Node Continuity interaction ((D - C) - (B - A))",
        "interaction_intro": [
            "A/B/C/D is a 2x2 factorial over `ANDERSON_ACCEL` and",
            "`NODE_CONTINUITY`. `B - A` measures Anderson under `EXPLICIT`;",
            "`D - C` measures it under `SEMI_IMPLICIT`; the interaction is the",
            "difference between those two simple effects,",
            "`(D - C) - (B - A)`, equivalently `D - C - B + A`. It answers the",
            "question no individual axis answers: **does switching to",
            "semi-implicit (Crank-Nicolson) node continuity change how",
            "effective Anderson acceleration is?**",
        ],
        "interaction_sign": [
            "**Sign convention.** The interaction is a signed change in the",
            "metric itself, in that metric's own units -- it is not a score.",
            "It is NEGATIVE when Anderson moves the metric further DOWN under",
            "`SEMI_IMPLICIT` than it does under `EXPLICIT`. So for",
            "`avg_iterations_per_step`, `pct_steps_not_converging` and",
            "`wall_ms`, where lower is better, **a negative interaction means",
            "Anderson helps MORE under Crank-Nicolson**, and a positive one",
            "means it helps less. Negative is the expected direction:",
            "`DWSolver::computeAASkipFlags` disables Anderson at every",
            "surcharged node under `EXPLICIT`/`EXTRAN` and not under",
            "`SEMI_IMPLICIT`, so `B - A` measures Anderson with its most",
            "valuable case switched off. For the `continuity_error_*` metrics",
            "the sign is directional in the error, not in its magnitude, so",
            "read those rows with `min` and `max` beside the mean.",
        ],
        "interaction_empty": [
            "No model produced both a `B - A` and a `D - C` delta for any",
            "metric, so the interaction could not be formed. It is null",
            "whenever either simple effect is -- an operand missing, or the",
            "pairing rejected by a commensurability screen -- so read this as",
            "*not measured*, never as *no interaction*.",
        ],
        "b_minus_a_heading": "## B - A by metric (Anderson acceleration effect)",
        "c_minus_a_heading": "## C - A by metric (Crank-Nicolson continuity effect)",
        "d_minus_c_heading": "## D - C by metric (incremental Anderson under Crank-Nicolson)",
        "d_minus_a_heading": "## D - A by metric (joint Anderson + Crank-Nicolson effect)",
        "e_minus_a_heading": "## E - A by metric (Dynamic Preissmann Slot effect)",
        "metric_header": "| metric | models | mean | median | min | max |",
        "metric_sep": "| --- | --- | --- | --- | --- | --- |",
        "iter_shift_b_heading": "## Iteration shift by family and hardness (B - A)",
        "iter_shift_c_heading": "## Iteration shift by family and hardness (C - A)",
        "iter_shift_d_heading": "## Iteration shift by family and hardness (D - C)",
        "iter_shift_e_heading": "## Iteration shift by family and hardness (E - A)",
        "family_header": "| family | hardness | models | mean abs iterations |",
        "family_sep": "| --- | --- | --- | --- |",
        "iter_excluded_heading": "## Iteration deltas withheld, by cause",
        "iter_excluded_header": "| cause | pairings | models |",
        "iter_excluded_sep": "| --- | --- | --- |",
        "iter_excluded_intro": [
            "Counted per (model, comparison) pairing, over pairings that had",
            "both operand values and would otherwise have been computed.",
            "`not_dynwave` is expected and healthy: a KINWAVE, STEADY or FV run",
            "has no Picard iteration count to compare. `no_baseline_row` is the",
            "one line counted over deltas that WERE computed: the pairing (a",
            "`D - C` whose A run crashed, say) produced a number, but has no",
            "variant A baseline to stratify it by, so it appears in the",
            "iteration-shift tables under the `unknown` hardness bucket.",
        ],
        "iter_excluded_none": [
            "No iteration delta was withheld by the routing-model guard.",
        ],
        "iter_unknown_note": [
            "**Some pairings were skipped for lack of a recorded routing",
            "model, not because the feature did nothing.** No",
            "`reported_routing_model` was recorded for at least one side, so",
            "there is no evidence the counter means Picard iterations at all",
            "and the delta was withheld rather than guessed. This usually",
            "means a store written before that column existed, or a report the",
            "parser could not read. Re-run those models before drawing any",
            "conclusion from the iteration sections.",
        ],
        "iter_empty_note": [
            "**The iteration-shift sections above are empty: not one iteration",
            "delta survived.** Read that as *nothing was measured*, never as",
            "*the feature has no effect* -- the causes are itemised in the",
            "table above, and an absent measurement refutes nothing.",
        ],
        "surcharge_heading": "## D and E by surcharge activity",
        "surcharge_header": "| axis | surcharge | metric | models | mean | median |",
        "surcharge_sep": "| --- | --- | --- | --- | --- | --- |",
        "surcharge_proxy": [
            "A model counts as `active` when its variant A run reports",
            "`pct_steps_not_converging` greater than zero, or any of its nodes",
            "reports `node_total_flood_volume` or `node_hours_flooded` greater",
            "than zero. Both signals are read from variant A only, so the",
            "stratum does not depend on the feature being measured.",
            "`node_max_hgl` is parsed but not used here: separating surcharge",
            "from ordinary free-surface depth needs each node's crown",
            "elevation, which the harness does not parse.",
        ],
        "surcharge_none": [
            "**No model in this store is surcharge-active by that proxy.** D and",
            "E act on the free-surface/surcharge transition, so this sweep",
            "measures them where they are not designed to act. Read the",
            "`inactive` rows below as *the feature was not exercised*, never as",
            "*the feature has no effect* -- those two conclusions are not the",
            "same, and only the second one is refutable by this table.",
        ],
        "surcharge_few": [
            "**Only {n} surcharge-active model(s) in this store.** Any mean over",
            "them is anecdote, not evidence; the `active` rows below are shown",
            "for inspection, not for a conclusion.",
        ],
        "surcharge_axis_empty": [
            "**The {axis} axis has no `active` row: this store has",
            "surcharge-active models, but not one of them produced a {axis}",
            "delta.** Nothing was measured on that axis where D and E actually",
            "act, so read its `inactive` rows -- and its silence -- as *not",
            "measured*, never as *no effect*. The announcement is per axis",
            "because one axis can come up empty while its neighbours are fully",
            "populated.",
        ],
        "caveats_heading": "## Caveats",
        "caveat_estimate": [
            "- `total_iterations_est` is a derived **estimate**",
            "  (`avg_iterations_per_step x duration / avg_step`); the report",
            "  publishes only a two-decimal mean. Its error grows on",
            "  variable-time-step models. Use `avg_iterations_per_step` for any",
            "  conclusion that matters.",
        ],
        "caveat_kind_mismatch": [
            "- Rows whose `iteration_metric_kind` differs from the other side's",
            "  are excluded from that pairing's iteration delta (B - A, C - A,",
            "  D - A, D - C, E - A and A - REF independently): FV substeps are",
            "  not Picard iterations.",
        ],
        "caveat_routing_model": [
            "- Iteration deltas are formed only when BOTH sides are known to be",
            "  `DYNWAVE`, and the iteration-shift sections show only those rows.",
            "  The engine prints `Average Iterations per Step` for KINWAVE and",
            "  STEADY too, where the counter has no Picard meaning; those runs",
            "  would otherwise agree on their kind and pass straight through.",
        ],
        "caveat_hardness": [
            "- The iteration-shift tables are split by variant A's",
            "  `avg_iterations_per_step`. Anderson acceleration cannot help a",
            "  model that already converges in two iterations, and this corpus",
            "  is dominated by small decks, so a single unstratified mean",
            "  understates the feature exactly where it is meant to pay off.",
        ],
        "caveat_surcharge_ref": [
            "- `X - REF` is dropped when the two sides echo different",
            "  `SURCHARGE_METHOD` values. EPA SWMM 5.2 has no Dynamic",
            "  Preissmann Slot, so `E - REF` is always dropped; the rule is",
            "  written against the echoes, so it also catches a reference run",
            "  under a surcharge method other than ours.",
        ],
        "caveat_option_echo": [
            "- A delta is dropped when either side's `reported_*` echo",
            "  contradicts what its variant's deck asked for. The engine",
            "  silently ignores an unrecognised option value, so an E run that",
            "  actually executed under `EXTRAN` would make `E - A` a",
            "  subtraction of two identical configurations and publish it as",
            "  ~0. A missing echo is not a contradiction; REF has no intent to",
            "  contradict.",
        ],
        "caveat_time_series": [
            "- Time series are compared only between our own runs. The external",
            "  anchor is summary-level.",
        ],
        "caveat_coverage": [
            "- A `ts_diff` row is reduced over the timestamps its two runs",
            "  share, so it carries `n_periods_left`, `n_periods_right`,",
            "  `n_common`, `coverage_fraction`, `start_time_match`,",
            "  `end_time_match` and `time_grid_match`. Anything below full",
            "  coverage is counted as an anomaly above; the row's `max_abs`,",
            "  `max_rel` and `rmse` are still written, but they describe the",
            "  shared sub-span only.",
        ],
    },
    "es": {
        "title": "# Resumen del Benchmark del Corpus",
        "coverage_heading": "## Cobertura",
        "coverage_header": "| estado | corridas |",
        "coverage_sep": "| --- | --- |",
        "anomaly_heading": "## Ecos de opciones que contradicen la intención de la variante",
        "anomaly_header": "| variante | opción | corridas | modelos |",
        "anomaly_sep": "| --- | --- | --- | --- |",
        "anomaly_intro": [
            "El motor ignora silenciosamente un valor de opción no reconocido,",
            "así que un deck que no tuvo efecto se leería como *la",
            "característica no tiene efecto* en todo el corpus. Cada corrida de",
            "abajo reportó un valor que su propio deck no pidió; todo delta con",
            "una corrida así en cualquiera de sus lados se retiene en vez de",
            "publicarse.",
        ],
        "anomaly_total": [
            "**{n} corrida(s) en {n_models} modelo(s) reportaron un valor de",
            "opción que contradice la intención de su variante.** Trate los",
            "ejes afectados como no medidos, no como medidos-y-planos, y",
            "corrija el deck antes de leer cualquier otra cosa aquí.",
        ],
        "anomaly_none": [
            "Ninguna corrida reportó un valor de opción que contradiga la "
            "intención de su variante.",
        ],
        "coverage_anomaly_heading": "## Comparaciones de series temporales sobre un solape incompleto",
        "coverage_anomaly_header": "| comparación | modelos | series | cobertura mínima |",
        "coverage_anomaly_sep": "| --- | --- | --- | --- |",
        "coverage_anomaly_intro": [
            "Una comparación de series temporales reduce solo las marcas de",
            "tiempo que ambas corridas COMPARTEN. Si una corrida se detuvo en",
            "la hora 12 de una simulación de 24 horas y ambas coincidieron en",
            "esas 12, su fila reporta un `max_abs` casi cero: un resultado de",
            "apariencia excelente calculado sobre media corrida que falló.",
            "`coverage_fraction` es `n_common` dividido por la cantidad de",
            "marcas de tiempo distintas que aparecen en CUALQUIERA de las dos",
            "series, así que vale 1.0 exactamente cuando las dos grillas de",
            "reporte son idénticas. `start_time_match`, `end_time_match` y",
            "`time_grid_match` distinguen una corrida truncada (coincide el",
            "inicio, no el final) de el mismo lapso muestreado distinto",
            "(coinciden ambos extremos, no la grilla).",
        ],
        "coverage_anomaly_total": [
            "**{n_models} modelo(s) en {n_comparisons} comparación(es)",
            "produjeron una comparación de series temporales sobre un solape",
            "incompleto.** Sus `max_abs`, `max_rel` y `rmse` SÍ se escriben en",
            "`ts_diff` -- una coincidencia exacta hasta el punto en que una",
            "corrida se detuvo es un hallazgo real, y borrarla destruiría la",
            "evidencia que identifica el truncamiento -- pero son estadísticos",
            "sobre un sublapso, y nunca deben leerse como una comparación",
            "completa. Revise `coverage_fraction` en cualquier fila antes de",
            "citar sus números.",
        ],
        "coverage_anomaly_none": [
            "Toda comparación de series temporales de este almacén cubrió por",
            "completo las grillas de reporte de ambas corridas.",
        ],
        "coverage_not_assessed": [
            "**No se evaluó la cobertura.** Este almacén no tiene filas de",
            "`ts_diff` con `coverage_fraction` ni `time_grid_match`: o no se",
            "ejecutó la etapa `diff`, o se ejecutó antes de que existieran esas",
            "columnas. Léalo como *no verificado*, nunca como *completo*: una",
            "corrida que se detuvo a mitad de camino sería invisible aquí.",
        ],
        "interaction_heading": "## Interacción Anderson × continuidad de nodos ((D - C) - (B - A))",
        "interaction_intro": [
            "A/B/C/D es un factorial 2x2 sobre `ANDERSON_ACCEL` y",
            "`NODE_CONTINUITY`. `B - A` mide Anderson bajo `EXPLICIT`;",
            "`D - C` lo mide bajo `SEMI_IMPLICIT`; la interacción es la",
            "diferencia entre esos dos efectos simples,",
            "`(D - C) - (B - A)`, equivalentemente `D - C - B + A`. Responde la",
            "pregunta que ningún eje individual responde: **¿cambia el pasar a",
            "continuidad de nodos semi-implícita (Crank-Nicolson) qué tan",
            "efectiva es la aceleración de Anderson?**",
        ],
        "interaction_sign": [
            "**Convención de signo.** La interacción es un cambio con signo en",
            "la métrica misma, en las unidades de esa métrica: no es un",
            "puntaje. Es NEGATIVA cuando Anderson mueve la métrica más hacia",
            "ABAJO bajo `SEMI_IMPLICIT` que bajo `EXPLICIT`. Así que para",
            "`avg_iterations_per_step`, `pct_steps_not_converging` y",
            "`wall_ms`, donde menos es mejor, **una interacción negativa",
            "significa que Anderson ayuda MÁS bajo Crank-Nicolson**, y una",
            "positiva significa que ayuda menos. Negativa es la dirección",
            "esperada: `DWSolver::computeAASkipFlags` desactiva Anderson en",
            "todo nodo sobrecargado bajo `EXPLICIT`/`EXTRAN` y no bajo",
            "`SEMI_IMPLICIT`, así que `B - A` mide Anderson con su caso más",
            "valioso apagado. Para las métricas `continuity_error_*` el signo",
            "es direccional en el error, no en su magnitud, así que lea esas",
            "filas con `min` y `max` junto a la media.",
        ],
        "interaction_empty": [
            "Ningún modelo produjo a la vez un delta `B - A` y uno `D - C` para",
            "métrica alguna, así que la interacción no pudo formarse. Es nula",
            "siempre que lo sea cualquiera de los dos efectos simples -- un",
            "operando ausente, o el par rechazado por un filtro de",
            "conmensurabilidad -- así que léalo como *no medido*, nunca como",
            "*sin interacción*.",
        ],
        "b_minus_a_heading": "## B - A por métrica (efecto de la aceleración de Anderson)",
        "c_minus_a_heading": "## C - A por métrica (efecto de continuidad de Crank-Nicolson)",
        "d_minus_c_heading": "## D - C por métrica (Anderson incremental sobre Crank-Nicolson)",
        "d_minus_a_heading": "## D - A por métrica (efecto conjunto de Anderson + Crank-Nicolson)",
        "e_minus_a_heading": "## E - A por métrica (efecto de la ranura de Preissmann dinámica)",
        "metric_header": "| métrica | modelos | media | mediana | mínimo | máximo |",
        "metric_sep": "| --- | --- | --- | --- | --- | --- |",
        "iter_shift_b_heading": "## Cambio de iteraciones por familia y dificultad (B - A)",
        "iter_shift_c_heading": "## Cambio de iteraciones por familia y dificultad (C - A)",
        "iter_shift_d_heading": "## Cambio de iteraciones por familia y dificultad (D - C)",
        "iter_shift_e_heading": "## Cambio de iteraciones por familia y dificultad (E - A)",
        "family_header": "| familia | dificultad | modelos | media de iteraciones absolutas |",
        "family_sep": "| --- | --- | --- | --- |",
        "iter_excluded_heading": "## Deltas de iteraciones retenidos, por causa",
        "iter_excluded_header": "| causa | pares | modelos |",
        "iter_excluded_sep": "| --- | --- | --- |",
        "iter_excluded_intro": [
            "Contados por par (modelo, comparación), sobre los pares que tenían",
            "ambos valores operandos y que de otro modo se habrían calculado.",
            "`not_dynwave` es esperado y sano: una corrida KINWAVE, STEADY o FV",
            "no tiene un conteo de iteraciones de Picard que comparar.",
            "`no_baseline_row` es la única línea contada sobre deltas que SÍ se",
            "calcularon: el par (por ejemplo un `D - C` cuya corrida A falló)",
            "produjo un número, pero no tiene una línea base de la variante A",
            "para estratificarlo, así que aparece en las tablas de cambio de",
            "iteraciones bajo el bucket de dificultad `unknown`.",
        ],
        "iter_excluded_none": [
            "El filtro de modelo de ruteo no retuvo ningún delta de iteraciones.",
        ],
        "iter_unknown_note": [
            "**Algunos pares se omitieron por falta de un modelo de ruteo",
            "registrado, no porque la característica no hiciera nada.** No se",
            "registró `reported_routing_model` para al menos uno de los lados,",
            "así que no hay evidencia de que el contador signifique iteraciones",
            "de Picard, y el delta se retuvo en vez de adivinarse. Esto suele",
            "indicar un almacén escrito antes de que existiera esa columna, o un",
            "informe que el parser no pudo leer. Vuelva a correr esos modelos",
            "antes de concluir nada de las secciones de iteraciones.",
        ],
        "iter_empty_note": [
            "**Las secciones de cambio de iteraciones de arriba están vacías: no",
            "sobrevivió ni un delta de iteraciones.** Léalo como *no se midió",
            "nada*, nunca como *la característica no tiene efecto*: las causas",
            "están detalladas en la tabla de arriba, y una medición ausente no",
            "refuta nada.",
        ],
        "surcharge_heading": "## D y E por actividad de sobrecarga",
        "surcharge_header": "| eje | sobrecarga | métrica | modelos | media | mediana |",
        "surcharge_sep": "| --- | --- | --- | --- | --- | --- |",
        "surcharge_proxy": [
            "Un modelo cuenta como `active` cuando su corrida de la variante A",
            "reporta `pct_steps_not_converging` mayor que cero, o alguno de sus",
            "nodos reporta `node_total_flood_volume` o `node_hours_flooded`",
            "mayor que cero. Ambas señales se leen solo de la variante A, de",
            "modo que el estrato no depende de la característica que se mide.",
            "`node_max_hgl` se parsea pero no se usa aquí: separar la sobrecarga",
            "de la profundidad ordinaria de superficie libre requiere la cota de",
            "clave de cada nodo, que el arnés no parsea.",
        ],
        "surcharge_none": [
            "**Ningún modelo de este almacén tiene sobrecarga activa según ese",
            "criterio.** D y E actúan sobre la transición entre superficie libre",
            "y sobrecarga, así que este barrido las mide donde no están",
            "diseñadas para actuar. Lea las filas `inactive` de abajo como *la",
            "característica no fue ejercitada*, nunca como *la característica no",
            "tiene efecto*: no son la misma conclusión, y solo la segunda sería",
            "refutable por esta tabla.",
        ],
        "surcharge_few": [
            "**Solo {n} modelo(s) con sobrecarga activa en este almacén.**",
            "Cualquier media sobre ellos es anécdota, no evidencia; las filas",
            "`active` de abajo se muestran para inspección, no para concluir.",
        ],
        "surcharge_axis_empty": [
            "**El eje {axis} no tiene ninguna fila `active`: este almacén sí",
            "tiene modelos con sobrecarga activa, pero ninguno de ellos produjo",
            "un delta {axis}.** No se midió nada en ese eje donde D y E",
            "realmente actúan, así que lea sus filas `inactive` -- y su",
            "silencio -- como *no medido*, nunca como *sin efecto*. El aviso es",
            "por eje porque un eje puede quedar vacío mientras sus vecinos",
            "están llenos.",
        ],
        "caveats_heading": "## Advertencias",
        "caveat_estimate": [
            "- `total_iterations_est` es una **estimación** derivada",
            "  (`avg_iterations_per_step x duration / avg_step`); el informe",
            "  publica solo una media con dos decimales. Su error crece en",
            "  modelos de paso de tiempo variable. Use `avg_iterations_per_step`",
            "  para cualquier conclusión que importe.",
        ],
        "caveat_kind_mismatch": [
            "- Las filas cuyo `iteration_metric_kind` difiere del del otro lado",
            "  se excluyen del delta de iteraciones de ese par (B - A, C - A,",
            "  D - A, D - C, E - A y A - REF de forma independiente): los",
            "  subpasos de FV no son iteraciones de Picard.",
        ],
        "caveat_routing_model": [
            "- Los deltas de iteraciones se forman solo cuando AMBOS lados son",
            "  `DYNWAVE` de forma conocida, y las secciones de cambio de",
            "  iteraciones muestran solo esas filas. El motor imprime",
            "  `Average Iterations per Step` también para KINWAVE y STEADY,",
            "  donde el contador no tiene significado de Picard; esas corridas",
            "  coincidirían en su tipo y pasarían sin ser filtradas.",
        ],
        "caveat_hardness": [
            "- Las tablas de cambio de iteraciones se dividen según el",
            "  `avg_iterations_per_step` de la variante A. La aceleración de",
            "  Anderson no puede ayudar a un modelo que ya converge en dos",
            "  iteraciones, y este corpus está dominado por decks pequeños, así",
            "  que una sola media sin estratificar subestima la característica",
            "  justo donde debería rendir.",
        ],
        "caveat_surcharge_ref": [
            "- `X - REF` se descarta cuando ambos lados reportan valores",
            "  distintos de `SURCHARGE_METHOD`. EPA SWMM 5.2 no tiene la ranura",
            "  de Preissmann dinámica, así que `E - REF` siempre se descarta; la",
            "  regla se escribe contra los ecos, de modo que también detecta una",
            "  referencia corrida con un método de sobrecarga distinto al",
            "  nuestro.",
        ],
        "caveat_option_echo": [
            "- Un delta se descarta cuando el eco `reported_*` de cualquiera de",
            "  los dos lados contradice lo que pidió el deck de su variante. El",
            "  motor ignora silenciosamente un valor de opción no reconocido,",
            "  así que una corrida E que en realidad se ejecutó bajo `EXTRAN`",
            "  haría de `E - A` una resta de dos configuraciones idénticas y la",
            "  publicaría como ~0. Un eco ausente no es una contradicción; REF",
            "  no tiene intención que contradecir.",
        ],
        "caveat_time_series": [
            "- Las series temporales se comparan solo entre nuestras propias",
            "  corridas. El ancla externa es a nivel de resumen.",
        ],
        "caveat_coverage": [
            "- Una fila de `ts_diff` se reduce sobre las marcas de tiempo que",
            "  comparten sus dos corridas, así que lleva `n_periods_left`,",
            "  `n_periods_right`, `n_common`, `coverage_fraction`,",
            "  `start_time_match`, `end_time_match` y `time_grid_match`. Toda",
            "  cobertura menor que la completa se cuenta como anomalía arriba;",
            "  el `max_abs`, `max_rel` y `rmse` de la fila sí se escriben, pero",
            "  describen únicamente el sublapso compartido.",
        ],
    },
}

#: `(delta column, heading key)` for the per-metric summary tables, in
#: report order. A/B/C first and unchanged, then the two D axes, then E.
METRIC_SECTIONS = (
    ("delta_b_minus_a", "b_minus_a_heading"),
    ("delta_c_minus_a", "c_minus_a_heading"),
    ("delta_d_minus_c", "d_minus_c_heading"),
    ("delta_d_minus_a", "d_minus_a_heading"),
    ("delta_e_minus_a", "e_minus_a_heading"),
)

#: `(delta column, heading key)` for the family x hardness iteration tables.
ITER_SECTIONS = (
    ("delta_b_minus_a", "iter_shift_b_heading"),
    ("delta_c_minus_a", "iter_shift_c_heading"),
    ("delta_d_minus_c", "iter_shift_d_heading"),
    ("delta_e_minus_a", "iter_shift_e_heading"),
)

#: `(axis label, delta column)` for the surcharge-activity strata. The axis
#: labels are notation, not prose, and stay verbatim in both languages.
SURCHARGE_SECTIONS = (
    ("D - C", "delta_d_minus_c"),
    ("D - A", "delta_d_minus_a"),
    ("E - A", "delta_e_minus_a"),
)


def _delta_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """`frame[column]` as floats, or an all-empty series when absent."""
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return frame[column].dropna().astype(float)


def write_markdown(
    deltas: pd.DataFrame,
    runs: pd.DataFrame,
    path: Path,
    lang: str = DEFAULT_LANG,
    elements: pd.DataFrame | None = None,
    ts_diff: pd.DataFrame | None = None,
) -> Path:
    """Write a summary readable without opening a notebook.

    `lang` selects the language of the generated prose only (headings, table
    column headers, the Caveats bullets). Metric names, status values, family
    names, hardness buckets, surcharge strata and the `B - A` / `C - A` /
    `D - C` / `D - A` / `E - A` / `A - REF` / `(D - C) - (B - A)` notation are
    data, not prose, and are identical in both languages.

    `elements` is optional and used only to strengthen the surcharge-activity
    proxy with per-node flooding totals; without it the proxy falls back to
    `pct_steps_not_converging` alone.

    `ts_diff` is optional and used only for the incomplete-overlap anomaly
    section. Passing nothing is reported as *coverage not assessed* rather
    than as *coverage complete*: the section exists to stop a comparison over
    half a run being read as a full one, and it could not do that if a
    missing table read as a clean bill of health.
    """
    path = Path(path)
    strings = MARKDOWN_STRINGS[lang]
    lines = [strings["title"], ""]

    if not runs.empty and "status" in runs.columns:
        lines += [strings["coverage_heading"], "",
                  strings["coverage_header"], strings["coverage_sep"]]
        for status, count in runs["status"].value_counts().items():
            lines.append(f"| {status} | {count} |")
        lines.append("")

    # Unconditional, and early: the one signal that catches a silently
    # ignored option belongs in the artifact people keep, not only in the
    # console of whoever happened to run the stage. A store with no anomaly
    # says so outright -- an absent section would be indistinguishable from
    # a harness that never looked.
    lines += _anomaly_section(runs, strings)

    # Unconditional for the same reason, and beside it: an incomplete overlap
    # is the second way a number in this store can be true and still
    # misleading. An absent section would be indistinguishable from a harness
    # that never looked.
    lines += _coverage_section(ts_diff, strings)

    if not deltas.empty:
        # One section per axis rather than columns of one table, so no axis
        # reads as derived from, or secondary to, another. D gets both of
        # its axes: D - C is the attributable incremental-Anderson
        # measurement, D - A the joint effect kept alongside it.
        for delta_col, heading_key in METRIC_SECTIONS:
            lines += [strings[heading_key], "",
                      strings["metric_header"], strings["metric_sep"]]
            if delta_col in deltas.columns:
                # `metric` here is plain object dtype (built row-by-row in
                # build_deltas), so groupby's default observed=False is
                # harmless.
                grouped = deltas.dropna(subset=[delta_col]).groupby("metric")
                for metric, group in grouped:
                    column = group[delta_col].astype(float)
                    lines.append(
                        f"| {metric} | {len(column)} | {column.mean():.4f} | "
                        f"{column.median():.4f} | {column.min():.4f} | "
                        f"{column.max():.4f} |"
                    )
            lines.append("")

        # After the five simple effects, because it is formed from two of
        # them and reads as nonsense before they have been seen.
        lines += _interaction_section(deltas, strings)

        # Only DYNWAVE baselines are STRATIFIED: a KINWAVE or STEADY run
        # still prints an iteration count, and bucketing on it would sort a
        # model by a number that counts nothing Picard-shaped. But the row is
        # no longer dropped for it. `build_deltas` has already nulled every
        # delta whose two sides are not both known DYNWAVE, so a row that
        # survives to here carries a real measurement -- and a `D - C` needs
        # no A run at all. Dropping such a row for want of an A baseline lost
        # a computed delta from the table AND from the exclusion ledger; it
        # is filed under `unknown` hardness instead, and counted there.
        iterations = deltas[deltas.metric == "avg_iterations_per_step"]
        iterations = iterations.assign(
            _hardness=reported_hardness(iterations, dynwave_models(runs)))

        emitted = 0
        for delta_col, heading_key in ITER_SECTIONS:
            lines += [strings[heading_key], "",
                      strings["family_header"], strings["family_sep"]]
            # `deltas["family"]` is object dtype in the direct build_deltas ->
            # write_markdown pipeline (build_deltas reconstructs it from
            # plain Python scalars), so this groupby is unaffected either
            # way here. But `deltas` can also be re-read from its own
            # persisted dataset (stage_report writes it partitioned by
            # family), where pandas restores `family` as a `category`
            # dtype. groupby on a categorical defaults to observed=False and
            # materialises EVERY category, including ones with zero rows in
            # the (already-filtered) `iterations` frame, as spurious
            # empty-group rows. observed=True avoids that in both cases.
            for family, group in iterations.groupby("family", observed=True):
                for bucket in REPORTED_HARDNESS_LABELS:
                    column = _delta_column(
                        group[group["_hardness"] == bucket], delta_col).abs()
                    if column.empty:
                        continue
                    lines.append(f"| {family} | {bucket} | {len(column)} | "
                                 f"{column.mean():.4f} |")
                    emitted += 1
            lines.append("")

        # The routing-model guard is deliberately stricter than the
        # iteration-kind one, and that strictness must not be silent: an
        # operator reading an empty iteration section concludes "the feature
        # has no effect", which is the same misreading the surcharge-activity
        # stratum exists to prevent. So the withheld pairings are counted and
        # itemised, and an entirely empty set of sections says so outright.
        lines += _exclusion_section(deltas, runs, strings, emitted)

        lines += _surcharge_section(deltas, runs, elements, strings)

    lines += [strings["caveats_heading"], ""]
    lines += strings["caveat_estimate"]
    lines += strings["caveat_kind_mismatch"]
    lines += strings["caveat_routing_model"]
    lines += strings["caveat_hardness"]
    lines += strings["caveat_surcharge_ref"]
    lines += strings["caveat_option_echo"]
    lines += strings["caveat_time_series"]
    lines += strings["caveat_coverage"]
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _anomaly_section(runs: pd.DataFrame, strings: dict) -> list[str]:
    """Runs whose engine-reported option value contradicts variant intent.

    The console warning `stage_report` prints is seen once, by one operator;
    `summary.md` is the artifact that gets kept and shared. A silently
    ignored option is the failure most likely to invert a conclusion -- an E
    deck that never took effect publishes `E - A ~ 0` -- so the count belongs
    in the document, broken down by variant and option, and the absence of
    any anomaly is stated rather than left to be inferred from a missing
    table.
    """
    anomalies = option_anomalies(runs) if not runs.empty else pd.DataFrame()
    lines = [strings["anomaly_heading"], ""]
    lines += strings["anomaly_intro"]
    lines.append("")

    if anomalies.empty:
        lines += strings["anomaly_none"]
        lines.append("")
        return lines

    lines += [line.format(n=len(anomalies),
                          n_models=anomalies["model_id"].nunique())
              for line in strings["anomaly_total"]]
    lines.append("")
    lines += [strings["anomaly_header"], strings["anomaly_sep"]]
    for _, row in anomaly_summary(anomalies).iterrows():
        lines.append(f"| {row['variant']} | {row['option']} | "
                     f"{row['runs']} | {row['models']} |")
    lines.append("")
    return lines


def _coverage_section(ts_diff: pd.DataFrame | None, strings: dict) -> list[str]:
    """Time-series comparisons computed over an incomplete overlap.

    Three outcomes, all stated rather than inferred from a missing table:
    coverage was never assessed (no `ts_diff`, or one predating the coverage
    columns), every comparison covered both grids in full, or some did not --
    in which case they are counted per comparison and attributed by model.
    """
    lines = [strings["coverage_anomaly_heading"], ""]
    lines += strings["coverage_anomaly_intro"]
    lines.append("")

    if not coverage_was_assessed(ts_diff):
        lines += strings["coverage_not_assessed"]
        lines.append("")
        return lines

    anomalies = coverage_anomalies(ts_diff)
    if anomalies.empty:
        lines += strings["coverage_anomaly_none"]
        lines.append("")
        return lines

    # Distinct models ACROSS the comparisons, not the sum of the table's
    # per-comparison counts: a single truncated run is flagged by every
    # comparison it takes part in, and summing them would announce one
    # stopped simulation as five models.
    lines += [line.format(n_models=coverage_anomaly_models(ts_diff),
                          n_comparisons=len(anomalies))
              for line in strings["coverage_anomaly_total"]]
    lines.append("")
    lines += [strings["coverage_anomaly_header"], strings["coverage_anomaly_sep"]]
    for _, row in anomalies.iterrows():
        coverage = row["min_coverage"]
        rendered = "n/a" if pd.isna(coverage) else f"{float(coverage):.4f}"
        lines.append(f"| {row['comparison']} | {row['models']} | "
                     f"{row['series']} | {rendered} |")
    lines.append("")
    return lines


def _interaction_section(deltas: pd.DataFrame, strings: dict) -> list[str]:
    """The 2x2 factorial interaction, with its sign convention spelled out.

    The sign is stated in words because a bare number cannot carry it: a
    reader looking at `-0.4` has no way to tell whether it means *Anderson
    helps more under Crank-Nicolson* or the reverse. It is the same table
    shape as the five simple-effect sections, so the numbers stay directly
    comparable against them.
    """
    lines = [strings["interaction_heading"], ""]
    lines += strings["interaction_intro"]
    lines.append("")
    lines += strings["interaction_sign"]
    lines.append("")
    lines += [strings["metric_header"], strings["metric_sep"]]

    emitted = 0
    if INTERACTION_COLUMN in deltas.columns:
        grouped = deltas.dropna(subset=[INTERACTION_COLUMN]).groupby("metric")
        for metric, group in grouped:
            column = group[INTERACTION_COLUMN].astype(float)
            lines.append(
                f"| {metric} | {len(column)} | {column.mean():.4f} | "
                f"{column.median():.4f} | {column.min():.4f} | "
                f"{column.max():.4f} |"
            )
            emitted += 1
    lines.append("")

    # An empty table here means neither simple effect survived for any
    # metric, which is *not measured* -- not *the two factors do not
    # interact*. Said outright, for the same reason the iteration sections
    # announce their own emptiness.
    if not emitted:
        lines += strings["interaction_empty"]
        lines.append("")

    return lines


def _exclusion_section(
    deltas: pd.DataFrame,
    runs: pd.DataFrame,
    strings: dict,
    emitted: int,
) -> list[str]:
    """Iteration deltas the routing-model guard withheld, itemised by cause.

    `emitted` is how many rows the iteration-shift sections actually printed;
    zero means nothing was measured at all, which the report must state in
    words rather than leaving four bare empty tables to be read as a result.
    """
    excluded = iteration_exclusions(deltas, runs)
    lines = [strings["iter_excluded_heading"], ""]
    lines += strings["iter_excluded_intro"]
    lines.append("")

    if excluded.empty:
        lines += strings["iter_excluded_none"]
        lines.append("")
    else:
        lines += [strings["iter_excluded_header"], strings["iter_excluded_sep"]]
        for _, row in excluded.iterrows():
            lines.append(f"| {row['cause']} | {row['pairings']} | "
                         f"{row['models']} |")
        lines.append("")

        unknown = excluded[excluded["cause"] == EXCLUSION_ROUTING_UNKNOWN]
        if not unknown.empty:
            lines += strings["iter_unknown_note"]
            lines.append("")

    if not emitted:
        lines += strings["iter_empty_note"]
        lines.append("")

    return lines


def _surcharge_section(
    deltas: pd.DataFrame,
    runs: pd.DataFrame,
    elements: pd.DataFrame | None,
    strings: dict,
) -> list[str]:
    """The D/E surcharge-activity strata, with their honesty note.

    An empty (or near-empty) `active` stratum is stated in words. Printing
    only an `inactive` mean would read as "the feature has no effect", when
    what actually happened is that the feature was never exercised -- and a
    markdown table cannot tell those apart on its own.

    The announcement is made PER AXIS as well as globally. A store can hold
    fifty surcharge-active models and still produce no `active` row for
    `D - C` -- every one of those deltas null for its own reason -- and the
    global note, which only fires when the active set itself is empty, would
    stay silent while that axis printed `inactive` rows alone: exactly the
    misreading this section exists to block.
    """
    active = surcharge_active_models(runs, elements) & set(deltas["model_id"])
    lines = [strings["surcharge_heading"], ""]
    lines += strings["surcharge_proxy"]
    lines.append("")

    if not active:
        lines += strings["surcharge_none"]
        lines.append("")
    elif len(active) < SURCHARGE_ACTIVE_MIN:
        lines += [line.format(n=len(active)) for line in strings["surcharge_few"]]
        lines.append("")

    lines += [strings["surcharge_header"], strings["surcharge_sep"]]
    strata = (
        (SURCHARGE_ACTIVE, deltas[deltas["model_id"].isin(active)]),
        (SURCHARGE_INACTIVE, deltas[~deltas["model_id"].isin(active)]),
    )
    empty_axes: list[str] = []
    for axis, delta_col in SURCHARGE_SECTIONS:
        emitted = 0
        for stratum, subset in strata:
            if subset.empty or delta_col not in subset.columns:
                continue
            for metric, group in subset.dropna(subset=[delta_col]).groupby("metric"):
                column = group[delta_col].astype(float)
                if column.empty:
                    continue
                lines.append(
                    f"| {axis} | {stratum} | {metric} | {len(column)} | "
                    f"{column.mean():.4f} | {column.median():.4f} |"
                )
                if stratum == SURCHARGE_ACTIVE:
                    emitted += 1
        if active and not emitted:
            empty_axes.append(axis)
    lines.append("")

    # After the table, so the notes never interrupt it. Only when the store
    # HAS surcharge-active models: when it has none, the global note above
    # already says so once, and repeating it per axis would bury it.
    for axis in empty_axes:
        lines += [line.format(axis=axis) for line in strings["surcharge_axis_empty"]]
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Element-level comparison
# ---------------------------------------------------------------------------

ELEMENT_KEY = ["model_id", "family", "element_type", "element_id", "metric"]


def build_element_deltas(elements: pd.DataFrame) -> pd.DataFrame:
    """Per-element A/B/C/D/E/REF values, all deltas and the interaction.

    No commensurability guard applies here: element metrics are depths,
    flows and volumes in fixed units, not iteration counters, so none of the
    three confounds `build_deltas` screens for can arise. `delta_interaction`
    is still formed the same way -- from `delta_d_minus_c` and
    `delta_b_minus_a` -- so it nulls whenever any of its four operands does.
    """
    if elements.empty:
        return pd.DataFrame()

    # `elements` is read back from a dataset partitioned by `family`
    # (store.write_table(..., partition_by=["family"])), so pandas restores
    # `family` as a `category` dtype. pivot_table's index includes `family`
    # (via ELEMENT_KEY); pivoting on a categorical index materialises the
    # full cartesian product of categories by default, which with 20+
    # families and thousands of elements is a real blow-up, not a rounding
    # error. Cast to plain str first so only the categories actually present
    # form the index -- the same hazard write_markdown documents and fixes
    # with observed=True for groupby.
    elements = elements.copy()
    if isinstance(elements["family"].dtype, pd.CategoricalDtype):
        elements["family"] = elements["family"].astype(str)

    wide = elements.pivot_table(
        index=ELEMENT_KEY, columns="variant", values="value", aggfunc="first"
    ).reset_index()

    for variant in DELTA_VARIANTS:
        if variant not in wide.columns:
            wide[variant] = pd.NA

    wide = wide.rename(columns={variant: f"value_{variant.lower()}"
                                for variant in DELTA_VARIANTS})

    for column, left, right in DELTA_SPECS:
        wide[column] = (wide[f"value_{left.lower()}"]
                        - wide[f"value_{right.lower()}"])

    # Formed from the two simple-effect columns, exactly as in `build_deltas`,
    # so the element table's interaction is null under the same conditions as
    # the scalar one -- here that reduces to "any of the four operands is
    # missing", since no commensurability guard applies at element level.
    _column, minuend, subtrahend = INTERACTION_SPEC
    wide[INTERACTION_COLUMN] = wide[minuend] - wide[subtrahend]

    wide = _attach_element_case_ids(wide, elements)
    return wide[ELEMENT_KEY + VALUE_COLUMNS + DELTA_COLUMNS + CASE_ID_COLUMNS]


def _attach_element_case_ids(
    wide: pd.DataFrame, elements: pd.DataFrame,
) -> pd.DataFrame:
    """Join each variant's source `case_id` onto the pivoted element rows.

    Pivoted the same way and on the same key as the values themselves, so a
    row's `case_id_c` is by construction the case its `value_c` came from --
    not a lookup that could pick a different run of the same model.
    """
    if "case_id" not in elements.columns:
        for column in CASE_ID_COLUMNS:
            wide[column] = pd.NA
        return wide

    cases = elements.pivot_table(
        index=ELEMENT_KEY, columns="variant", values="case_id", aggfunc="first"
    ).rename(columns={variant: f"case_id_{variant.lower()}"
                      for variant in DELTA_VARIANTS})
    for column in CASE_ID_COLUMNS:
        if column not in cases.columns:
            cases[column] = pd.NA
    return wide.merge(cases[CASE_ID_COLUMNS].reset_index(),
                      on=ELEMENT_KEY, how="left")


def topology_status(elements: pd.DataFrame) -> pd.DataFrame:
    """Models whose reference report shares no element ids with variant A.

    Their scalars remain comparable; only the per-element join does not
    apply. Left unnamed, an empty join would read as "no differences".
    """
    if elements.empty:
        return pd.DataFrame(columns=["model_id", "status"])

    rows: list[dict] = []
    for model_id, group in elements.groupby("model_id"):
        ours = set(group.loc[group.variant == schema.VARIANT_A, "element_id"])
        theirs = set(group.loc[group.variant == schema.VARIANT_REF, "element_id"])
        if ours and theirs and not (ours & theirs):
            rows.append({
                "model_id": model_id,
                "status": schema.Status.REF_TOPOLOGY_MISMATCH,
            })
    return pd.DataFrame(rows, columns=["model_id", "status"])
