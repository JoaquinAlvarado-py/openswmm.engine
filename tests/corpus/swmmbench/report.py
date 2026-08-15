"""Materialise the comparison axes and a human-readable summary.

`B - A` isolates Anderson acceleration. `C - A` isolates semi-implicit
(Crank-Nicolson) node continuity. `A - REF` measures parity debt against EPA
SWMM 5.2. Keeping all of them side by side rather than collapsed is the
reason the A/B/C matrix exists: folding two feature changes into one variant
would make an observed shift unattributable between their two causes.
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

#: Metrics whose comparability depends on iteration_metric_kind matching.
ITERATION_METRICS = {"avg_iterations_per_step", "pct_steps_not_converging"}


def build_deltas(runs: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """One row per (model, metric) with A, B, C, REF values and all deltas."""
    if runs.empty:
        return pd.DataFrame()

    present = [m for m in metrics if m in runs.columns]
    if not present:
        return pd.DataFrame()

    kinds = (
        runs.pivot_table(index="model_id", columns="variant",
                         values="iteration_metric_kind", aggfunc="first")
        if "iteration_metric_kind" in runs.columns else pd.DataFrame()
    )

    families = runs.groupby("model_id")["family"].first()
    rows: list[dict] = []

    for metric in present:
        wide = runs.pivot_table(index="model_id", columns="variant",
                                values=metric, aggfunc="first")
        for model_id, record in wide.iterrows():
            value_a = record.get(schema.VARIANT_A)
            value_b = record.get(schema.VARIANT_B)
            value_c = record.get(schema.VARIANT_C)
            value_ref = record.get(schema.VARIANT_REF)

            # Picard iterations and FV substeps are different counters;
            # subtracting one from the other would be meaningless. The
            # exclusion is per-delta, not per-row, and only trips when BOTH
            # kinds involved are actually known and differ -- a missing
            # kind (e.g. B crashed or timed out) is not a mismatch, and must
            # not cost the model its A-REF parity-debt delta.
            kind_mismatch_b = False
            kind_mismatch_c = False
            kind_mismatch_ref = False
            if (metric in ITERATION_METRICS and not kinds.empty
                    and model_id in kinds.index):
                kind = kinds.loc[model_id]
                kind_a = kind.get(schema.VARIANT_A)
                kind_b = kind.get(schema.VARIANT_B)
                kind_c = kind.get(schema.VARIANT_C)
                kind_ref = kind.get(schema.VARIANT_REF)
                kind_mismatch_b = (pd.notna(kind_a) and pd.notna(kind_b)
                                    and kind_a != kind_b)
                kind_mismatch_c = (pd.notna(kind_a) and pd.notna(kind_c)
                                    and kind_a != kind_c)
                # The reference is EPA SWMM 5.2, which always reports Picard
                # iterations; an FV-routed A run is equally incommensurable
                # against it.
                kind_mismatch_ref = (pd.notna(kind_a) and pd.notna(kind_ref)
                                      and kind_a != kind_ref)

            rows.append({
                "model_id": model_id,
                "family": families.get(model_id),
                "metric": metric,
                "value_a": value_a,
                "value_b": value_b,
                "value_c": value_c,
                "value_ref": value_ref,
                "delta_b_minus_a": (value_b - value_a)
                    if pd.notna(value_a) and pd.notna(value_b)
                    and not kind_mismatch_b else pd.NA,
                "delta_c_minus_a": (value_c - value_a)
                    if pd.notna(value_a) and pd.notna(value_c)
                    and not kind_mismatch_c else pd.NA,
                "delta_a_minus_ref": (value_a - value_ref)
                    if pd.notna(value_a) and pd.notna(value_ref)
                    and not kind_mismatch_ref else pd.NA,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Engine-echoed option values vs. variant intent
# ---------------------------------------------------------------------------

#: Options whose engine-reported value can be checked against the variant's
#: stated intent, keyed by the `runs` column that carries the echo
#: (DefaultReportPlugin.cpp prints both under the DYNWAVE-only Analysis
#: Options block; rptparse.parse_scalars reads them into these columns).
#: OptionsHandler.cpp silently ignores an unrecognised value for either --
#: no `else`, no warning -- so `options_applied` (what the harness's deck
#: asked for) cannot alone prove the engine actually did it. Without this
#: check, a typo in an option name or value would read as "the feature has
#: no effect" across the whole corpus instead of as a broken deck.
OPTION_ECHO_COLUMNS = {
    "NODE_CONTINUITY": "reported_node_continuity",
    "ANDERSON_ACCEL": "reported_anderson_accel",
}

ANOMALY_COLUMNS = ["model_id", "variant", "option", "expected", "reported"]


def option_anomalies(runs: pd.DataFrame) -> pd.DataFrame:
    """Executed runs whose engine-reported option value contradicts intent.

    A row with no echo (FV/STEADY/KINWAVE routing never prints the block at
    all) is not an anomaly: there is nothing to contradict, so it is silently
    skipped rather than flagged.
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
            if reported is None or (isinstance(reported, float) and pd.isna(reported)):
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


def write_markdown(deltas: pd.DataFrame, runs: pd.DataFrame, path: Path) -> Path:
    """Write a summary readable without opening a notebook."""
    path = Path(path)
    lines = ["# Corpus Benchmark Summary", ""]

    if not runs.empty and "status" in runs.columns:
        lines += ["## Coverage", "", "| status | runs |", "| --- | --- |"]
        for status, count in runs["status"].value_counts().items():
            lines.append(f"| {status} | {count} |")
        lines.append("")

    if not deltas.empty:
        # B - A isolates Anderson acceleration; C - A isolates semi-implicit
        # (Crank-Nicolson) node continuity. Kept as two separate sections
        # rather than columns of one table so neither axis reads as derived
        # from, or secondary to, the other.
        for delta_col, heading in (
            ("delta_b_minus_a", "## B - A by metric (Anderson acceleration effect)"),
            ("delta_c_minus_a", "## C - A by metric (Crank-Nicolson continuity effect)"),
        ):
            lines += [heading, "",
                      "| metric | models | mean | median | min | max |",
                      "| --- | --- | --- | --- | --- | --- |"]
            # `metric` here is plain object dtype (built row-by-row in
            # build_deltas), so groupby's default observed=False is harmless.
            grouped = deltas.dropna(subset=[delta_col]).groupby("metric")
            for metric, group in grouped:
                column = group[delta_col].astype(float)
                lines.append(
                    f"| {metric} | {len(column)} | {column.mean():.4f} | "
                    f"{column.median():.4f} | {column.min():.4f} | {column.max():.4f} |"
                )
            lines.append("")

        iterations = deltas[deltas.metric == "avg_iterations_per_step"]
        for delta_col, heading in (
            ("delta_b_minus_a", "## Iteration shift by family (B - A)"),
            ("delta_c_minus_a", "## Iteration shift by family (C - A)"),
        ):
            lines += [heading, "",
                      "| family | models | mean abs iterations |",
                      "| --- | --- | --- |"]
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
                column = group[delta_col].dropna().astype(float).abs()
                if column.empty:
                    continue
                lines.append(f"| {family} | {len(column)} | {column.mean():.4f} |")
            lines.append("")

    lines += [
        "## Caveats",
        "",
        "- `total_iterations_est` is a derived **estimate**",
        "  (`avg_iterations_per_step x duration / avg_step`); the report",
        "  publishes only a two-decimal mean. Its error grows on",
        "  variable-time-step models. Use `avg_iterations_per_step` for any",
        "  conclusion that matters.",
        "- Rows whose `iteration_metric_kind` differs from A's are excluded",
        "  from that variant's iteration deltas (B - A, C - A independently):",
        "  FV substeps are not Picard iterations.",
        "- Time series are compared only between our own runs. The external",
        "  anchor is summary-level.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Element-level comparison
# ---------------------------------------------------------------------------

ELEMENT_KEY = ["model_id", "family", "element_type", "element_id", "metric"]


def build_element_deltas(elements: pd.DataFrame) -> pd.DataFrame:
    """Per-element A/B/C/REF values and all deltas."""
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

    for variant in (schema.VARIANT_A, schema.VARIANT_B, schema.VARIANT_C,
                    schema.VARIANT_REF):
        if variant not in wide.columns:
            wide[variant] = pd.NA

    wide = wide.rename(columns={
        schema.VARIANT_A: "value_a",
        schema.VARIANT_B: "value_b",
        schema.VARIANT_C: "value_c",
        schema.VARIANT_REF: "value_ref",
    })
    wide["delta_b_minus_a"] = wide["value_b"] - wide["value_a"]
    wide["delta_c_minus_a"] = wide["value_c"] - wide["value_a"]
    wide["delta_a_minus_ref"] = wide["value_a"] - wide["value_ref"]
    return wide[ELEMENT_KEY + ["value_a", "value_b", "value_c", "value_ref",
                               "delta_b_minus_a", "delta_c_minus_a",
                               "delta_a_minus_ref"]]


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
