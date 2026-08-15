"""Materialise the two comparison axes and a human-readable summary.

`B - A` isolates the effect of the new options. `A - REF` measures parity
debt against EPA SWMM 5.2. Keeping them side by side rather than collapsed
is the reason the A/B/C matrix exists.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema

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
    """One row per (model, metric) with A, B, REF values and both deltas."""
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
            value_ref = record.get(schema.VARIANT_REF)

            if metric in ITERATION_METRICS and not kinds.empty:
                kind = kinds.loc[model_id] if model_id in kinds.index else None
                if kind is not None:
                    kind_a = kind.get(schema.VARIANT_A)
                    kind_b = kind.get(schema.VARIANT_B)
                    # Picard iterations and FV substeps are different counters.
                    # Subtracting one from the other would be meaningless.
                    if kind_a != kind_b:
                        continue

            rows.append({
                "model_id": model_id,
                "family": families.get(model_id),
                "metric": metric,
                "value_a": value_a,
                "value_b": value_b,
                "value_ref": value_ref,
                "delta_b_minus_a": (value_b - value_a)
                    if pd.notna(value_a) and pd.notna(value_b) else pd.NA,
                "delta_a_minus_ref": (value_a - value_ref)
                    if pd.notna(value_a) and pd.notna(value_ref) else pd.NA,
            })

    return pd.DataFrame(rows)


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
        lines += ["## B - A by metric (feature effect)", "",
                  "| metric | models | mean | median | min | max |",
                  "| --- | --- | --- | --- | --- | --- |"]
        # `metric` is plain object dtype here (built row-by-row above), so
        # groupby's default observed=False is harmless. `family`, however,
        # round-trips out of a partitioned Parquet dataset as a pandas
        # `category` dtype; grouping it with observed=False would
        # materialise every category in the dataset -- including families
        # with zero rows in `deltas` -- as spurious empty-group rows below.
        grouped = deltas.dropna(subset=["delta_b_minus_a"]).groupby("metric")
        for metric, group in grouped:
            column = group["delta_b_minus_a"].astype(float)
            lines.append(
                f"| {metric} | {len(column)} | {column.mean():.4f} | "
                f"{column.median():.4f} | {column.min():.4f} | {column.max():.4f} |"
            )
        lines.append("")

        lines += ["## Iteration shift by family", "",
                  "| family | models | mean abs B-A iterations |",
                  "| --- | --- | --- |"]
        iterations = deltas[deltas.metric == "avg_iterations_per_step"]
        for family, group in iterations.groupby("family", observed=True):
            column = group["delta_b_minus_a"].dropna().astype(float).abs()
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
        "- Rows whose `iteration_metric_kind` differs between A and B are",
        "  excluded from iteration deltas: FV substeps are not Picard",
        "  iterations.",
        "- Time series are compared only between our own runs. The external",
        "  anchor is summary-level.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
