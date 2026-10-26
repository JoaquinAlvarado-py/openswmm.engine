"""Pairwise time-series comparison of two `.out` files.

The pairs are whatever `cli.DIFF_COMPARISONS` names -- four A-anchored
comparisons plus D - C, over the five executed variants. Nothing here is
anchored on A: `diff_out_files(first, second)` reduces `second - first` for
any two files.

Diffs are aggregated per element and attribute. Retaining raw series for a
~1700-model corpus across five configurations would run to billions of rows,
so each pair of `.out` files is read, reduced, and discarded.

Because the raw series are discarded, every row must carry enough evidence to
say what it was computed OVER: a comparison covering only the timestamps two
runs share is otherwise indistinguishable from one covering the whole run,
and a run that stopped halfway would publish an excellent `max_abs`. That is
what `series_coverage` is for.

The `openswmm` binding is imported lazily: it requires a built engine, and
every other stage of the harness must remain usable without one.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

ABS_TOL = 1e-9


#: Every field `series_coverage` contributes to a `diff_series` result, in
#: report order. Named once so `diff_out_files`, the `ts_diff` schema and the
#: summary's coverage-anomaly section cannot disagree about the field set.
COVERAGE_FIELDS = (
    "n_periods_left", "n_periods_right", "n_common", "coverage_fraction",
    "start_time_match", "end_time_match", "time_grid_match",
)


def series_coverage(
    a: dict[datetime, float],
    b: dict[datetime, float],
) -> dict:
    """How much of the two series' reporting grids the comparison covers.

    `diff_series` compares only the timestamps the two series share, so a run
    that stopped halfway through still produces divergence statistics -- over
    the half it managed. Without these fields a near-perfect `max_abs`
    computed over twelve of twenty-four hours is indistinguishable from one
    computed over the whole run, which is precisely the silent failure this
    benchmark exists to catch.

    The fields, all derived from the two sets of timestamps alone:

    * `n_periods_left` / `n_periods_right` -- distinct timestamps reported by
      each side. Both are given because "shorter" is not "truncated": either
      side can be the short one.
    * `n_common` -- timestamps present in BOTH, i.e. the number of periods
      `diff_series` actually reduced over.
    * `coverage_fraction` -- **`n_common` divided by the number of distinct
      timestamps appearing in EITHER series** (the size of the union), not by
      either side's own length and not by a span in hours. Stated as a ratio
      of period COUNTS so the denominator is a thing the reader can see in
      the two columns beside it. It is 1.0 exactly when the two grids are
      identical, and it falls below 1.0 whenever either side reports an
      instant the other does not -- from a truncated run, a late start, or a
      different reporting interval alike. `None` when neither side reported
      anything at all, because 0/0 is not a coverage of zero.
    * `start_time_match` / `end_time_match` -- whether the two series' first
      and last reported instants coincide. A truncated right-hand run is the
      case where `start_time_match` holds and `end_time_match` does not.
    * `time_grid_match` -- whether the two sets of timestamps are equal. This
      is what separates *same span, different sampling* (both endpoints
      match, the grid does not) from *same grid* (all three hold). `None`
      only when both series are empty, where there is no grid to compare.
    """
    left, right = set(a), set(b)
    common, union = left & right, left | right
    both = bool(left) and bool(right)
    return {
        "n_periods_left": len(left),
        "n_periods_right": len(right),
        "n_common": len(common),
        "coverage_fraction": (len(common) / len(union)) if union else None,
        "start_time_match": (min(left) == min(right)) if both else None,
        "end_time_match": (max(left) == max(right)) if both else None,
        "time_grid_match": (left == right) if union else None,
    }


def diff_series(
    a: dict[datetime, float],
    b: dict[datetime, float],
    abs_tol: float = ABS_TOL,
) -> dict:
    """Reduce two series to divergence statistics plus coverage evidence.

    The divergence statistics (`max_abs`, `max_rel`, `rmse`,
    `first_div_period`, `first_div_time`) are reduced over the timestamps the
    two series SHARE, and nothing else. They are reported for an incomplete
    overlap rather than withheld -- "the two runs agreed exactly until one of
    them stopped" is a real finding, and destroying it would remove the very
    evidence that identifies the truncation as the whole story -- but they are
    then statistics about a sub-span, and every field of `series_coverage`
    ships on the same row so no reader can mistake one for a full comparison.
    `report.coverage_anomalies` turns an incomplete overlap into a counted,
    attributed anomaly in `summary.md`; see `COVERAGE_FIELDS`.

    `n_common` is the number of periods actually compared. There is no
    separate `n_periods`: one count named for what it is beats two names for
    one number, and the two per-side counts sit beside it.
    """
    coverage = series_coverage(a, b)
    times = sorted(set(a) & set(b))
    if not times:
        return {"max_abs": None, "max_rel": None, "rmse": None,
                "first_div_period": None, "first_div_time": None,
                **coverage}

    left = np.array([a[t] for t in times], dtype=np.float64)
    right = np.array([b[t] for t in times], dtype=np.float64)
    delta = np.abs(right - left)

    # Scale by the baseline magnitude, falling back to 1.0 where the baseline
    # is zero so a 0 -> 5 step reports as a finite ratio rather than infinity.
    scale = np.where(np.abs(left) > abs_tol, np.abs(left), 1.0)

    diverged = np.flatnonzero(delta > abs_tol)
    first = int(diverged[0]) if diverged.size else None

    return {
        "max_abs": float(delta.max()),
        "max_rel": float((delta / scale).max()),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        # Indexed into the COMMON grid, not into either side's own: it is the
        # first compared period past tolerance, which is the only period
        # ordering both series agree on.
        "first_div_period": first,
        "first_div_time": times[first] if first is not None else None,
        **coverage,
    }


#: Attributes diffed per element type, as (accessor name, attribute enum name).
NODE_ATTRIBUTES = ("INVERT_DEPTH", "HYDRAULIC_HEAD", "TOTAL_INFLOW",
                   "FLOODING_LOSSES")
LINK_ATTRIBUTES = ("FLOW_RATE", "FLOW_DEPTH", "FLOW_VELOCITY", "CAPACITY")


def diff_out_files(a_out: Path, b_out: Path, abs_tol: float) -> list[dict]:
    """Per-element, per-attribute divergence between two `.out` files.

    `abs_tol` is absolute: it is compared against `|b - a|` directly. The
    reported `max_rel` is scaled by the baseline but is never thresholded.
    """
    a_out, b_out = Path(a_out), Path(b_out)
    if not (a_out.is_file() and b_out.is_file()):
        return []

    from openswmm.legacy.output import (  # noqa: PLC0415  lazy by design
        ElementType, LinkAttribute, NodeAttribute, Output,
    )

    rows: list[dict] = []
    with Output(str(a_out)) as left, Output(str(b_out)) as right:
        for element_type, attributes, enum, getter in (
            (ElementType.NODE, NODE_ATTRIBUTES, NodeAttribute,
             "get_node_timeseries"),
            (ElementType.LINK, LINK_ATTRIBUTES, LinkAttribute,
             "get_link_timeseries"),
        ):
            names_a = left.get_element_names(element_type)
            names_b = set(right.get_element_names(element_type))
            for name in names_a:
                if name not in names_b:
                    continue  # topology differs; scalars still comparable
                for attribute in attributes:
                    code = getattr(enum, attribute)
                    stats = diff_series(
                        getattr(left, getter)(name, code),
                        getattr(right, getter)(name, code),
                        abs_tol=abs_tol,
                    )
                    # Nothing at all in common: there is no comparison to
                    # report, only two disjoint series. The pairing is not a
                    # partial-coverage anomaly, it is an absent row.
                    if stats["n_common"] == 0:
                        continue
                    rows.append({
                        "element_type": element_type.name,
                        "element_id": name,
                        "attribute": attribute,
                        **stats,
                    })
    return rows
