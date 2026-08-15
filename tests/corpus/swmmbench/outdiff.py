"""B - A time-series comparison.

Diffs are aggregated per element and attribute. Retaining raw series for 878
models across two configurations would run to billions of rows, so each pair
of `.out` files is read, reduced, and discarded.

The `openswmm` binding is imported lazily: it requires a built engine, and
every other stage of the harness must remain usable without one.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

ABS_TOL = 1e-9


def diff_series(
    a: dict[datetime, float],
    b: dict[datetime, float],
    abs_tol: float = ABS_TOL,
) -> dict:
    """Reduce two aligned series to divergence statistics."""
    times = sorted(set(a) & set(b))
    if not times:
        return {"max_abs": None, "max_rel": None, "rmse": None,
                "first_div_period": None, "first_div_time": None,
                "n_periods": 0}

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
        "first_div_period": first,
        "first_div_time": times[first] if first is not None else None,
        "n_periods": len(times),
    }


#: Attributes diffed per element type, as (accessor name, attribute enum name).
NODE_ATTRIBUTES = ("INVERT_DEPTH", "HYDRAULIC_HEAD", "TOTAL_INFLOW",
                   "FLOODING_LOSSES")
LINK_ATTRIBUTES = ("FLOW_RATE", "FLOW_DEPTH", "FLOW_VELOCITY", "CAPACITY")


def diff_out_files(a_out: Path, b_out: Path, rel_tol: float) -> list[dict]:
    """Per-element, per-attribute divergence between two `.out` files."""
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
                        abs_tol=rel_tol,
                    )
                    if stats["n_periods"] == 0:
                        continue
                    rows.append({
                        "element_type": element_type.name,
                        "element_id": name,
                        "attribute": attribute,
                        **stats,
                    })
    return rows
