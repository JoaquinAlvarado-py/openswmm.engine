"""Parse SWMM `.rpt` reports into tabular records.

One parser serves our own runs and the corpus reference reports alike. A
separate reference parser would drift from the run parser and manufacture
differences that are artifacts of parsing rather than of the engine.

Reports are Latin-1: SI reports carry characters such as `ft³`.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from . import schema

ENCODING = "latin-1"

#: Continuity blocks, keyed by the title line that opens them.
CONTINUITY_BLOCKS = {
    "Runoff Quantity Continuity": "continuity_error_runoff",
    "Flow Routing Continuity": "continuity_error_flow",
    "Quality Routing Continuity": "continuity_error_quality",
    "2D Surface Routing Continuity": "continuity_error_2d",
}

SCALAR_KEYS = (
    "reported_version", "start_date", "end_date", "duration_s",
    *CONTINUITY_BLOCKS.values(),
    "min_step", "avg_step", "max_step", "pct_steady_state",
    "avg_iterations_per_step", "iteration_metric_kind",
    "pct_steps_not_converging", "n_steps_est", "total_iterations_est",
)

_CONTINUITY_ERROR = re.compile(r"Continuity Error \(%\)\s*\.+\s*(-?[\d.]+)")
_DOTTED = re.compile(r"^\s*(?P<label>.+?)\s*\.{3,}\s*(?P<value>.*?)\s*$")
_COLON = re.compile(r"^\s*(?P<label>.+?)\s*:\s*(?P<value>.*?)\s*$")
_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

_TIME_STEP_FIELDS = {
    "Minimum Time Step": "min_step",
    "Average Time Step": "avg_step",
    "Maximum Time Step": "max_step",
    "% of Time in Steady State": "pct_steady_state",
}


def read(path: Path) -> str:
    return Path(path).read_text(encoding=ENCODING, errors="replace")


def _number(text: str) -> float | None:
    """First float in `text`, or None. `n/a` and blanks yield None."""
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _parse_date(text: str) -> datetime | None:
    try:
        return datetime.strptime(text.strip(), _DATE_FORMAT)
    except ValueError:
        return None


def parse_scalars(text: str) -> dict:
    """Extract per-run scalars. Every key in SCALAR_KEYS is always present."""
    out: dict = {key: None for key in SCALAR_KEYS}
    lines = text.splitlines()

    for line in lines[:3]:
        if "VERSION" in line.upper():
            out["reported_version"] = line.strip()
            break

    for line in lines:
        dotted = _DOTTED.match(line)
        if not dotted:
            continue
        label = dotted.group("label").strip()
        if label == "Starting Date":
            out["start_date"] = _parse_date(dotted.group("value"))
        elif label == "Ending Date":
            out["end_date"] = _parse_date(dotted.group("value"))

    if out["start_date"] and out["end_date"]:
        out["duration_s"] = (out["end_date"] - out["start_date"]).total_seconds()

    # Continuity: from each block title, take the first "Continuity Error (%)"
    # that follows it. Blocks are short and always terminated by that line.
    for index, line in enumerate(lines):
        title = line.strip()
        field = CONTINUITY_BLOCKS.get(title.split("  ")[0].strip())
        if field is None or out[field] is not None:
            continue
        for candidate in lines[index: index + 40]:
            error = _CONTINUITY_ERROR.search(candidate)
            if error:
                out[field] = float(error.group(1))
                break

    for line in lines:
        colon = _COLON.match(line)
        if not colon:
            continue
        label = colon.group("label").strip()
        value = colon.group("value").strip()

        if label in _TIME_STEP_FIELDS:
            out[_TIME_STEP_FIELDS[label]] = _number(value)
        elif label == "Average Iterations per Step":
            out["avg_iterations_per_step"] = _number(value)
            out["iteration_metric_kind"] = schema.ITER_PICARD
        elif label == "Average Substeps per Step":
            out["avg_iterations_per_step"] = _number(value)
            out["iteration_metric_kind"] = schema.ITER_FV
        elif label == "% of Steps Not Converging":
            # `n/a` under FV routing. Zero would read as perfect convergence.
            out["pct_steps_not_converging"] = _number(value)

    avg_step = out["avg_step"]
    if out["duration_s"] and avg_step:
        out["n_steps_est"] = out["duration_s"] / avg_step
        if out["avg_iterations_per_step"] is not None:
            out["total_iterations_est"] = (
                out["n_steps_est"] * out["avg_iterations_per_step"]
            )

    return out
