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
    "reported_routing_model", "reported_surcharge_method",
    "reported_node_continuity", "reported_anderson_accel",
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
        elif label == "Flow Routing Method":
            # DefaultReportPlugin.cpp:556. Printed for EVERY routing model
            # (FV, DYNWAVE, KINWAVE, STEADY) -- unlike the three echoes
            # below, this one is never gated and is the stratification key
            # for them: its presence does not imply theirs.
            out["reported_routing_model"] = dotted.group("value").strip()
        elif label == "Surcharge Method":
            # DefaultReportPlugin.cpp:560, inside the `if (rm == 2)` block
            # (lines 558-567) -- printed only when routing is DYNWAVE. Absent
            # for FV/KINWAVE/STEADY reports; that absence is not an anomaly.
            out["reported_surcharge_method"] = dotted.group("value").strip()
        elif label == "Node Continuity":
            # DefaultReportPlugin.cpp:562-564 echoes what OptionsHandler.cpp
            # actually resolved NODE_CONTINUITY to. OptionsHandler.cpp:443
            # silently ignores an unrecognised value (no else, no warning),
            # so `options_applied` alone (the harness's intent) cannot prove
            # the deck's option was actually honoured -- only present when
            # routing is DYNWAVE.
            out["reported_node_continuity"] = dotted.group("value").strip()
        elif label == "Anderson Acceleration":
            # DefaultReportPlugin.cpp:565-566, same DYNWAVE-only gate.
            out["reported_anderson_accel"] = dotted.group("value").strip()

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
                try:
                    out[field] = float(error.group(1))
                    break
                except ValueError:
                    # Malformed continuity error value — skip and continue looking
                    continue

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


# ---------------------------------------------------------------------------
# Per-element convergence rankings
# ---------------------------------------------------------------------------

#: Ranking block title -> metric name emitted for its entries.
RANKING_BLOCKS = {
    "Highest Continuity Errors": "continuity_error_pct",
    "Time-Step Critical Elements": "time_step_critical",
    "Highest Flow Instability Indexes": "flow_instability_index",
    "Most Frequent Nonconverging Nodes": "nonconverging_pct",
}

#: Prose the engine prints instead of entries when a ranking is empty. These
#: are statements of health, not parse failures.
EMPTY_SENTINELS = (
    "none",
    "all links are stable.",
    "convergence obtained at all time steps.",
    "all nodes converged.",
)

#: `Node 10208 (3.68%)` and `Link 8060 (1)` — the percent sign is optional
#: because the instability index is a bare count.
_RANKING_ENTRY = re.compile(
    r"^\s*(?P<kind>Node|Link)\s+(?P<id>\S+)\s+\((?P<value>-?[\d.]+)%?\)\s*$"
)


def parse_rankings(text: str) -> list[dict]:
    """Extract per-element ranking entries from every ranking block."""
    lines = text.splitlines()
    rows: list[dict] = []

    for index, line in enumerate(lines):
        metric = RANKING_BLOCKS.get(line.strip())
        if metric is None:
            continue
        # The title sits inside a three-line asterisk banner; entries begin
        # after the closing banner line.
        cursor = index + 1
        if cursor < len(lines) and set(lines[cursor].strip()) == {"*"}:
            cursor += 1

        while cursor < len(lines):
            body = lines[cursor].strip()
            if not body:
                break
            if body.lower() in EMPTY_SENTINELS:
                break
            entry = _RANKING_ENTRY.match(lines[cursor])
            if not entry:
                break
            try:
                value = float(entry.group("value"))
            except ValueError:
                # Malformed value (e.g., "1.2.3" or ".") — terminate this block
                break
            rows.append({
                "element_type": entry.group("kind").upper(),
                "element_id": entry.group("id"),
                "metric": metric,
                "value": value,
            })
            cursor += 1

    return rows


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

#: Sentinel for the two-token `days hr:min` field. It consumes two columns
#: and emits no metric.
TIME_OF_MAX = "@time_of_max"

#: Column layout per summary table, after the element id. `None` skips a
#: single token (typically the element `Type`).
TABLE_SPECS = {
    "Node Depth Summary": ("NODE", [
        None, "node_avg_depth", "node_max_depth", "node_max_hgl",
        TIME_OF_MAX, "node_reported_max_depth",
    ]),
    "Node Inflow Summary": ("NODE", [
        None, "node_max_lateral_inflow", "node_max_total_inflow",
        TIME_OF_MAX, "node_lateral_inflow_volume", "node_total_inflow_volume",
        "node_flow_balance_error",
    ]),
    "Node Flooding Summary": ("NODE", [
        "node_hours_flooded", "node_max_flooding_rate",
        TIME_OF_MAX, "node_total_flood_volume", "node_max_ponded_depth",
    ]),
    "Link Flow Summary": ("LINK", [
        None, "link_max_flow", TIME_OF_MAX, "link_max_velocity",
        "link_max_full_flow", "link_max_full_depth",
    ]),
    "Outfall Loading Summary": ("NODE", [
        "outfall_pct_freq", "outfall_avg_flow", "outfall_max_flow",
        "outfall_total_volume",
    ]),
    "Storage Volume Summary": ("NODE", [
        None, "storage_avg_volume", "storage_avg_pct_full",
        "storage_evap_pct_loss", "storage_exfil_pct_loss",
        "storage_max_volume", "storage_max_pct_full",
        TIME_OF_MAX, "storage_max_outflow",
    ]),
}


def _table_row(tokens: list[str], columns: list[str | None]) -> dict[str, float]:
    """Map tokens after the element id onto metric names. Empty on mismatch."""
    values: dict[str, float] = {}
    cursor = 0
    for column in columns:
        if column is TIME_OF_MAX:
            cursor += 2  # `days` and `hr:min`
            continue
        if cursor >= len(tokens):
            return {}
        if column is not None:
            try:
                values[column] = float(tokens[cursor])
            except ValueError:
                return {}
        cursor += 1
    return values


def parse_tables(text: str) -> list[dict]:
    """Extract per-element metrics from every recognised summary table."""
    lines = text.splitlines()
    rows: list[dict] = []

    for index, line in enumerate(lines):
        spec = TABLE_SPECS.get(line.strip())
        if spec is None:
            continue
        element_type, columns = spec

        # Skip forward past the banner and the dashed header block; data
        # begins after the second run of dashes.
        cursor = index + 1
        dashes_seen = 0
        while cursor < len(lines) and dashes_seen < 2:
            if set(lines[cursor].strip()) == {"-"}:
                dashes_seen += 1
            cursor += 1
        if dashes_seen < 2:
            continue

        while cursor < len(lines):
            body = lines[cursor].strip()
            if not body or set(body) in ({"-"}, {"*"}):
                break
            tokens = body.split()
            values = _table_row(tokens[1:], columns)
            for metric, value in values.items():
                rows.append({
                    "element_type": element_type,
                    "element_id": tokens[0],
                    "metric": metric,
                    "value": value,
                })
            cursor += 1

    return rows
