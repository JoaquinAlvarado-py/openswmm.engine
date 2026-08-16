"""Emission of the A, B, C, D and E decks.

Every variant states every option under study explicitly. Leaving A to
inherit engine defaults would let a future change to a default silently
redefine the baseline and invalidate comparisons against earlier sweeps.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import schema

#: Pins shared by every variant, merged into each variant's own overrides
#: below rather than repeated five times.
#:
#: `VIRTUAL_JUNCTION_MOMENTUM BASIC` (FULL is retired and warns).
#:
#: `DPS_CELERITY`, `DPS_ALPHA` and `DPS_DECAY_TIME` are the engine's own
#: defaults (SimulationOptions.hpp:248-262) for the Dynamic Preissmann Slot.
#: They are inert under EXTRAN (A, B, C, D), so pinning them there is
#: harmless -- but deliberate: it stops a future change to those defaults
#: from silently drifting E's baseline out from under it. Note the keyword
#: is `DPS_CELERITY`, not `DPS_TARGET_CELERITY` -- the latter is only the
#: internal field name (OptionsHandler.cpp).
_COMMON: dict[str, str] = {
    "VIRTUAL_JUNCTION_MOMENTUM": "BASIC",
    "DPS_CELERITY": "25.0",
    "DPS_ALPHA": "3.0",
    "DPS_DECAY_TIME": "0.5",
}

#: Options under study, written verbatim into all five decks. Keyword
#: spellings are the `[OPTIONS]` keys accepted by OptionsHandler.cpp:
#:   ANDERSON_ACCEL       YES | NO
#:   NODE_CONTINUITY      EXPLICIT | SEMI_IMPLICIT
#:   SURCHARGE_METHOD     EXTRAN | SLOT | DYNAMIC_SLOT
#:
#: B isolates Anderson acceleration under EXPLICIT/EXTRAN -- its weakest
#: regime: DWSolver::computeAASkipFlags (DynamicWave.cpp:2934) disables
#: Anderson on every surcharged node whenever surcharge_method == EXTRAN
#: and node_continuity == EXPLICIT, which is exactly B's configuration.
#: C isolates semi-implicit (Crank-Nicolson) node continuity alone.
#: E isolates the Dynamic Preissmann Slot surcharge method alone. For B, C
#: and E, exactly one key differs from A, so B - A, C - A and E - A are each
#: attributable to a single cause. Do not fold two of these features into
#: one variant; doing so would make an observed shift unattributable
#: between their causes.
#:
#: D is the one deliberate exception to that one-feature-per-variant rule.
#: Under SEMI_IMPLICIT the unified Crank-Nicolson update is C1-smooth
#: through the free-surface/surcharge transition, so computeAASkipFlags no
#: longer disables Anderson at surcharged nodes -- exactly the regime
#: Anderson was designed for, and the regime B cannot exercise. D turns on
#: both ANDERSON_ACCEL and NODE_CONTINUITY relative to A so that regime can
#: be run at all; its value comes from D - C (incremental Anderson on top of
#: Crank-Nicolson, both keys held at C's value except ANDERSON_ACCEL), not
#: from D - A alone, which would be an unattributable joint effect.
#:
#: `NODE_CONTINUITY` is stated explicitly in every variant, including A and
#: B, even though EXPLICIT is the engine default (OptionsHandler.cpp:443):
#: leaving it implicit would let a future change to that default silently
#: redefine the baseline these variants are compared against. The same
#: reasoning is why `SURCHARGE_METHOD` is stated explicitly everywhere too.
OPTIONS: dict[str, dict[str, str]] = {
    schema.VARIANT_A: {
        **_COMMON,
        "ANDERSON_ACCEL": "NO",
        "NODE_CONTINUITY": "EXPLICIT",
        "SURCHARGE_METHOD": "EXTRAN",
    },
    schema.VARIANT_B: {
        **_COMMON,
        "ANDERSON_ACCEL": "YES",
        "NODE_CONTINUITY": "EXPLICIT",
        "SURCHARGE_METHOD": "EXTRAN",
    },
    schema.VARIANT_C: {
        **_COMMON,
        "ANDERSON_ACCEL": "NO",
        "NODE_CONTINUITY": "SEMI_IMPLICIT",
        "SURCHARGE_METHOD": "EXTRAN",
    },
    schema.VARIANT_D: {
        **_COMMON,
        "ANDERSON_ACCEL": "YES",
        "NODE_CONTINUITY": "SEMI_IMPLICIT",
        "SURCHARGE_METHOD": "EXTRAN",
    },
    schema.VARIANT_E: {
        **_COMMON,
        "ANDERSON_ACCEL": "NO",
        "NODE_CONTINUITY": "EXPLICIT",
        "SURCHARGE_METHOD": "DYNAMIC_SLOT",
    },
}

TEMP_PREFIX = ".swmmbench_"

_SECTION = re.compile(r"^\s*\[(?P<name>[A-Za-z0-9_ ]+)\]")


def _format(key: str, value: str) -> str:
    return f"{key:<20} {value}"


def apply_options(text: str, options: dict[str, str]) -> str:
    """Return `text` with `options` set in its `[OPTIONS]` section."""
    lines = text.splitlines()
    remaining = dict(options)

    start = end = None
    for index, line in enumerate(lines):
        match = _SECTION.match(line)
        if not match:
            continue
        if match.group("name").strip().upper() == "OPTIONS":
            start = index + 1
        elif start is not None and end is None:
            end = index

    if start is None:
        # No [OPTIONS] section: insert one before the first other section, or
        # append if the deck has no sections at all.
        body = [f"[OPTIONS]"] + [_format(k, v) for k, v in remaining.items()] + [""]
        for index, line in enumerate(lines):
            if _SECTION.match(line):
                return "\n".join(lines[:index] + body + lines[index:]) + "\n"
        return "\n".join(lines + [""] + body) + "\n"

    if end is None:
        end = len(lines)

    rewritten: list[str] = []
    for line in lines[start:end]:
        key = line.split(None, 1)[0].upper() if line.split() else ""
        if key in remaining:
            rewritten.append(_format(key, remaining.pop(key)))
        else:
            rewritten.append(line)

    # Anything not already present is appended at the end of the section, but
    # before its trailing blank lines so the file stays readable.
    tail = len(rewritten)
    while tail > 0 and not rewritten[tail - 1].strip():
        tail -= 1
    additions = [_format(k, v) for k, v in remaining.items()]

    out = lines[:start] + rewritten[:tail] + additions + rewritten[tail:] + lines[end:]
    return "\n".join(out) + "\n"


@contextmanager
def temp_deck(inp_path: Path, variant: str) -> Iterator[Path]:
    """Write the variant deck beside its original; always remove it.

    Corpus decks resolve `DataFiles/*.dat` and loose `.txt` inputs relative to
    their own directory, so the variant cannot be relocated.
    """
    inp_path = Path(inp_path)
    deck = inp_path.with_name(f"{TEMP_PREFIX}{variant}_{inp_path.name}")
    original = inp_path.read_text(encoding="latin-1")
    deck.write_text(apply_options(original, OPTIONS[variant]), encoding="latin-1")
    try:
        yield deck
    finally:
        deck.unlink(missing_ok=True)
