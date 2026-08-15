"""Emission of the A, B and C decks.

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

#: Options under study, written verbatim into all three decks.
#:
#: B isolates Anderson acceleration alone: `ANDERSON_ACCEL` is the ONLY key
#: whose value differs between A and B. C isolates semi-implicit
#: (Crank-Nicolson) node continuity alone: `NODE_CONTINUITY` is the ONLY key
#: whose value differs between A and C. Every other option is stated
#: identically across all three so that a single variable moves per pairing
#: and the B - A and C - A deltas are each attributable to exactly one
#: feature. This is deliberate -- do not fold Anderson and Crank-Nicolson
#: into a single variant; doing so would make every observed shift
#: unattributable between the two causes.
#:
#: `NODE_CONTINUITY` is stated explicitly in every variant, including A and
#: B, even though EXPLICIT is the engine default (OptionsHandler.cpp:443):
#: leaving it implicit would let a future change to that default silently
#: redefine the baseline these variants are compared against.
#:
#: The other continuity knobs the engine exposes, for whoever adds the next
#: variant. Keyword spellings are the `[OPTIONS]` keys accepted by
#: OptionsHandler.cpp:
#:   SURCHARGE_METHOD             EXTRAN | SLOT | DYNAMIC_SLOT
#:   VIRTUAL_JUNCTION_MOMENTUM    BASIC (FULL is retired and warns)
#:   DPS_CELERITY                 the dps_target_celerity knob
#:   DPS_ALPHA
#:   DPS_DECAY_TIME
OPTIONS: dict[str, dict[str, str]] = {
    schema.VARIANT_A: {
        "ANDERSON_ACCEL": "NO",
        "NODE_CONTINUITY": "EXPLICIT",
        "SURCHARGE_METHOD": "EXTRAN",
        "VIRTUAL_JUNCTION_MOMENTUM": "BASIC",
    },
    schema.VARIANT_B: {
        "ANDERSON_ACCEL": "YES",
        "NODE_CONTINUITY": "EXPLICIT",
        "SURCHARGE_METHOD": "EXTRAN",
        "VIRTUAL_JUNCTION_MOMENTUM": "BASIC",
    },
    schema.VARIANT_C: {
        "ANDERSON_ACCEL": "NO",
        "NODE_CONTINUITY": "SEMI_IMPLICIT",
        "SURCHARGE_METHOD": "EXTRAN",
        "VIRTUAL_JUNCTION_MOMENTUM": "BASIC",
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
