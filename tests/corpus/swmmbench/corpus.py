"""Discovery and pairing of corpus models.

The corpus is third-party and irregular: three of its largest families are
mostly unpaired, two directories hold reports with no inputs at all, and one
directory holds three reports for a single input. Pairing is therefore strict
and explicit rather than best-effort.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from . import variants

# Report-only directories holding historical reference runs for models that
# live elsewhere in the tree. They contribute anchors, never models.
VERSION_ANCHOR_DIRS = {"v12": "ref_v12_path", "v13": "ref_v13_path"}

CHUNK = 1 << 20


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    family: str
    rel_path: str
    abs_path: Path
    inp_sha256: str
    size_bytes: int
    ref_path: str | None
    ref_v12_path: str | None
    ref_v13_path: str | None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _family(rel_path: str) -> str:
    head, sep, _ = rel_path.partition("/")
    return head if sep else "(root)"


def discover(root: Path) -> list[ModelRecord]:
    """Walk `root` and return one record per `.inp`, sorted by model_id."""
    root = Path(root)

    anchors: dict[str, dict[str, str]] = {}
    for directory, field in VERSION_ANCHOR_DIRS.items():
        for report in sorted((root / directory).glob("*.rpt")):
            anchors.setdefault(report.stem, {})[field] = _posix(root, report)

    records: list[ModelRecord] = []
    for inp in sorted(root.rglob("*.inp")):
        # A variant deck that survived a killed sweep is the harness's own
        # leftover, not a corpus model. Inventorying it would run the corpus
        # against a deck we wrote, and grow the model count every time a sweep
        # was interrupted.
        if inp.name.startswith(variants.TEMP_PREFIX):
            continue

        rel = _posix(root, inp)
        family = _family(rel)
        if family in VERSION_ANCHOR_DIRS:
            continue

        # Strict pairing: same directory, same full stem. `X.inp.rpt` has stem
        # `X.inp`, so it cannot collide with `X`.
        sibling = inp.with_suffix(".rpt")
        ref_path = _posix(root, sibling) if sibling.is_file() else None

        anchor = anchors.get(inp.stem, {})
        records.append(ModelRecord(
            model_id=rel[: -len(".inp")],
            family=family,
            rel_path=rel,
            abs_path=inp,
            inp_sha256=sha256_of(inp),
            size_bytes=inp.stat().st_size,
            ref_path=ref_path,
            ref_v12_path=anchor.get("ref_v12_path"),
            ref_v13_path=anchor.get("ref_v13_path"),
        ))
    return records
