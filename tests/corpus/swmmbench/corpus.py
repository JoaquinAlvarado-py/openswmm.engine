"""Discovery and pairing of corpus models.

The corpus is third-party and irregular: three of its largest families are
mostly unpaired, two directories hold reports with no inputs at all, and one
directory holds three reports for a single input. Pairing is therefore strict
and explicit rather than best-effort.
"""

from __future__ import annotations

import hashlib
import subprocess
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
    #: The corpus commit this record was taken at -- the identity of every
    #: file the deck depends on, not just the deck. See `commit_sha`.
    corpus_commit: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The corpus dependency identity
# ---------------------------------------------------------------------------

#: Prefix marking a real, cryptographic pin of the corpus state.
GIT_COMMIT_PREFIX = "git:"

#: Recorded when the corpus root is not a git repository, or git is not
#: available. Deliberately a legible sentinel rather than a hash of anything:
#: it must be impossible to mistake for a pin when read back out of `runs`,
#: and it must never compare equal to a commit -- so a store built against an
#: unpinned corpus can neither masquerade as reproducible nor be silently
#: resumed as if the corpus had been pinned all along.
UNPINNED_CORPUS = "unpinned:not-a-git-repo"

GIT_TIMEOUT_S = 30.0

#: Roots already warned about, so a sweep says it once instead of once per
#: stage. Keyed by root rather than a bare flag: two corpora in one process
#: (the test suite does exactly this) each deserve their own warning.
_warned_roots: set[str] = set()


def commit_sha(root: Path) -> str:
    """The corpus's git commit -- the identity of everything a deck depends on.

    `inp_sha256` hashes the deck ALONE, and corpus decks reference external
    data by relative path: `DataFiles/*.dat`, loose `.txt` series, interface
    files. If `Example.inp` is untouched but `DataFiles/rainfall.dat` changes,
    a deck-only hash says "already done" and the sweep republishes stale
    numbers as current.

    The dependency identity is therefore the corpus's git commit, NOT a parse
    of the deck's file references. Enumerating those references means
    covering `[RAINGAGES] FILE`, `[TIMESERIES] FILE`, `[TEMPERATURE] FILE`,
    the `[FILES]` interface section, `[LID_USAGE]` report files and more --
    and any section type missed makes the hash LIE, reintroducing exactly
    this bug with more code to be wrong in. The corpus is a git repository,
    so its commit covers every referenced file exactly, for free.

    It over-invalidates: moving the corpus to a new commit re-runs every
    model, including those whose files did not change. That is rare (the
    corpus is a third-party pin, not a working tree) and it is the safe
    direction -- a needless re-run costs hours, a missed one publishes wrong
    numbers. It also does not cover UNCOMMITTED edits inside the corpus; the
    harness treats the corpus as read-only by contract and `swmmbench check`
    enforces that it leaves nothing behind.

    Degrades rather than fails: a corpus that is not a git repository, or a
    machine with no git, records `UNPINNED_CORPUS` and warns once. Refusing
    to sweep would be worse than sweeping with the dependency openly marked
    as unpinned.
    """
    root = Path(root)
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
            errors="replace",
        )
        sha = (proc.stdout or "").strip()
        if proc.returncode == 0 and sha:
            return f"{GIT_COMMIT_PREFIX}{sha}"
    except (OSError, subprocess.SubprocessError):
        pass

    key = str(root)
    if key not in _warned_roots:
        _warned_roots.add(key)
        print(f"WARNING: {root} is not a git repository (or git is "
              f"unavailable); the corpus dependency identity is recorded as "
              f"{UNPINNED_CORPUS!r}. Results cannot be attributed to a "
              f"corpus state, and a change to a deck's external data files "
              f"will not invalidate a resumed sweep.")
    return UNPINNED_CORPUS


def _posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _family(rel_path: str) -> str:
    head, sep, _ = rel_path.partition("/")
    return head if sep else "(root)"


def discover(root: Path) -> list[ModelRecord]:
    """Walk `root` and return one record per `.inp`, sorted by model_id.

    The corpus commit is read ONCE here, at inventory time, and stamped on
    every record: the model list and the dependency identity then describe
    the same corpus state by construction, instead of being read at two
    moments that a `git checkout` could fall between.
    """
    root = Path(root)
    corpus_commit = commit_sha(root)

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
            corpus_commit=corpus_commit,
        ))
    return records
