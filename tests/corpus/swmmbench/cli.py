"""Stage dispatch for the corpus benchmark harness."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import platform
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import corpus, rptparse, runner, schema, store, variants

#: The resume key is the case identity and nothing else. `case_id` already
#: folds in the model, the variant, the engine BUILD (a content hash of the
#: executable, not the version string it prints) and the corpus commit, so
#: listing those columns again here could only let the two definitions drift.
#: The previous key -- model/variant/engine_version/inp_sha256 -- skipped work
#: it should have re-run twice over: two executables both printing
#: `OpenSWMM 6.0` were one key, and a deck whose `DataFiles/*.dat` changed was
#: unchanged. A store written before `case_id` existed has no such column, so
#: `store.completed_keys` returns nothing and every case re-runs; that is the
#: safe direction, since those rows' engine build cannot be established after
#: the fact.
RESUME_KEY = ["case_id"]

#: How many completed units of work are persisted between Parquet flushes,
#: in BOTH long stages: one unit is a simulation in `run`, a model in `diff`.
#:
#: A full sweep is ~8200 simulations over several hours, and every run row
#: plus its (potentially hundreds of) element rows would otherwise be held in
#: memory until the last one finished -- so a machine failure near the end
#: lost the whole sweep, not its tail. 50 is chosen from the 25-100 band: low
#: enough that a crash costs at most a minute or two of re-running, high
#: enough that the per-write overhead (a Parquet file per family per flush,
#: each with its own footer and dictionary pages) stays a rounding error
#: against the simulations themselves. Below ~25 the store fragments into
#: thousands of tiny files and the read side pays for it; above ~100 the
#: memory and the loss window grow with no compensating gain.
FLUSH_BATCH_SIZE = 50

#: Parsed scalars that are NOT metrics and must stay out of the long-format
#: `scalars` table. Its `value` column is numeric by construction; melting a
#: string-valued key into it (a version banner, a timestamp, a routing-model
#: or option echo) mixes types in one Arrow column and the write fails
#: outright. They all remain available, one column each, in `runs`.
NON_METRIC_SCALARS = frozenset({
    "reported_version", "start_date", "end_date", "iteration_metric_kind",
    "reported_routing_model", "reported_surcharge_method",
    "reported_node_continuity", "reported_anderson_accel",
})

#: The metric columns `scalars` is melted from -- every parsed scalar that is
#: not in the denylist above.
METRIC_SCALARS = tuple(key for key in rptparse.SCALAR_KEYS
                       if key not in NON_METRIC_SCALARS)

#: Every column a `runs` row can carry, in write order.
#:
#: Load-bearing now that `run` writes in batches rather than once: a plain
#: `pd.DataFrame(rows)` takes the union of the keys the rows in THAT batch
#: happen to have, and the batches are not alike -- a crash row has no
#: `out_path`, a reference row has no `wall_ms`, `exit_code` or `host`. Two
#: fragments with different columns no longer read back as one dataset, so
#: the column set is stated once and every batch is conformed to it.
RUN_COLUMNS = (
    "case_id", "model_id", "family", "variant",
    "engine_version", "engine_build_id", "engine_build_info", "corpus_commit",
    "inp_sha256", "options_applied", "host", "run_started_at", "timeout_s",
    "status", "exit_code", "signal", "wall_ms", "peak_rss_mb",
    "stderr_tail", "out_path",
    *rptparse.SCALAR_KEYS,
)

#: `runs` columns written as float64, and as datetimes, respectively.
#: Everything else is written as a nullable string. Stated rather than left
#: to inference for the same reason as `RUN_COLUMNS`: a batch in which a
#: column happens to be entirely null infers Arrow's `null` type and no
#: longer matches its neighbours.
RUN_NUMERIC_COLUMNS = frozenset({
    "timeout_s", "exit_code", "signal", "wall_ms", "peak_rss_mb",
    *METRIC_SCALARS,
})
RUN_DATETIME_COLUMNS = frozenset({"start_date", "end_date"})

#: Every column an `elements` row carries, for the same reason.
ELEMENT_COLUMNS = (
    "case_id", "model_id", "family", "variant", "engine_version",
    "element_type", "element_id", "metric", "value",
)


def stage_inventory(corpus_root: Path, out_dir: Path) -> pd.DataFrame:
    """Re-derive the model list from the corpus, replacing any earlier one.

    `models` is a pure function of the corpus tree, so a second `inventory`
    must replace it rather than append: `store.write_table` appends by design
    (fresh UUID basename per call), and the README makes `inventory` step 1 of
    every sweep, so an appending write would silently double the corpus -- and
    every downstream count computed over it -- with no error.
    """
    records = corpus.discover(Path(corpus_root))
    frame = pd.DataFrame([
        {k: v for k, v in dataclasses.asdict(r).items() if k != "abs_path"}
        for r in records
    ])
    _replace_table(frame, out_dir, "models", partition_by=["family"])
    return frame


#: Prefix on an `engine_build_id` derived from the CONTENT of the executable.
#: The load-bearing case: two builds differ here iff their bytes differ.
ENGINE_BUILD_HASHED_PREFIX = "sha256:"

#: Prefix on an `engine_build_id` that could NOT be derived from file
#: content, because the argv's first element does not name a readable file --
#: a command the OS resolves some other way, or a stand-in. Clearly marked
#: because it is NOT a cryptographic build identity: two different engines
#: reachable under one name collide under it. Marked rather than fatal: a
#: sweep must not crash because `--version` was pointed at something odd.
ENGINE_BUILD_UNHASHED_PREFIX = "argv:"

VERSION_TIMEOUT_S = 30

#: Lines of `--version` output worth keeping beyond the bare banner: a git
#: commit, a branch, a build type. Purely a human-readable extra -- captured
#: when the engine happens to print it, never depended on. `engine_build_id`
#: must not depend on the engine printing ANYTHING.
_BUILD_INFO_LINE = re.compile(r"\b(commit|revision|branch|build|hash)\b",
                              re.IGNORECASE)


def _version_output(engine: list[str]) -> str:
    """Raw `--version` output, or empty when the engine cannot be asked."""
    import subprocess
    try:
        proc = subprocess.run(
            list(engine) + ["--version"],
            capture_output=True, text=True, timeout=VERSION_TIMEOUT_S,
            errors="replace",
        )
        return (proc.stdout or proc.stderr) or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _engine_version(engine: list[str]) -> str:
    """The engine's human-readable LABEL. Not an identity.

    Kept because an operator reading `runs` needs something legible, but it
    is exactly what must not be trusted as identity: two executables with
    completely different Anderson, continuity or slot implementations can
    both print `OpenSWMM 6.0`. `engine_build_id` is the identity.
    """
    lines = _version_output(engine).strip().splitlines()
    return lines[0][:200] if lines else " ".join(engine)[:200]


def _build_info(output: str) -> str | None:
    """Any git commit / branch / build-type lines the version output carried."""
    hits = [line.strip() for line in output.splitlines()
            if line.strip() and _BUILD_INFO_LINE.search(line)]
    return "; ".join(hits)[:200] or None


def _argv_fingerprint(token: str) -> str:
    """The content hash of the file `token` names, or the token itself.

    Resolution follows the OS's own order -- an explicit path first, then
    PATH -- so a bare `openswmm` fingerprints identically to the absolute
    path of the same file.
    """
    path = Path(token)
    try:
        if not path.is_file():
            resolved = shutil.which(token)
            if resolved:
                path = Path(resolved)
        if path.is_file():
            return ENGINE_BUILD_HASHED_PREFIX + corpus.sha256_of(path)
    except OSError:
        pass
    return ENGINE_BUILD_UNHASHED_PREFIX + token


def _engine_build_id(engine: list[str]) -> str:
    """A cryptographic identity for the engine actually being invoked.

    The load-bearing part is the sha256 of the EXECUTABLE FILE -- argv's
    first element -- and it never depends on the engine printing anything.
    The digest is taken over the whole argv, each element replaced by its
    file hash where it names a readable file and left literal where it does
    not, because the executable alone is not the whole engine whenever the
    real program is an argument to a launcher: `[python, engine.py]`,
    `[wine, openswmm.exe]`, `[mpirun, -n, 4, openswmm]`. For the ordinary
    `[/path/to/openswmm]` argv this reduces to exactly the hash of that one
    file, wrapped once more.

    When the first element is NOT a readable file -- a shell builtin, a
    command resolved by something other than PATH, a deleted binary -- the
    result is still deterministic but is prefixed `argv:` instead of
    `sha256:`, so a reader can never mistake it for a real build pin. It
    degrades; it does not raise. A sweep of 8200 simulations must not die
    because the identity could not be computed.
    """
    argv = list(engine) or [""]
    fingerprints = [_argv_fingerprint(token) for token in argv]
    digest = hashlib.sha256("\n".join(fingerprints).encode("utf-8")).hexdigest()
    prefix = (ENGINE_BUILD_HASHED_PREFIX
              if fingerprints[0].startswith(ENGINE_BUILD_HASHED_PREFIX)
              else ENGINE_BUILD_UNHASHED_PREFIX)
    return prefix + digest


def _engine_identity(engine: list[str]) -> dict:
    """`engine_build_id` (identity), `engine_version` and `engine_build_info`.

    One `--version` invocation serves both derived labels. Never raises.
    """
    output = _version_output(engine)
    lines = output.strip().splitlines()
    return {
        "engine_build_id": _engine_build_id(engine),
        "engine_version": (lines[0][:200] if lines
                           else " ".join(engine)[:200]),
        "engine_build_info": _build_info(output),
    }


def _run_one(job: dict) -> dict:
    """Execute one (model, variant) pair. Never raises."""
    record = job["record"]
    variant = job["variant"]
    work = Path(job["work_dir"]) / record["model_id"].replace("/", "__") / variant
    row = {
        # The case identity is derived once, in `stage_run`, and travels in
        # the job: the resume key that decided to dispatch this job and the
        # `case_id` written on its rows are then the same value, not two
        # computations that could disagree.
        "case_id": job["case_id"],
        "model_id": record["model_id"],
        "family": record["family"],
        "variant": variant,
        "engine_version": job["engine_version"],
        "engine_build_id": job["engine_build_id"],
        "engine_build_info": job["engine_build_info"],
        "corpus_commit": job["corpus_commit"],
        "inp_sha256": record["inp_sha256"],
        "options_applied": "; ".join(
            f"{k}={v}" for k, v in variants.OPTIONS[variant].items()
        ),
        "host": platform.node(),
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "timeout_s": job["timeout_s"],
    }
    scalars: dict = {key: None for key in rptparse.SCALAR_KEYS}
    elements: list[dict] = []

    inp_path = Path(job["corpus_root"]) / record["rel_path"]
    try:
        with variants.temp_deck(inp_path, variant) as deck:
            result = runner.run_once(job["engine"], deck, work, job["timeout_s"])
    except OSError as error:
        row.update(status=schema.Status.CRASH, exit_code=None, signal=None,
                   wall_ms=0.0, peak_rss_mb=None, stderr_tail=str(error))
        return {"run": {**row, **scalars}, "elements": elements}

    row.update(
        status=result.status,
        exit_code=result.exit_code,
        signal=result.signal,
        wall_ms=result.wall_ms,
        peak_rss_mb=result.peak_rss_mb,
        stderr_tail=result.stderr_tail,
        # Recorded rather than reconstructed by the diff stage: the output
        # filename is runner.py's convention and must not be duplicated.
        out_path=str(result.out_path) if result.out_path else None,
    )

    if result.rpt_path is not None:
        try:
            text = rptparse.read(result.rpt_path)
            scalars = rptparse.parse_scalars(text)
            elements = rptparse.parse_rankings(text) + rptparse.parse_tables(text)
        except (OSError, ValueError):
            row["status"] = schema.Status.PARSE_ERROR

    # `case_id` travels with every element row for the same reason it is the
    # `runs` resume key: two engine builds -- or two corpus commits --
    # legitimately coexist in one store, and without the column their rows
    # cannot be told apart even in principle, so `build_element_deltas`'s
    # `pivot_table(..., aggfunc="first")` would silently pick one of them.
    # `engine_version` stays alongside it as the legible label.
    for element in elements:
        element.update(case_id=job["case_id"], model_id=record["model_id"],
                       family=record["family"], variant=variant,
                       engine_version=job["engine_version"])

    return {"run": {**row, **scalars}, "elements": elements}


#: Stable identity for reference rows. It must NOT be the version string read
#: out of the report: the resume key is compared against this value before the
#: report has been parsed, so a parsed-version key would never match and every
#: resumed sweep would duplicate its reference rows.
REF_ENGINE_VERSION = "corpus-reference"

#: The `engine_build_id` a reference row carries. No build of ours produced
#: it, so there is no executable to hash; the same sentinel stands in, and
#: `executed_engine_versions` excludes REF rows by variant so it can never be
#: counted as a build. A REF case therefore depends only on the model and the
#: corpus commit -- which is right: re-inventorying at a new commit must
#: re-read the anchors, since the corpus `.rpt` may itself have changed.
REF_ENGINE_BUILD_ID = REF_ENGINE_VERSION


def _reference_rows(
    corpus_root: Path,
    record: dict,
    case_id: str,
    corpus_commit: str,
) -> tuple[dict, list[dict]]:
    """Parse the anchor report. It is never re-run, only read.

    A model with no anchor still gets a row, carrying status `no_ref`. An
    absent row would be indistinguishable from a model the sweep skipped.
    """
    base = {
        "case_id": case_id,
        "model_id": record["model_id"],
        "family": record["family"],
        "variant": schema.VARIANT_REF,
        "engine_version": REF_ENGINE_VERSION,
        "engine_build_id": REF_ENGINE_BUILD_ID,
        "engine_build_info": None,
        "corpus_commit": corpus_commit,
        "inp_sha256": record["inp_sha256"],
    }
    empty = {key: None for key in rptparse.SCALAR_KEYS}

    if not record.get("ref_path"):
        return {**base, "status": schema.Status.NO_REF, **empty}, []

    path = Path(corpus_root) / record["ref_path"]
    try:
        text = rptparse.read(path)
    except OSError:
        return {**base, "status": schema.Status.PARSE_ERROR, **empty}, []

    try:
        scalars = rptparse.parse_scalars(text)
        elements = rptparse.parse_rankings(text) + rptparse.parse_tables(text)
    except ValueError:
        return {**base, "status": schema.Status.PARSE_ERROR, **empty}, []

    for element in elements:
        element.update(case_id=case_id, model_id=record["model_id"],
                       family=record["family"], variant=schema.VARIANT_REF,
                       engine_version=REF_ENGINE_VERSION)
    return {**base, "status": schema.Status.OK, **scalars}, elements


def _run_frame(rows: list[dict]) -> pd.DataFrame:
    """One batch of `runs` rows, conformed to `RUN_COLUMNS` and its dtypes.

    Every batch is written with the same columns and the same Arrow types, so
    the fragments a batched sweep leaves behind still read back as one
    dataset. Without this a batch of only reference rows (no `wall_ms`) or
    only crash rows (no `out_path`), or one in which some scalar happened to
    be null throughout, would write an incompatible fragment.
    """
    frame = pd.DataFrame(rows, columns=list(RUN_COLUMNS))
    for column in frame.columns:
        if column in RUN_NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce").astype("float64")
        elif column in RUN_DATETIME_COLUMNS:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        else:
            frame[column] = frame[column].astype("string")
    return frame


def _scalar_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """The long-format `scalars` view of one already-conformed `runs` batch."""
    frame = runs.melt(
        # `case_id` is THE id: `engine_version` alone could not tell one
        # build's scalar row from another's, and nothing at all could tell a
        # pre-change model's row from its post-change replacement.
        # `engine_version` stays as the legible label beside it.
        id_vars=["case_id", "model_id", "family", "variant", "engine_version"],
        value_vars=list(METRIC_SCALARS),
        var_name="metric", value_name="value",
    )
    frame["metric"] = frame["metric"].astype("string")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype("float64")
    return frame


def _element_frame(rows: list[dict]) -> pd.DataFrame:
    """One batch of `elements` rows, conformed to `ELEMENT_COLUMNS`."""
    frame = pd.DataFrame(rows, columns=list(ELEMENT_COLUMNS))
    for column in frame.columns:
        if column == "value":
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce").astype("float64")
        else:
            frame[column] = frame[column].astype("string")
    return frame


def _flush_run_batch(
    out_dir: Path,
    run_rows: list[dict],
    element_rows: list[dict],
) -> None:
    """Persist one batch of completed simulations.

    `scalars` is derived from the very frame written to `runs`, not from the
    raw rows, so the two can never disagree about a value or an id.

    `runs` is written LAST, for the same asymmetry `_flush_diff_batch`
    observes: `runs` is the resume authority, so a crash between the writes
    must not leave a case recorded as done whose `scalars` and `elements`
    never landed -- irrecoverable without deleting run rows by hand. Written
    in this order the crash instead leaves derived rows for a case `runs`
    does not claim, which the resumed sweep simply recomputes; a case is
    deterministic in its id, so the re-appended rows are duplicates of equal
    values rather than a contradiction.
    """
    if element_rows:
        store.write_table(_element_frame(element_rows), out_dir, "elements",
                          partition_by=["family"])
    if run_rows:
        frame = _run_frame(run_rows)
        store.write_table(_scalar_frame(frame), out_dir, "scalars",
                          partition_by=["family"])
        store.write_table(frame, out_dir, "runs", partition_by=["family"])


def _crash_result(job: dict, error: BaseException) -> dict:
    """A `runs` row for a job whose WORKER died, not whose model failed.

    `_run_one` never raises, so this is reached only when the pool itself
    loses the process (`BrokenProcessPool`, an OOM kill). The taxonomy still
    has to cover it: an absent row would be indistinguishable from work the
    resume key skipped, and would be silently retried forever.
    """
    record = job["record"]
    return {
        "run": {
            "case_id": job["case_id"],
            "model_id": record["model_id"],
            "family": record["family"],
            "variant": job["variant"],
            "engine_version": job["engine_version"],
            "engine_build_id": job["engine_build_id"],
            "engine_build_info": job["engine_build_info"],
            "corpus_commit": job["corpus_commit"],
            "inp_sha256": record["inp_sha256"],
            "host": platform.node(),
            "timeout_s": job["timeout_s"],
            "status": schema.Status.CRASH,
            "wall_ms": 0.0,
            "stderr_tail": f"worker process lost: {error}",
            **{key: None for key in rptparse.SCALAR_KEYS},
        },
        "elements": [],
    }


def stage_run(
    corpus_root: Path,
    out_dir: Path,
    engine: list[str],
    timeout_s: float,
    jobs: int,
    limit: int | None,
) -> int:
    """Run every outstanding (model, variant) pair. Returns a process exit code.

    Rows are flushed to Parquet every `FLUSH_BATCH_SIZE` completed
    simulations rather than once at the end. A full sweep is ~8200
    simulations over several hours; accumulating every run row and every
    element row until the last one finished meant a machine failure near the
    end lost the whole sweep instead of its tail, and held the entire result
    set in memory to boot. Completions are consumed as they arrive
    (`as_completed`), so a slow model delays only its own batch. This mirrors
    `stage_diff`, which already batches by model against the same constant.

    Non-zero means the corpus was left dirty. That is a hard failure, not a
    warning: the harness's input is read-only by contract, and a surviving
    temporary deck becomes a corpus model on the next `inventory`, poisoning
    every subsequent sweep.
    """
    corpus_root = Path(corpus_root)
    models = store.read_table(out_dir, "models")
    if models.empty:
        models = stage_inventory(corpus_root, out_dir)
    if limit:
        models = models.head(limit)

    identity = _engine_identity(engine)
    # Read at inventory time and carried on `models`, so the model list and
    # the dependency identity describe one corpus state rather than two
    # readings a `git checkout` could fall between. Falls back to reading the
    # root for a store written before the column existed.
    corpus_commit = _corpus_commit(models, corpus_root)
    done = store.completed_keys(out_dir, "runs", RESUME_KEY)

    work_dir = Path(out_dir) / "work"
    payload = []
    for record in models.to_dict("records"):
        for variant in schema.VARIANTS:
            case_id = schema.case_id(record["model_id"], variant,
                                     identity["engine_build_id"], corpus_commit)
            if (case_id,) in done:
                continue
            payload.append({
                "record": record, "variant": variant, "engine": engine,
                "case_id": case_id, "corpus_commit": corpus_commit,
                "timeout_s": timeout_s, **identity,
                "corpus_root": str(corpus_root), "work_dir": str(work_dir),
            })

    run_rows: list[dict] = []
    element_rows: list[dict] = []
    batch_count = 0

    def absorb(result: dict) -> None:
        """Queue one completed unit of work, flushing at a batch boundary."""
        nonlocal run_rows, element_rows, batch_count
        run_rows.append(result["run"])
        element_rows.extend(result["elements"])
        batch_count += 1
        if batch_count >= FLUSH_BATCH_SIZE:
            _flush_run_batch(out_dir, run_rows, element_rows)
            run_rows, element_rows, batch_count = [], [], 0

    if jobs > 1 and payload:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_run_one, job): job for job in payload}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001  isolate one model
                    result = _crash_result(futures[future], error)
                absorb(result)
    else:
        for job in payload:
            absorb(_run_one(job))

    for record in models.to_dict("records"):
        case_id = schema.case_id(record["model_id"], schema.VARIANT_REF,
                                 REF_ENGINE_BUILD_ID, corpus_commit)
        if (case_id,) in done:
            continue
        ref_row, ref_elements = _reference_rows(corpus_root, record, case_id,
                                                corpus_commit)
        absorb({"run": ref_row, "elements": ref_elements})

    _flush_run_batch(out_dir, run_rows, element_rows)

    clean, leftovers = corpus_is_clean(corpus_root)
    if not clean:
        print(f"ERROR: {len(leftovers)} temporary decks left in the corpus:")
        for item in leftovers[:10]:
            print(f"  {item}")
        return 1
    return 0


def _corpus_commit(models: pd.DataFrame, corpus_root: Path) -> str:
    """The corpus commit `models` was inventoried at, or a fresh reading.

    Preferring the recorded value keeps `models` and `runs` describing the
    same corpus state -- the README makes `inventory` step 1 of every sweep,
    so it is also the current one. The fallback covers a store written before
    the column existed, where re-reading the root is strictly better than
    pretending the dependency was never pinned.
    """
    if "corpus_commit" in models.columns:
        values = models["corpus_commit"].dropna().astype(str).unique()
        if len(values) == 1:
            return str(values[0])
    return corpus.commit_sha(corpus_root)


def corpus_is_clean(corpus_root: Path) -> tuple[bool, list[str]]:
    """True when no harness temporary deck survives in the corpus."""
    leftovers = [
        str(path)
        for path in Path(corpus_root).rglob(f"{variants.TEMP_PREFIX}*")
    ]
    return not leftovers, leftovers


#: `ts_diff` columns naming the two cases a comparison row was computed from.
#: A comparison has two sources, not one, so a single `case_id` would have to
#: pick a side; both are carried, matching the label's own `X_minus_Y` order.
TS_DIFF_CASE_COLUMNS = ("case_id_left", "case_id_right")

#: `ts_diff`'s coverage evidence, by the dtype each is written as. Every one
#: of them is nullable -- an empty overlap has no `coverage_fraction`, and two
#: empty series have no grids to compare -- and a whole batch can legitimately
#: be null in any of them. Stated rather than inferred for exactly the reason
#: `RUN_COLUMNS` is: an all-null batch infers Arrow's `null` type and its
#: fragment stops matching the ones around it, so the dataset no longer reads
#: back as one table.
#:
#: `Int64` and `boolean` (not `int64`/`bool`) because both must hold `<NA>`;
#: `time_grid_match` in particular must be able to say "unknown" rather than
#: being coerced to `False`, which the summary would then count as an anomaly.
TS_DIFF_COVERAGE_DTYPES = {
    "n_periods_left": "Int64",
    "n_periods_right": "Int64",
    "n_common": "Int64",
    "coverage_fraction": "float64",
    "start_time_match": "boolean",
    "end_time_match": "boolean",
    "time_grid_match": "boolean",
}


def _ts_diff_frame(rows: list[dict]) -> pd.DataFrame:
    """One batch of `ts_diff` rows, conformed to its nullable column dtypes.

    Conformed for the same reason `_run_frame` conforms `runs`: a batch in
    which neither side had a `case_id` (a store written before the column
    existed), or in which every row's coverage happened to be unknown, would
    otherwise infer Arrow's `null` type and stop matching the fragments
    around it.
    """
    frame = pd.DataFrame(rows)
    for column in TS_DIFF_CASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
        frame[column] = frame[column].astype("string")
    for column, dtype in TS_DIFF_COVERAGE_DTYPES.items():
        if column not in frame.columns:
            frame[column] = None
        frame[column] = frame[column].astype(dtype)
    return frame


def _flush_diff_batch(out_dir: Path, rows: list[dict], pending: list[Path]) -> None:
    """Persist accumulated diff rows, then delete the `.out` files behind them.

    Order matters: writing before deleting means a crash between the two
    calls leaves `.out` files undeleted (harmless, re-runnable) rather than
    deleting source data whose derived rows were never written (irrecoverable).
    """
    if rows:
        store.write_table(_ts_diff_frame(rows), out_dir, "ts_diff",
                          partition_by=["family"])
    for path in pending:
        path.unlink(missing_ok=True)


#: Time-series comparisons, as `(left, right, label)`: each produces rows
#: labelled `label` holding the divergence of `left`'s `.out` from `right`'s.
#:
#: `D_minus_C` is a FIRST-CLASS comparison here, not something a reader can
#: reconstruct downstream. A `ts_diff` row holds `max_abs`, `max_rel`, `rmse`
#: and `first_div_period` -- non-linear reductions over the pointwise
#: difference of two series -- and `max_abs(D, C)` is NOT
#: `max_abs(D, A) - max_abs(C, A)`; the triangle inequality gives a bound,
#: never a value. So pairing `D_minus_A` with `C_minus_A` cannot yield
#: D - C at the time-series level, and without this entry the only reason
#: variant D exists would be unmeasurable from anything the harness writes.
#: (At the *scalar* level `D - C` IS derivable, because `value_d - value_c`
#: is linear; that is `report.build_deltas`'s business, not this stage's.)
#:
#: B isolates Anderson acceleration (EXPLICIT/EXTRAN); C isolates
#: semi-implicit (Crank-Nicolson) node continuity; D - A is the joint
#: Anderson + Crank-Nicolson effect and D - C the attributable incremental
#: Anderson one; E isolates the Dynamic Preissmann Slot surcharge method.
DIFF_COMPARISONS = (
    (schema.VARIANT_B, schema.VARIANT_A, "B_minus_A"),
    (schema.VARIANT_C, schema.VARIANT_A, "C_minus_A"),
    (schema.VARIANT_D, schema.VARIANT_A, "D_minus_A"),
    (schema.VARIANT_E, schema.VARIANT_A, "E_minus_A"),
    (schema.VARIANT_D, schema.VARIANT_C, "D_minus_C"),
)


def _comparison_is_resolved(
    comparison: tuple[str, str, str],
    variant: str,
    succeeded: set[str],
    usable: dict[str, str],
    recorded: set[str],
) -> bool:
    """True when `comparison` can no longer need `variant`'s `.out` on disk.

    Resolved means either it succeeded on this pass (its rows are queued for
    the flush that precedes any deletion), or it is *terminally unavailable*:
    the other side is recorded in `runs` for this build with a status that is
    not `ok`, or `ok` with no `out_path`. The resume key guarantees
    `stage_run` will not retry such a run, so no future `diff` can ever
    produce this comparison for this model under this build -- retaining a
    file for it would be waste with no recovery path.

    Deliberately NOT resolved: the other side has no run row at all (merely
    pending -- someone may run A/B/C, diff, then add D/E and diff again), or
    it ran `ok` but its diff raised on this pass (retryable).
    """
    left, right, label = comparison
    if label in succeeded:
        return True
    other = right if variant == left else left
    return other not in usable and other in recorded


#: Console messages `stage_diff` prints when it refuses a mixed-build store.
#: English-only, like every other `stage_run`/`stage_diff` message; only
#: `stage_report`'s output is translated (see `STAGE_REPORT_STRINGS`).
DIFF_MIXED_ENGINE_ERROR = (
    "ERROR: `runs` mixes results from more than one engine build; diff "
    "refuses to compare `.out` files it cannot attribute to one build."
)
DIFF_MIXED_ENGINE_HINT = (
    "Re-run `diff` against a store holding a single engine build."
)


def stage_diff(out_dir: Path, abs_tol: float = 1e-9) -> int:
    """Diff the retained `.out` pairs, then discard the binaries safely.

    Returns a process exit code; non-zero means nothing was written.

    `abs_tol` is an absolute tolerance on the value difference, not a relative
    one: it is the threshold `outdiff.diff_series` compares `|b - a|` against.

    Each comparison in `DIFF_COMPARISONS` runs whenever BOTH of its own
    variants' `.out` files are present, independently of every other
    comparison. A truncated (or missing) `.out` for one variant must not cost
    an otherwise-computable comparison for another -- the same "a
    missing/bad thing must not cost an unrelated computable result"
    principle already applied to the iteration-kind guard in
    `report.build_deltas` and, one level up, to per-model isolation in this
    very stage. So a model whose C run crashed still yields B - A, D - A and
    E - A, and a model whose A run crashed still yields D - C.

    Deletion follows one general rule, not a per-variant special case: for
    each `.out` file, the set of comparisons that READ it is computed from
    `DIFF_COMPARISONS`, and the file is deleted only once every comparison
    in that set is resolved -- succeeded on this pass, or terminally
    unavailable (see `_comparison_is_resolved`). It is retained while any
    reader is merely pending (the other variant has not run yet) or
    retryable (it ran `ok` but its diff raised this pass). `a_out` is read by
    four comparisons and `c_out`/`d_out` by two each, so neither `c_out` nor
    `d_out` may be dropped when only `C - A` / `D - A` has flushed --
    `D - C` still needs them.

    The trade is deliberately asymmetric: `.out` files are regenerable by
    re-running the model, whereas the diff rows derived from them are not.
    Over-retention costs disk; over-deletion costs a re-run. Which is why
    rows are always flushed to disk BEFORE any file behind them is unlinked,
    in every path -- batch boundaries and the final remainder alike. Rows
    are flushed in batches of `FLUSH_BATCH_SIZE` models, and only the `.out`
    files behind an already-flushed batch are deleted.
    """
    from . import outdiff

    out_dir = Path(out_dir)

    runs = store.read_table(out_dir, "runs")
    if runs.empty or "out_path" not in runs.columns:
        return 0

    # `runs` is keyed on (model_id, variant, engine_version, inp_sha256), so
    # two engine builds legitimately coexist in one store. Taking the last
    # `out_path` per variant would then silently diff build 1's A against
    # build 2's E -- a comparison of two different engines wearing one
    # label. `stage_report` already refuses on this; so does this stage.
    versions = executed_engine_versions(runs)
    if len(versions) > 1:
        print(DIFF_MIXED_ENGINE_ERROR)
        for version in versions:
            print(f"  {version}")
        print(DIFF_MIXED_ENGINE_HINT)
        return 1

    executed = runs[runs["variant"].isin(schema.VARIANTS)]

    rows: list[dict] = []
    pending: list[Path] = []
    batch_count = 0

    for model_id, group in executed.groupby("model_id"):
        # Every variant with a run row at all, versus those whose row is
        # usable as a diff input. The difference is exactly what separates a
        # merely-pending comparison (retain its inputs) from a terminally
        # unavailable one (release them).
        recorded = set(group["variant"])
        ok = group[(group["status"] == schema.Status.OK)
                   & group["out_path"].notna()]
        usable = dict(zip(ok["variant"], ok["out_path"]))
        # Empty for a store written before `case_id` existed; the rows then
        # carry nulls rather than the stage refusing to diff at all.
        cases = (dict(zip(ok["variant"], ok["case_id"]))
                 if "case_id" in ok.columns else {})
        family = group["family"].iloc[0]

        succeeded: set[str] = set()
        for comparison in DIFF_COMPARISONS:
            left, right, label = comparison
            if left not in usable or right not in usable:
                continue  # an input is missing; not attempted
            # `outdiff.diff_out_files` reduces `second - first`, so the
            # RIGHT side of the label is the first argument: D_minus_C
            # passes (c_out, d_out) and reports D's divergence from C.
            base_out, other_out = Path(usable[right]), Path(usable[left])
            try:
                diff_rows = outdiff.diff_out_files(base_out, other_out, abs_tol)
            except Exception as error:  # noqa: BLE001  isolate one bad comparison
                print(f"WARNING: diff failed for model {model_id!r} "
                      f"({label}): {error}")
                continue

            for row in diff_rows:
                rows.append({
                    "model_id": model_id, "family": family,
                    "comparison": label,
                    "case_id_left": cases.get(left),
                    "case_id_right": cases.get(right),
                    **row,
                })
            succeeded.add(label)

        for variant, out_path in usable.items():
            readers = [c for c in DIFF_COMPARISONS if variant in (c[0], c[1])]
            if not readers:
                continue  # read by nothing; keep it rather than guess why
            if all(_comparison_is_resolved(c, variant, succeeded, usable, recorded)
                   for c in readers):
                pending.append(Path(out_path))

        batch_count += 1
        if batch_count >= FLUSH_BATCH_SIZE:
            _flush_diff_batch(out_dir, rows, pending)
            rows, pending, batch_count = [], [], 0

    _flush_diff_batch(out_dir, rows, pending)
    return 0


def _replace_table(frame: pd.DataFrame, out_dir: Path, name: str,
                    partition_by: list[str] | None = None) -> None:
    """Delete the existing dataset for a derived table, then write `frame`.

    `models`, `deltas`, `element_deltas` and `topology_status` are pure
    functions of already-persisted (or on-disk) data -- unlike `runs`,
    `elements`, `scalars` and `ts_diff`, which are resumable keyed appends --
    so re-running the stage that produces them must replace them, not append
    to them. store.write_table's basename carries a fresh UUID on every call
    (deliberately, so a resumed sweep accumulates), which means a second run
    would otherwise double every row and silently corrupt any downstream
    aggregation.

    Deleting even when `frame` is empty ensures a report run that no longer
    produces a status (e.g. every reference topology now matches) actually
    clears the stale dataset from a previous run, instead of leaving it
    behind to be read as still current.
    """
    shutil.rmtree(Path(out_dir) / name, ignore_errors=True)
    if not frame.empty:
        store.write_table(frame, out_dir, name, partition_by=partition_by)


def executed_engine_versions(runs: pd.DataFrame) -> list[str]:
    """Distinct engine builds among the executed (non-reference) rows.

    `engine_build_id` is the authority and `engine_version` only the
    fallback: the version string is what the engine PRINTED, and two builds
    with entirely different Anderson, continuity or slot implementations can
    both print `OpenSWMM 6.0`. A guard reading the string alone would wave
    exactly the mix it exists to catch straight through. The fallback exists
    for rows written before the hash did, and is applied per row so a store
    holding both kinds is still screened rather than half-ignored.

    Reference rows are excluded: they always carry the `corpus-reference`
    sentinel, which identifies the corpus anchor rather than any build of ours.
    """
    if runs.empty:
        return []
    executed = runs[runs["variant"].isin(schema.VARIANTS)]
    build = executed.get("engine_build_id")
    version = executed.get("engine_version")
    if build is None and version is None:
        return []
    if build is None:
        identity = version
    elif version is None:
        identity = build
    else:
        identity = build.astype(object).where(build.notna(), version)
    return sorted(identity.dropna().astype(str).unique())


#: Operator-facing console messages `stage_report` prints, keyed by language
#: alongside `report.MARKDOWN_STRINGS` so the two cannot drift apart. Only
#: the messages `stage_report` itself prints are here -- `stage_run` and
#: `stage_diff` warnings are out of scope and stay English-only.
STAGE_REPORT_STRINGS = {
    "en": {
        "empty_store": "no runs to report",
        "mixed_engine_error": (
            "ERROR: `runs` mixes results from more than one engine build; "
            "report refuses to guess which to publish."
        ),
        "mixed_engine_hint": "Re-run `report` against a store holding a single engine build.",
        "anomaly_warning": (
            "WARNING: {n} run(s) across {n_models} model(s) report an option "
            "value that contradicts their variant's intent (the engine may "
            "have silently ignored the option); every delta with such a run "
            "on either side is withheld, and summary.md counts them:"
        ),
        "anomaly_row": "  {model_id} ({variant}): {option} expected {expected!r}, engine reported {reported!r}",
    },
    "es": {
        "empty_store": "no hay corridas para reportar",
        "mixed_engine_error": (
            "ERROR: `runs` mezcla resultados de más de una compilación del "
            "motor; el informe se niega a adivinar cuál publicar."
        ),
        "mixed_engine_hint": "Vuelva a ejecutar `report` sobre un almacén que contenga una sola compilación del motor.",
        "anomaly_warning": (
            "ADVERTENCIA: {n} corrida(s) en {n_models} modelo(s) reportan un "
            "valor de opción que contradice la intención de su variante (el "
            "motor pudo haber ignorado la opción silenciosamente); todo delta "
            "con una corrida así en cualquiera de sus lados se retiene, y "
            "summary.md los cuenta:"
        ),
        "anomaly_row": "  {model_id} ({variant}): {option} esperado {expected!r}, motor reportó {reported!r}",
    },
}


def stage_report(out_dir: Path, lang: str = "es") -> int:
    """Materialise the derived tables and the markdown summary.

    `lang` (`es` or `en`) selects the language of the generated `summary.md`
    prose and of this stage's own console messages; it is passed straight
    through to `report.write_markdown`. Returns a process exit code.
    Non-zero means nothing was written.
    """
    from . import report

    out_dir = Path(out_dir)
    strings = STAGE_REPORT_STRINGS[lang]
    runs = store.read_table(out_dir, "runs")
    if runs.empty:
        print(strings["empty_store"])
        return 0

    # `runs` is keyed on (model_id, variant, engine_version, inp_sha256), so
    # two engine builds legitimately coexist in one store. The report pivots
    # on model_id/variant with aggfunc="first", which would arbitrarily pick
    # one build's value per cell -- a number no reader could attribute. Refuse
    # rather than guess; splitting the store is the operator's call.
    versions = executed_engine_versions(runs)
    if len(versions) > 1:
        print(strings["mixed_engine_error"])
        for version in versions:
            print(f"  {version}")
        print(strings["mixed_engine_hint"])
        return 1

    # A reported anomaly, not a hard failure: an operator deliberately
    # studying a different configuration should not be blocked. It exists
    # because OptionsHandler.cpp silently ignores an unrecognised
    # NODE_CONTINUITY/ANDERSON_ACCEL value -- a typo would otherwise read as
    # "the feature has no effect" across the whole corpus.
    #
    # Printed here AND counted in summary.md (report.write_markdown recomputes
    # it from `runs`), because this console line is seen once by whoever ran
    # the stage while the summary is the artifact that gets kept and shared.
    # `report.build_deltas` reads the same mapping and withholds the deltas.
    anomalies = report.option_anomalies(runs)
    if not anomalies.empty:
        n_models = anomalies["model_id"].nunique()
        print(strings["anomaly_warning"].format(n=len(anomalies), n_models=n_models))
        for _, row in anomalies.iterrows():
            print(strings["anomaly_row"].format(
                model_id=row["model_id"], variant=row["variant"],
                option=row["option"], expected=row["expected"],
                reported=row["reported"],
            ))

    deltas = report.build_deltas(runs, report.DEFAULT_METRICS)
    _replace_table(deltas, out_dir, "deltas", partition_by=["family"])

    elements = store.read_table(out_dir, "elements")
    if not elements.empty:
        element_deltas = report.build_element_deltas(elements)
        _replace_table(element_deltas, out_dir, "element_deltas",
                       partition_by=["family"])
        mismatched = report.topology_status(elements)
        _replace_table(mismatched, out_dir, "topology_status")
        if not mismatched.empty:
            print(f"{len(mismatched)} models have a mismatched reference topology")
    else:
        _replace_table(pd.DataFrame(), out_dir, "element_deltas")
        _replace_table(pd.DataFrame(), out_dir, "topology_status")

    # `elements` is passed so the summary's surcharge-activity stratum can
    # use per-node flooding totals as well as `pct_steps_not_converging`;
    # without it the proxy silently degrades to the scalar signal alone.
    #
    # `ts_diff` is passed for the incomplete-overlap anomaly section. It is
    # read here rather than inside `write_markdown` so the stage stays the one
    # place that touches the store; an absent table is an empty frame, which
    # the section reports as *coverage not assessed* -- never as complete.
    ts_diff = store.read_table(out_dir, "ts_diff")
    path = report.write_markdown(deltas, runs, out_dir / "summary.md",
                                 lang=lang, elements=elements, ts_diff=ts_diff)
    print(f"wrote {path}")
    return 0


#: Stages that read the corpus tree. `diff` and `report` work purely from the
#: Parquet store, so requiring `--corpus-root` for them at the parser level
#: would make the documented `diff`/`report` invocations exit 2 before any
#: stage code ran.
CORPUS_STAGES = ("inventory", "run", "check")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swmmbench")
    parser.add_argument("stage", choices=["inventory", "run", "diff", "report", "check"])
    parser.add_argument("--corpus-root", type=Path, default=None,
                        help=f"required for the {', '.join(CORPUS_STAGES)} stages")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--engine", nargs="+", default=None,
                        help="engine executable and any leading arguments")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--abs-tol", type=float, default=1e-9,
                        help="absolute tolerance on time-series differences")
    parser.add_argument("--lang", choices=["es", "en"], default="es",
                        help="language of the generated summary.md and report "
                             "stage console messages (default: es)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.stage in CORPUS_STAGES and args.corpus_root is None:
        print(f"error: --corpus-root is required for the {args.stage} stage")
        return 2

    if args.stage == "inventory":
        frame = stage_inventory(args.corpus_root, args.out)
        print(f"{len(frame)} models inventoried")
        return 0

    if args.stage == "check":
        clean, leftovers = corpus_is_clean(args.corpus_root)
        print("corpus clean" if clean else f"{len(leftovers)} leftovers")
        return 0 if clean else 1

    if args.stage == "diff":
        return stage_diff(args.out, args.abs_tol)

    if args.stage == "report":
        return stage_report(args.out, args.lang)

    if not args.engine:
        print("error: --engine is required for the run stage")
        return 2

    return stage_run(args.corpus_root, args.out, args.engine,
                     args.timeout, args.jobs, args.limit)
