"""Stage dispatch for the corpus benchmark harness."""

from __future__ import annotations

import argparse
import dataclasses
import platform
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import corpus, rptparse, runner, schema, store, variants

RESUME_KEY = ["model_id", "variant", "engine_version", "inp_sha256"]

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


def _engine_version(engine: list[str]) -> str:
    """Best-effort engine identity; falls back to the argv itself."""
    import subprocess
    try:
        proc = subprocess.run(
            list(engine) + ["--version"],
            capture_output=True, text=True, timeout=30, errors="replace",
        )
        line = (proc.stdout or proc.stderr).strip().splitlines()
        if line:
            return line[0][:200]
    except (OSError, subprocess.SubprocessError):
        pass
    return " ".join(engine)[:200]


def _run_one(job: dict) -> dict:
    """Execute one (model, variant) pair. Never raises."""
    record = job["record"]
    variant = job["variant"]
    work = Path(job["work_dir"]) / record["model_id"].replace("/", "__") / variant
    row = {
        "model_id": record["model_id"],
        "family": record["family"],
        "variant": variant,
        "engine_version": job["engine_version"],
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

    # `engine_version` travels with every element row for the same reason it
    # is part of the `runs` resume key: two engine builds legitimately coexist
    # in one store, and without the column their rows cannot be told apart --
    # not even in principle -- so any pivot over them would silently mix builds.
    for element in elements:
        element.update(model_id=record["model_id"], family=record["family"],
                       variant=variant, engine_version=job["engine_version"])

    return {"run": {**row, **scalars}, "elements": elements}


#: Stable identity for reference rows. It must NOT be the version string read
#: out of the report: the resume key is compared against this value before the
#: report has been parsed, so a parsed-version key would never match and every
#: resumed sweep would duplicate its reference rows.
REF_ENGINE_VERSION = "corpus-reference"


def _reference_rows(corpus_root: Path, record: dict) -> tuple[dict, list[dict]]:
    """Parse the anchor report. It is never re-run, only read.

    A model with no anchor still gets a row, carrying status `no_ref`. An
    absent row would be indistinguishable from a model the sweep skipped.
    """
    base = {
        "model_id": record["model_id"],
        "family": record["family"],
        "variant": schema.VARIANT_REF,
        "engine_version": REF_ENGINE_VERSION,
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
        element.update(model_id=record["model_id"], family=record["family"],
                       variant=schema.VARIANT_REF,
                       engine_version=REF_ENGINE_VERSION)
    return {**base, "status": schema.Status.OK, **scalars}, elements


def stage_run(
    corpus_root: Path,
    out_dir: Path,
    engine: list[str],
    timeout_s: float,
    jobs: int,
    limit: int | None,
) -> int:
    """Run every outstanding (model, variant) pair. Returns a process exit code.

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

    engine_version = _engine_version(engine)
    done = store.completed_keys(out_dir, "runs", RESUME_KEY)

    work_dir = Path(out_dir) / "work"
    payload = []
    for record in models.to_dict("records"):
        for variant in schema.VARIANTS:
            key = (record["model_id"], variant, engine_version,
                   str(record["inp_sha256"]))
            if key in done:
                continue
            payload.append({
                "record": record, "variant": variant, "engine": engine,
                "engine_version": engine_version, "timeout_s": timeout_s,
                "corpus_root": str(corpus_root), "work_dir": str(work_dir),
            })

    if jobs > 1 and payload:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_run_one, payload))
    else:
        results = [_run_one(job) for job in payload]

    run_rows = [r["run"] for r in results]
    element_rows = [e for r in results for e in r["elements"]]

    for record in models.to_dict("records"):
        key = (record["model_id"], schema.VARIANT_REF, REF_ENGINE_VERSION,
               str(record["inp_sha256"]))
        if key in done:
            continue
        ref_row, ref_elements = _reference_rows(corpus_root, record)
        run_rows.append(ref_row)
        element_rows.extend(ref_elements)

    if run_rows:
        store.write_table(pd.DataFrame(run_rows), out_dir, "runs",
                          partition_by=["family"])
        scalar_frame = pd.DataFrame(run_rows).melt(
            # `engine_version` is an id, not a metric: without it a scalar row
            # from one engine build is indistinguishable from another's.
            id_vars=["model_id", "family", "variant", "engine_version"],
            value_vars=[c for c in rptparse.SCALAR_KEYS
                        if c not in NON_METRIC_SCALARS],
            var_name="metric", value_name="value",
        )
        store.write_table(scalar_frame, out_dir, "scalars", partition_by=["family"])
    if element_rows:
        store.write_table(pd.DataFrame(element_rows), out_dir, "elements",
                          partition_by=["family"])

    clean, leftovers = corpus_is_clean(corpus_root)
    if not clean:
        print(f"ERROR: {len(leftovers)} temporary decks left in the corpus:")
        for item in leftovers[:10]:
            print(f"  {item}")
        return 1
    return 0


def corpus_is_clean(corpus_root: Path) -> tuple[bool, list[str]]:
    """True when no harness temporary deck survives in the corpus."""
    leftovers = [
        str(path)
        for path in Path(corpus_root).rglob(f"{variants.TEMP_PREFIX}*")
    ]
    return not leftovers, leftovers


#: How many models' rows are persisted between flushes. Bounds the cost of a
#: crash (or an unhandled exception) mid-sweep to at most one unflushed
#: batch, rather than every model diffed since the stage started.
FLUSH_BATCH_SIZE = 50


def _flush_diff_batch(out_dir: Path, rows: list[dict], pending: list[Path]) -> None:
    """Persist accumulated diff rows, then delete the `.out` files behind them.

    Order matters: writing before deleting means a crash between the two
    calls leaves `.out` files undeleted (harmless, re-runnable) rather than
    deleting source data whose derived rows were never written (irrecoverable).
    """
    if rows:
        store.write_table(pd.DataFrame(rows), out_dir, "ts_diff",
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
                    "comparison": label, **row,
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
    import shutil

    shutil.rmtree(Path(out_dir) / name, ignore_errors=True)
    if not frame.empty:
        store.write_table(frame, out_dir, name, partition_by=partition_by)


def executed_engine_versions(runs: pd.DataFrame) -> list[str]:
    """Distinct engine builds among the executed (non-reference) rows.

    Reference rows are excluded: they always carry the `corpus-reference`
    sentinel, which identifies the corpus anchor rather than any build of ours.
    """
    if runs.empty or "engine_version" not in runs.columns:
        return []
    executed = runs[runs["variant"].isin(schema.VARIANTS)]
    return sorted(executed["engine_version"].dropna().astype(str).unique())


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
    path = report.write_markdown(deltas, runs, out_dir / "summary.md",
                                 lang=lang, elements=elements)
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
