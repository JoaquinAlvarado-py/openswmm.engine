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


def stage_inventory(corpus_root: Path, out_dir: Path) -> pd.DataFrame:
    records = corpus.discover(Path(corpus_root))
    frame = pd.DataFrame([
        {k: v for k, v in dataclasses.asdict(r).items() if k != "abs_path"}
        for r in records
    ])
    if not frame.empty:
        store.write_table(frame, out_dir, "models", partition_by=["family"])
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

    for element in elements:
        element.update(model_id=record["model_id"], family=record["family"],
                       variant=variant)

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
                       variant=schema.VARIANT_REF)
    return {**base, "status": schema.Status.OK, **scalars}, elements


def stage_run(
    corpus_root: Path,
    out_dir: Path,
    engine: list[str],
    timeout_s: float,
    jobs: int,
    limit: int | None,
) -> None:
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
            id_vars=["model_id", "family", "variant"],
            value_vars=[c for c in rptparse.SCALAR_KEYS
                        if c not in ("reported_version", "start_date", "end_date",
                                     "iteration_metric_kind")],
            var_name="metric", value_name="value",
        )
        store.write_table(scalar_frame, out_dir, "scalars", partition_by=["family"])
    if element_rows:
        store.write_table(pd.DataFrame(element_rows), out_dir, "elements",
                          partition_by=["family"])

    clean, leftovers = corpus_is_clean(corpus_root)
    if not clean:
        print(f"WARNING: {len(leftovers)} temporary decks left in the corpus:")
        for item in leftovers[:10]:
            print(f"  {item}")


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


def stage_diff(out_dir: Path, rel_tol: float = 1e-9) -> None:
    """Diff the retained A/B `.out` pairs, then discard the binaries.

    Each model's diff is isolated: an unreadable or truncated `.out` costs
    only that model's row, never the batch, and its files are left in place
    for a future re-run rather than deleted out from under a lost result.
    Rows are flushed to disk in batches of `FLUSH_BATCH_SIZE` models, and
    only the `.out` files behind an already-flushed batch are deleted.
    """
    from . import outdiff

    out_dir = Path(out_dir)

    runs = store.read_table(out_dir, "runs")
    if runs.empty or "out_path" not in runs.columns:
        return

    ok = runs[(runs["status"] == schema.Status.OK)
              & (runs["variant"].isin(schema.VARIANTS))
              & runs["out_path"].notna()]

    rows: list[dict] = []
    pending: list[Path] = []
    batch_count = 0

    for model_id, group in ok.groupby("model_id"):
        paths = dict(zip(group["variant"], group["out_path"]))
        if set(paths) != set(schema.VARIANTS):
            continue
        a_out = Path(paths[schema.VARIANT_A])
        b_out = Path(paths[schema.VARIANT_B])
        family = group["family"].iloc[0]

        try:
            diff_rows = outdiff.diff_out_files(a_out, b_out, rel_tol)
        except Exception as error:  # noqa: BLE001  isolate one bad model
            print(f"WARNING: diff failed for model {model_id!r}: {error}")
            continue

        for row in diff_rows:
            rows.append({"model_id": model_id, "family": family, **row})
        # Deletion is deferred to `_flush_diff_batch`, after these rows have
        # been written: retaining raw series for 878 models across two
        # configurations would run to billions of rows, but deleting before
        # persisting would let one unreadable `.out` lose every model diffed
        # earlier in the sweep.
        pending.extend([a_out, b_out])
        batch_count += 1

        if batch_count >= FLUSH_BATCH_SIZE:
            _flush_diff_batch(out_dir, rows, pending)
            rows, pending, batch_count = [], [], 0

    _flush_diff_batch(out_dir, rows, pending)


def stage_report(out_dir: Path) -> None:
    from . import report

    out_dir = Path(out_dir)
    runs = store.read_table(out_dir, "runs")
    if runs.empty:
        print("no runs to report")
        return

    deltas = report.build_deltas(runs, report.DEFAULT_METRICS)
    if not deltas.empty:
        store.write_table(deltas, out_dir, "deltas", partition_by=["family"])
    path = report.write_markdown(deltas, runs, out_dir / "summary.md")
    print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swmmbench")
    parser.add_argument("stage", choices=["inventory", "run", "diff", "report", "check"])
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--engine", nargs="+", default=None,
                        help="engine executable and any leading arguments")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rel-tol", type=float, default=1e-9)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.stage == "inventory":
        frame = stage_inventory(args.corpus_root, args.out)
        print(f"{len(frame)} models inventoried")
        return 0

    if args.stage == "check":
        clean, leftovers = corpus_is_clean(args.corpus_root)
        print("corpus clean" if clean else f"{len(leftovers)} leftovers")
        return 0 if clean else 1

    if args.stage == "diff":
        stage_diff(args.out, args.rel_tol)
        return 0

    if args.stage == "report":
        stage_report(args.out)
        return 0

    if not args.engine:
        print("error: --engine is required for the run stage")
        return 2

    stage_run(args.corpus_root, args.out, args.engine,
              args.timeout, args.jobs, args.limit)
    return 0
