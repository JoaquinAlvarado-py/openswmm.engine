"""One subprocess per simulation.

Process isolation is the point: on a corpus of 1646 third-party models a
crash or a hang is data about the engine, not an interruption of the sweep.
Nothing in this module may raise on a per-model failure.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a declared dependency
    psutil = None

from . import schema

STDERR_TAIL_CHARS = 2000
POLL_INTERVAL_S = 0.05


@dataclass
class RunResult:
    status: str
    exit_code: int | None
    signal: int | None
    wall_ms: float
    peak_rss_mb: float | None
    rpt_path: Path | None
    out_path: Path | None
    stderr_tail: str


def _peak_rss_mb(process: subprocess.Popen, deadline: float) -> float | None:
    """Sample RSS while the child runs. Returns None without psutil."""
    if psutil is None:
        return None
    try:
        handle = psutil.Process(process.pid)
    except psutil.Error:
        return None

    peak = 0.0
    while process.poll() is None and time.monotonic() < deadline:
        try:
            peak = max(peak, handle.memory_info().rss / (1024 * 1024))
        except psutil.Error:
            break
        time.sleep(POLL_INTERVAL_S)
    return peak or None


def run_once(
    engine: list[str] | str,
    deck: Path,
    work_dir: Path,
    timeout_s: float,
) -> RunResult:
    """Run `engine <deck> <rpt> <out>` with the deck's directory as CWD."""
    engine_argv = [engine] if isinstance(engine, str) else list(engine)
    deck = Path(deck)
    work_dir = Path(work_dir)

    rpt = work_dir / f"{deck.stem}.rpt"
    out = work_dir / f"{deck.stem}.out"
    argv = engine_argv + [str(deck), str(rpt), str(out)]

    started = time.monotonic()
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            argv,
            cwd=str(deck.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
    except OSError as error:
        return RunResult(
            status=schema.Status.CRASH,
            exit_code=None, signal=None,
            wall_ms=(time.monotonic() - started) * 1000.0,
            peak_rss_mb=None, rpt_path=None, out_path=None,
            stderr_tail=str(error),
        )

    deadline = started + timeout_s
    peak_rss = _peak_rss_mb(process, deadline)

    timed_out = False
    try:
        _, stderr = process.communicate(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            process.kill()
        except OSError:
            # Process exited in the race window between timeout and kill.
            pass
        _, stderr = process.communicate()

    wall_ms = (time.monotonic() - started) * 1000.0
    code = process.returncode
    signal_number = -code if code is not None and code < 0 else None

    rpt_path = rpt if rpt.is_file() and rpt.stat().st_size > 0 else None
    out_path = out if out.is_file() and out.stat().st_size > 0 else None

    if timed_out:
        status = schema.Status.TIMEOUT
    elif code == 0:
        status = schema.Status.OK
    elif rpt_path is not None:
        # SWMM diagnosed the failure and wrote it down. The report is the
        # most informative artifact this run produced.
        status = schema.Status.ENGINE_ERROR
    else:
        status = schema.Status.CRASH

    return RunResult(
        status=status,
        exit_code=code,
        signal=signal_number,
        wall_ms=wall_ms,
        peak_rss_mb=peak_rss,
        rpt_path=rpt_path,
        out_path=out_path,
        stderr_tail=(stderr or "")[-STDERR_TAIL_CHARS:],
    )
