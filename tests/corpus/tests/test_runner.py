import sys
import textwrap
from pathlib import Path

import pytest

from swmmbench import runner, schema


def _fake_engine(tmp_path: Path, body: str) -> Path:
    """A Python script standing in for the engine CLI: argv = inp rpt out."""
    script = tmp_path / "fake_engine.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return script


def _run(script: Path, tmp_path: Path, deck: Path, timeout_s: float = 30.0):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return runner.run_once(
        engine=[sys.executable, str(script)],
        deck=deck,
        work_dir=work,
        timeout_s=timeout_s,
    )


@pytest.fixture
def deck(tmp_path):
    path = tmp_path / "model.inp"
    path.write_text("[OPTIONS]\n", encoding="latin-1")
    return path


def test_successful_run_is_ok_and_reports_wall_time(tmp_path, deck):
    script = _fake_engine(tmp_path, """
        import sys, pathlib
        pathlib.Path(sys.argv[2]).write_text("report", encoding="latin-1")
        pathlib.Path(sys.argv[3]).write_bytes(b"binary")
    """)

    result = _run(script, tmp_path, deck)

    assert result.status == schema.Status.OK
    assert result.exit_code == 0
    assert result.wall_ms > 0
    assert result.rpt_path.read_text(encoding="latin-1") == "report"
    assert result.out_path.exists()


def test_nonzero_exit_with_a_report_is_engine_error_not_crash(tmp_path, deck):
    script = _fake_engine(tmp_path, """
        import sys, pathlib
        pathlib.Path(sys.argv[2]).write_text("ERROR 138: deck is bad", encoding="latin-1")
        sys.exit(1)
    """)

    result = _run(script, tmp_path, deck)

    assert result.status == schema.Status.ENGINE_ERROR
    assert result.exit_code == 1
    assert result.rpt_path is not None


def test_nonzero_exit_without_a_report_is_a_crash(tmp_path, deck):
    script = _fake_engine(tmp_path, """
        import sys
        sys.stderr.write("segfault-ish\\n")
        sys.exit(3)
    """)

    result = _run(script, tmp_path, deck)

    assert result.status == schema.Status.CRASH
    assert result.rpt_path is None
    assert "segfault-ish" in result.stderr_tail


def test_a_hang_is_a_timeout_not_a_hang(tmp_path, deck):
    script = _fake_engine(tmp_path, """
        import time
        time.sleep(30)
    """)

    result = _run(script, tmp_path, deck, timeout_s=1.0)

    assert result.status == schema.Status.TIMEOUT
    assert result.wall_ms >= 1000


def test_runner_never_propagates_a_missing_engine(tmp_path, deck):
    result = runner.run_once(
        engine=[str(tmp_path / "does_not_exist")],
        deck=deck,
        work_dir=tmp_path / "work2",
        timeout_s=5.0,
    )

    assert result.status == schema.Status.CRASH


def test_the_deck_directory_is_the_working_directory(tmp_path, deck):
    # Corpus decks resolve DataFiles/*.dat relative to their own directory.
    script = _fake_engine(tmp_path, """
        import sys, pathlib, os
        pathlib.Path(sys.argv[2]).write_text(os.getcwd(), encoding="latin-1")
    """)

    result = _run(script, tmp_path, deck)

    assert Path(result.rpt_path.read_text(encoding="latin-1")).resolve() == deck.parent.resolve()
