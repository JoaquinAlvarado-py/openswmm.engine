import sys
import textwrap
from pathlib import Path

import pytest

from swmmbench import cli, schema, store

DECK = "[TITLE]\nt\n\n[OPTIONS]\nFLOW_UNITS CFS\n"

FAKE_REPORT = """  EPA STORM WATER MANAGEMENT MODEL - VERSION 5.2 (Build 5.2.4)

  ****************
  Analysis Options
  ****************
  Starting Date ............ 01/01/2002 00:00:00
  Ending Date .............. 01/01/2002 01:00:00

  **************************        Volume        Volume
  Flow Routing Continuity        hectare-m      10^6 ltr
  **************************     ---------     ---------
  Continuity Error (%) .....         0.500

  *************************
  Routing Time Step Summary
  *************************
  Minimum Time Step           :    10.00 sec
  Average Time Step           :    10.00 sec
  Maximum Time Step           :    10.00 sec
  Average Iterations per Step :     2.50
  % of Steps Not Converging   :     1.00
"""


@pytest.fixture
def corpus_root(tmp_path):
    root = tmp_path / "corpus"
    (root / "EPA").mkdir(parents=True)
    (root / "EPA" / "m1.inp").write_text(DECK, encoding="latin-1")
    (root / "EPA" / "m1.rpt").write_text(FAKE_REPORT, encoding="latin-1")
    (root / "LID").mkdir()
    (root / "LID" / "m2.inp").write_text(DECK, encoding="latin-1")
    return root


@pytest.fixture
def fake_engine(tmp_path):
    script = tmp_path / "engine.py"
    script.write_text(textwrap.dedent(f"""
        import sys, pathlib
        pathlib.Path(sys.argv[2]).write_text({FAKE_REPORT!r}, encoding="latin-1")
        pathlib.Path(sys.argv[3]).write_bytes(b"out")
    """), encoding="utf-8")
    return [sys.executable, str(script)]


def test_inventory_writes_one_row_per_model(corpus_root, tmp_path):
    frame = cli.stage_inventory(corpus_root, tmp_path / "out")

    assert len(frame) == 2
    assert set(frame["model_id"]) == {"EPA/m1", "LID/m2"}
    assert store.read_table(tmp_path / "out", "models").shape[0] == 2


def test_run_produces_two_variants_per_model(corpus_root, tmp_path, fake_engine):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    runs = store.read_table(out, "runs")
    executed = runs[runs["variant"].isin(schema.VARIANTS)]

    assert len(executed) == 4  # 2 models x variants A and B
    assert set(executed["variant"]) == {"A", "B"}
    assert set(executed["status"]) == {schema.Status.OK}


def test_a_model_without_an_anchor_gets_a_no_ref_row(corpus_root, tmp_path, fake_engine):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    runs = store.read_table(out, "runs")
    ref = runs[runs["variant"] == schema.VARIANT_REF].set_index("model_id")

    # An absent row would be indistinguishable from a skipped model.
    assert ref.loc["LID/m2", "status"] == schema.Status.NO_REF
    assert ref.loc["EPA/m1", "status"] == schema.Status.OK


def test_run_records_the_parsed_iteration_metric(corpus_root, tmp_path, fake_engine):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    runs = store.read_table(out, "runs")

    assert runs["avg_iterations_per_step"].dropna().unique().tolist() == [2.5]
    # LID/m2's REF row carries status no_ref and all-None scalars by design
    # (a model with no anchor still gets a row); its iteration_metric_kind
    # is therefore None and must be excluded here, exactly as the assertion
    # above excludes its avg_iterations_per_step via dropna().
    assert set(runs["iteration_metric_kind"].dropna()) == {schema.ITER_PICARD}


def test_reference_reports_are_parsed_without_being_run(corpus_root, tmp_path, fake_engine):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    scalars = store.read_table(out, "scalars")
    ref = scalars[(scalars["variant"] == schema.VARIANT_REF)
                  & (scalars["metric"] == "continuity_error_flow")]

    parsed = ref.dropna(subset=["value"])

    assert set(parsed["model_id"]) == {"EPA/m1"}  # LID/m2 has no reference
    assert parsed["value"].iloc[0] == pytest.approx(0.5)


def test_a_second_run_skips_completed_work(corpus_root, tmp_path, fake_engine):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)
    before = len(store.read_table(out, "runs"))

    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    assert len(store.read_table(out, "runs")) == before


def test_the_corpus_is_left_clean(corpus_root, tmp_path, fake_engine):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    clean, leftovers = cli.corpus_is_clean(corpus_root)

    assert clean, leftovers


def test_cleanliness_check_detects_a_leftover_deck(corpus_root):
    (corpus_root / "EPA" / ".swmmbench_A_m1.inp").write_text("x", encoding="latin-1")

    clean, leftovers = cli.corpus_is_clean(corpus_root)

    assert not clean
    assert any("m1" in item for item in leftovers)


def test_main_requires_an_engine_for_the_run_stage(corpus_root, tmp_path, capsys):
    code = cli.main(["run", "--corpus-root", str(corpus_root),
                     "--out", str(tmp_path / "out")])

    assert code != 0
