import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from swmmbench import cli, outdiff, schema, store

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


def test_a_second_inventory_replaces_rather_than_doubles_the_corpus(
    corpus_root, tmp_path,
):
    # `inventory` is step 1 of every documented sweep, so an appending write
    # would double the corpus -- and every downstream number -- with no error.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    before = store.read_table(out, "models")

    cli.stage_inventory(corpus_root, out)
    after = store.read_table(out, "models")

    assert len(after) == len(before) == 2
    assert sorted(after["model_id"]) == ["EPA/m1", "LID/m2"]


def test_inventory_drops_a_model_that_left_the_corpus(corpus_root, tmp_path):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    (corpus_root / "LID" / "m2.inp").unlink()

    cli.stage_inventory(corpus_root, out)

    assert sorted(store.read_table(out, "models")["model_id"]) == ["EPA/m1"]


def test_run_produces_three_variants_per_model(corpus_root, tmp_path, fake_engine):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    runs = store.read_table(out, "runs")
    executed = runs[runs["variant"].isin(schema.VARIANTS)]

    assert len(executed) == 6  # 2 models x variants A, B and C
    assert set(executed["variant"]) == {"A", "B", "C"}
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


def _diff_runs_fixture(tmp_path):
    """A `runs` table with two OK models, each with real (fake) A/B/C .out files."""
    a1, b1, c1 = (tmp_path / f"bad_{v}.out" for v in "ABC")
    a2, b2, c2 = (tmp_path / f"good_{v}.out" for v in "ABC")
    for path in (a1, b1, c1, a2, b2, c2):
        path.write_bytes(b"not really a binary .out file")

    runs = pd.DataFrame([
        {"model_id": "F/bad", "family": "F", "variant": schema.VARIANT_A,
         "status": schema.Status.OK, "out_path": str(a1)},
        {"model_id": "F/bad", "family": "F", "variant": schema.VARIANT_B,
         "status": schema.Status.OK, "out_path": str(b1)},
        {"model_id": "F/bad", "family": "F", "variant": schema.VARIANT_C,
         "status": schema.Status.OK, "out_path": str(c1)},
        {"model_id": "F/good", "family": "F", "variant": schema.VARIANT_A,
         "status": schema.Status.OK, "out_path": str(a2)},
        {"model_id": "F/good", "family": "F", "variant": schema.VARIANT_B,
         "status": schema.Status.OK, "out_path": str(b2)},
        {"model_id": "F/good", "family": "F", "variant": schema.VARIANT_C,
         "status": schema.Status.OK, "out_path": str(c2)},
    ])
    return runs, (a1, b1, c1), (a2, b2, c2)


def _fake_diff_out_files_raising_for_bad(a_out, b_out, abs_tol):
    if "bad" in str(a_out):
        raise RuntimeError("truncated .out file")
    return [{"element_type": "NODE", "element_id": "n1",
             "attribute": "INVERT_DEPTH", "max_abs": 1.0, "max_rel": 1.0,
             "rmse": 1.0, "first_div_period": None, "first_div_time": None,
             "n_periods": 1}]


def test_diff_stage_does_not_raise_when_one_models_diff_raises(
    tmp_path, monkeypatch,
):
    out = tmp_path / "out"
    runs, _bad_paths, _good_paths = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_for_bad)

    cli.stage_diff(out, abs_tol=1e-9)  # must not raise


def test_diff_stage_keeps_out_files_for_a_model_whose_diff_raised(
    tmp_path, monkeypatch,
):
    out = tmp_path / "out"
    runs, (bad_a, bad_b, bad_c), _good_paths = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_for_bad)

    cli.stage_diff(out, abs_tol=1e-9)

    assert bad_a.exists()
    assert bad_b.exists()
    assert bad_c.exists()


def test_diff_stage_deletes_out_files_and_writes_rows_for_a_succeeding_model(
    tmp_path, monkeypatch,
):
    out = tmp_path / "out"
    runs, _bad_paths, (good_a, good_b, good_c) = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_for_bad)

    cli.stage_diff(out, abs_tol=1e-9)

    assert not good_a.exists()
    assert not good_b.exists()
    assert not good_c.exists()

    ts_diff = store.read_table(out, "ts_diff")
    assert set(ts_diff["model_id"]) == {"F/good"}


def test_diff_stage_emits_both_comparisons_distinguishable_by_column(
    tmp_path, monkeypatch,
):
    out = tmp_path / "out"
    runs, _bad_paths, _good_paths = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_for_bad)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    good = ts_diff[ts_diff["model_id"] == "F/good"]

    assert "comparison" in good.columns
    assert set(good["comparison"]) == {"B_minus_A", "C_minus_A"}
    assert len(good) == 2  # one diff row per comparison from the fake


def test_diff_stage_does_not_delete_a_models_out_files_until_both_comparisons_flush(
    tmp_path, monkeypatch,
):
    # If the second comparison (C - A) fails after the first (B - A)
    # succeeded, `a_out` -- read by both -- must not have been deleted on
    # the strength of the first alone, and nothing for this model should be
    # persisted: the model is isolated as a whole, not comparison-by-comparison.
    out = tmp_path / "out"
    runs, _bad_paths, (good_a, good_b, good_c) = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])

    def diff_fails_for_c(a_out, other_out, abs_tol):
        if "_C" in str(other_out):
            raise RuntimeError("truncated .out file")
        return [{"element_type": "NODE", "element_id": "n1",
                 "attribute": "INVERT_DEPTH", "max_abs": 1.0, "max_rel": 1.0,
                 "rmse": 1.0, "first_div_period": None, "first_div_time": None,
                 "n_periods": 1}]

    monkeypatch.setattr(outdiff, "diff_out_files", diff_fails_for_c)

    cli.stage_diff(out, abs_tol=1e-9)

    assert good_a.exists()
    assert good_b.exists()
    assert good_c.exists()
    ts_diff = store.read_table(out, "ts_diff")
    assert ts_diff.empty or "F/good" not in set(ts_diff["model_id"])


def test_diff_stage_prints_a_warning_naming_the_failed_model(
    tmp_path, monkeypatch, capsys,
):
    out = tmp_path / "out"
    runs, _bad_paths, _good_paths = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_for_bad)

    cli.stage_diff(out, abs_tol=1e-9)

    captured = capsys.readouterr()
    assert "F/bad" in captured.out


def test_multi_worker_runs_produce_the_same_results_as_a_single_worker(
    corpus_root, tmp_path, fake_engine,
):
    # `--jobs 8` is the documented production invocation. Every value in the
    # job dict crosses a process boundary, so an unpicklable addition would
    # break every real sweep while the single-worker path kept passing.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=2, limit=None)

    runs = store.read_table(out, "runs")
    executed = runs[runs["variant"].isin(schema.VARIANTS)]

    assert len(executed) == 6
    assert set(executed["variant"]) == {"A", "B", "C"}
    assert set(executed["status"]) == {schema.Status.OK}
    assert executed["avg_iterations_per_step"].dropna().unique().tolist() == [2.5]


#: FAKE_REPORT plus a Node Depth Summary, so `parse_tables` yields element
#: rows. Kept separate from FAKE_REPORT so the scalar-only tests above keep
#: exercising the minimal report.
FAKE_REPORT_WITH_ELEMENTS = FAKE_REPORT + """
  ******************
  Node Depth Summary
  ******************

  ---------------------------------------------------------------------------------
                                 Average  Maximum  Maximum  Time of Max    Reported
                                   Depth    Depth      HGL   Occurrence   Max Depth
  Node                 Type         Feet     Feet     Feet  days hr:min        Feet
  ---------------------------------------------------------------------------------
  N1                   JUNCTION     4.11    13.40   138.00     0  00:38       12.75
"""


@pytest.fixture
def corpus_with_elements(tmp_path):
    root = tmp_path / "corpus_e"
    (root / "EPA").mkdir(parents=True)
    (root / "EPA" / "m1.inp").write_text(DECK, encoding="latin-1")
    (root / "EPA" / "m1.rpt").write_text(FAKE_REPORT_WITH_ELEMENTS,
                                         encoding="latin-1")
    return root


@pytest.fixture
def engine_with_elements(tmp_path):
    script = tmp_path / "engine_e.py"
    script.write_text(textwrap.dedent(f"""
        import sys, pathlib
        pathlib.Path(sys.argv[2]).write_text(
            {FAKE_REPORT_WITH_ELEMENTS!r}, encoding="latin-1")
        pathlib.Path(sys.argv[3]).write_bytes(b"out")
    """), encoding="utf-8")
    return [sys.executable, str(script)]


def test_element_rows_carry_the_engine_version(
    corpus_with_elements, tmp_path, engine_with_elements,
):
    # Without this column, rows from two engine builds cannot be told apart
    # even in principle, so no pivot over `elements` can be trusted.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_with_elements, out)
    cli.stage_run(corpus_with_elements, out, engine_with_elements,
                  timeout_s=30.0, jobs=1, limit=None)

    elements = store.read_table(out, "elements")

    assert not elements.empty
    assert "engine_version" in elements.columns
    assert elements["engine_version"].notna().all()

    executed = elements[elements["variant"].isin(schema.VARIANTS)]
    assert set(executed["engine_version"]) == {
        cli._engine_version(engine_with_elements)}
    ref = elements[elements["variant"] == schema.VARIANT_REF]
    assert set(ref["engine_version"]) == {cli.REF_ENGINE_VERSION}


def test_scalar_rows_carry_the_engine_version(corpus_root, tmp_path, fake_engine):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    scalars = store.read_table(out, "scalars")

    assert "engine_version" in scalars.columns
    assert scalars["engine_version"].notna().all()


# ---------------------------------------------------------------------------
# The corpus cleanliness gate
# ---------------------------------------------------------------------------


def test_a_dirty_corpus_fails_the_run_stage(corpus_root, tmp_path, fake_engine,
                                            capsys):
    """A leftover deck is a hard failure, not a warning.

    The harness's input is read-only by contract; a surviving temporary deck
    would be inventoried as a corpus model by the next sweep.
    """
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    # Named for a model no longer in the corpus, so this run's own temp_deck
    # cleanup cannot remove it -- exactly what a killed earlier sweep leaves.
    leftover = corpus_root / "EPA" / ".swmmbench_A_killed.inp"
    leftover.write_text("x", encoding="latin-1")

    code = cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0,
                         jobs=1, limit=None)

    assert code != 0
    assert ".swmmbench_A_killed.inp" in capsys.readouterr().out


def test_main_returns_non_zero_when_the_run_stage_dirties_the_corpus(
    corpus_root, tmp_path, fake_engine,
):
    # Both entry points must agree: the standalone `check` stage already
    # returns non-zero, and `run` must not disagree with it.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    (corpus_root / "EPA" / ".swmmbench_A_killed.inp").write_text("x", encoding="latin-1")

    code = cli.main(["run", "--corpus-root", str(corpus_root), "--out", str(out),
                     "--engine", *fake_engine])

    assert code != 0


def test_main_returns_zero_for_a_clean_run(corpus_root, tmp_path, fake_engine):
    out = tmp_path / "out"

    code = cli.main(["run", "--corpus-root", str(corpus_root), "--out", str(out),
                     "--engine", *fake_engine])

    assert code == 0


# ---------------------------------------------------------------------------
# --corpus-root applies only to the stages that read the corpus
# ---------------------------------------------------------------------------


def test_diff_is_invocable_without_a_corpus_root(tmp_path):
    # `diff` works purely from the Parquet store; requiring --corpus-root made
    # the documented invocation exit 2 before any stage code ran.
    code = cli.main(["diff", "--out", str(tmp_path / "out")])

    assert code == 0


def test_report_is_invocable_without_a_corpus_root(tmp_path, capsys):
    code = cli.main(["report", "--out", str(tmp_path / "out")])

    assert code == 0
    assert "no runs" in capsys.readouterr().out


def test_inventory_without_a_corpus_root_fails_cleanly(tmp_path, capsys):
    code = cli.main(["inventory", "--out", str(tmp_path / "out")])

    assert code != 0
    assert "--corpus-root" in capsys.readouterr().out


def test_check_without_a_corpus_root_fails_cleanly(tmp_path, capsys):
    code = cli.main(["check", "--out", str(tmp_path / "out")])

    assert code != 0
    assert "--corpus-root" in capsys.readouterr().out


def test_run_without_a_corpus_root_fails_cleanly(tmp_path, capsys):
    code = cli.main(["run", "--out", str(tmp_path / "out"), "--engine", "x"])

    assert code != 0
    assert "--corpus-root" in capsys.readouterr().out
