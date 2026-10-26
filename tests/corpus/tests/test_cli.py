import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from swmmbench import (cli, corpus, outdiff, report, rptparse, schema, store,
                       variants)

DECK = "[TITLE]\nt\n\n[OPTIONS]\nFLOW_UNITS CFS\n"

#: One `outdiff.diff_out_files` row, as the real reducer would return it --
#: including every `outdiff.COVERAGE_FIELDS` entry, so the fakes below
#: exercise the same `ts_diff` schema the engine-backed path writes. Spelled
#: once so a field added to the reducer is added here, not in four places.
FAKE_DIFF_ROW = {
    "element_type": "NODE", "element_id": "n1", "attribute": "INVERT_DEPTH",
    "max_abs": 1.0, "max_rel": 1.0, "rmse": 1.0,
    "first_div_period": None, "first_div_time": None,
    "n_periods_left": 1, "n_periods_right": 1, "n_common": 1,
    "coverage_fraction": 1.0, "start_time_match": True,
    "end_time_match": True, "time_grid_match": True,
}

FAKE_REPORT = """  EPA STORM WATER MANAGEMENT MODEL - VERSION 5.2 (Build 5.2.4)

  ****************
  Analysis Options
  ****************
  Flow Routing Method ...... DYNWAVE
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


def test_run_produces_five_variants_per_model(corpus_root, tmp_path, fake_engine):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    runs = store.read_table(out, "runs")
    executed = runs[runs["variant"].isin(schema.VARIANTS)]

    assert len(executed) == 10  # 2 models x variants A, B, C, D and E
    assert set(executed["variant"]) == {"A", "B", "C", "D", "E"}
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
    """A `runs` table with two OK models, each with real (fake) .out files
    for all five variants (A-E), so every DIFF_COMPARISONS entry is
    exercised."""
    bad = tuple(tmp_path / f"bad_{v}.out" for v in schema.VARIANTS)
    good = tuple(tmp_path / f"good_{v}.out" for v in schema.VARIANTS)
    for path in bad + good:
        path.write_bytes(b"not really a binary .out file")

    rows = []
    for model_id, paths in (("F/bad", bad), ("F/good", good)):
        for variant, path in zip(schema.VARIANTS, paths):
            rows.append({"model_id": model_id, "family": "F", "variant": variant,
                        "status": schema.Status.OK, "out_path": str(path)})
    return pd.DataFrame(rows), bad, good


def _fake_diff_out_files_raising_for_bad(a_out, b_out, abs_tol):
    if "bad" in str(a_out):
        raise RuntimeError("truncated .out file")
    return [dict(FAKE_DIFF_ROW)]


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
    runs, (bad_a, bad_b, bad_c, bad_d, bad_e), _good_paths = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_for_bad)

    cli.stage_diff(out, abs_tol=1e-9)

    assert bad_a.exists()
    assert bad_b.exists()
    assert bad_c.exists()
    assert bad_d.exists()
    assert bad_e.exists()


def test_diff_stage_deletes_out_files_and_writes_rows_for_a_succeeding_model(
    tmp_path, monkeypatch,
):
    out = tmp_path / "out"
    runs, _bad_paths, (good_a, good_b, good_c, good_d, good_e) = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_for_bad)

    cli.stage_diff(out, abs_tol=1e-9)

    assert not good_a.exists()
    assert not good_b.exists()
    assert not good_c.exists()
    assert not good_d.exists()
    assert not good_e.exists()

    ts_diff = store.read_table(out, "ts_diff")
    assert set(ts_diff["model_id"]) == {"F/good"}


def test_diff_stage_emits_all_five_comparisons_distinguishable_by_column(
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
    # `D_minus_C` is not derivable downstream from `D_minus_A` and
    # `C_minus_A` -- these are non-linear reductions of a pointwise
    # difference -- so it must be produced here as its own comparison.
    assert set(good["comparison"]) == {
        "B_minus_A", "C_minus_A", "D_minus_A", "E_minus_A", "D_minus_C",
    }
    assert len(good) == 5  # one diff row per comparison from the fake


def _fake_diff_out_files_raising_when(fail_needle):
    """A fake `diff_out_files` that fails only for the comparison whose
    `other_out` path contains `fail_needle`, succeeding for every other."""
    def fake(a_out, other_out, abs_tol):
        if fail_needle in str(other_out):
            raise RuntimeError(f"truncated {fail_needle} .out file")
        return [dict(FAKE_DIFF_ROW)]
    return fake


def _fake_diff_out_files_always_raising(a_out, other_out, abs_tol):
    raise RuntimeError("truncated .out file")


def test_a_failing_b_minus_a_does_not_cost_the_other_comparisons(
    tmp_path, monkeypatch,
):
    # b_out is the only bad input; a_out, c_out, d_out and e_out are
    # perfectly readable, so C - A, D - A, E - A and D - C must still be
    # computed and persisted. Only b_out -- the input the failed comparison
    # actually read -- may be deleted... and since it failed, it must NOT
    # be: it stays for a future retry of B - A. a_out is read by four
    # comparisons, so with B - A still outstanding it must survive too.
    out = tmp_path / "out"
    runs, _bad_paths, (good_a, good_b, good_c, good_d, good_e) = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_when("_B"))

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    good = ts_diff[ts_diff["model_id"] == "F/good"]
    assert set(good["comparison"]) == {
        "C_minus_A", "D_minus_A", "E_minus_A", "D_minus_C",
    }

    assert good_a.exists()
    assert good_b.exists()
    # c_out and d_out are each read by two comparisons; both of each pair
    # succeeded here, so both files are releasable.
    assert not good_c.exists()
    assert not good_d.exists()
    assert not good_e.exists()


def test_a_failing_c_minus_a_does_not_cost_the_other_comparisons(
    tmp_path, monkeypatch,
):
    # Mirror of the above: the C - A comparison is the only failing one
    # (the fake keys off the non-baseline argument, so D - C, which also
    # reads c_out, still succeeds).
    out = tmp_path / "out"
    runs, _bad_paths, (good_a, good_b, good_c, good_d, good_e) = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_when("_C"))

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    good = ts_diff[ts_diff["model_id"] == "F/good"]
    assert set(good["comparison"]) == {
        "B_minus_A", "D_minus_A", "E_minus_A", "D_minus_C",
    }

    assert good_a.exists()
    assert not good_b.exists()
    # C - A still needs a retry, so c_out stays even though its other
    # reader (D - C) is done with it.
    assert good_c.exists()
    assert not good_d.exists()
    assert not good_e.exists()


def test_all_comparisons_failing_writes_nothing_and_keeps_all_five_files(
    tmp_path, monkeypatch,
):
    out = tmp_path / "out"
    runs, _bad_paths, (good_a, good_b, good_c, good_d, good_e) = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_raising)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    assert ts_diff.empty or "F/good" not in set(ts_diff["model_id"])
    assert good_a.exists()
    assert good_b.exists()
    assert good_c.exists()
    assert good_d.exists()
    assert good_e.exists()


def test_a_out_is_deleted_only_once_every_comparison_reading_it_has_flushed(
    tmp_path, monkeypatch,
):
    # Every comparison but E - A succeeds. Three out of four A-anchored
    # comparisons flushing is not enough: a_out is read by all four, so it
    # must survive until E - A (or a retry of it) also flushes.
    out = tmp_path / "out"
    runs, _bad_paths, (good_a, good_b, good_c, good_d, good_e) = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_when("_E"))

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    good = ts_diff[ts_diff["model_id"] == "F/good"]
    assert set(good["comparison"]) == {
        "B_minus_A", "C_minus_A", "D_minus_A", "D_minus_C",
    }

    assert good_a.exists()
    assert good_e.exists()
    assert not good_b.exists()
    assert not good_c.exists()
    assert not good_d.exists()


def _fake_diff_out_files_raising_only_for_d_minus_c(base_out, other_out, abs_tol):
    """Fails exactly the D - C comparison: it is the only one whose baseline
    (first) argument is the C `.out`."""
    if "_C" in str(base_out) and "_D" in str(other_out):
        raise RuntimeError("truncated .out file")
    return [dict(FAKE_DIFF_ROW)]


def test_c_out_survives_c_minus_a_when_d_minus_c_has_not_flushed(
    tmp_path, monkeypatch,
):
    # c_out is read by TWO comparisons now. Deleting it the moment C - A
    # flushes would strand D - C -- the only honest measurement of Anderson
    # under Crank-Nicolson, and the sole reason variant D exists.
    out = tmp_path / "out"
    runs, _bad_paths, (good_a, good_b, good_c, good_d, good_e) = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_raising_only_for_d_minus_c)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    good = ts_diff[ts_diff["model_id"] == "F/good"]
    assert set(good["comparison"]) == {
        "B_minus_A", "C_minus_A", "D_minus_A", "E_minus_A",
    }

    # Both of D - C's inputs stay for its retry, even though each has
    # already served its A-anchored comparison.
    assert good_c.exists()
    assert good_d.exists()
    # a_out is read by the four A-anchored comparisons only; all four
    # flushed, so it goes.
    assert not good_a.exists()
    assert not good_b.exists()
    assert not good_e.exists()


def test_c_out_is_deleted_once_both_c_minus_a_and_d_minus_c_have_flushed(
    tmp_path, monkeypatch,
):
    # The other half of the rule above: once every comparison that reads
    # c_out has succeeded and flushed, retaining it is pure waste.
    out = tmp_path / "out"
    runs, _bad_paths, (good_a, good_b, good_c, good_d, good_e) = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    good = ts_diff[ts_diff["model_id"] == "F/good"]
    assert "C_minus_A" in set(good["comparison"])
    assert "D_minus_C" in set(good["comparison"])

    assert not good_c.exists()
    assert not good_d.exists()


def _custom_diff_runs_fixture(tmp_path, spec):
    """One model whose per-variant `runs` rows are given as
    {variant: status} or {variant: (status, has_out_file)}."""
    paths = {}
    rows = []
    for variant, value in spec.items():
        status, has_out = value if isinstance(value, tuple) else (value, True)
        out_path = None
        if has_out:
            path = tmp_path / f"custom_{variant}.out"
            path.write_bytes(b"not really a binary .out file")
            paths[variant] = path
            out_path = str(path)
        rows.append({"model_id": "F/custom", "family": "F", "variant": variant,
                     "status": status, "out_path": out_path})
    return pd.DataFrame(rows), paths


def test_out_files_are_deleted_when_their_remaining_comparisons_are_terminal(
    tmp_path, monkeypatch,
):
    # C - A succeeds. Every other comparison reading a_out or c_out is
    # terminally unavailable: B and D crashed, E ran `ok` but produced no
    # `.out`. The resume key means `stage_run` will never retry those runs
    # for this build, so no future `diff` can ever produce B - A, D - A,
    # E - A or D - C here -- retaining their inputs would be waste with no
    # recovery path.
    out = tmp_path / "out"
    runs, paths = _custom_diff_runs_fixture(tmp_path, {
        schema.VARIANT_A: schema.Status.OK,
        schema.VARIANT_B: (schema.Status.CRASH, False),
        schema.VARIANT_C: schema.Status.OK,
        schema.VARIANT_D: (schema.Status.TIMEOUT, False),
        schema.VARIANT_E: (schema.Status.OK, False),  # ok, but null out_path
    })
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    rows = ts_diff[ts_diff["model_id"] == "F/custom"]
    assert set(rows["comparison"]) == {"C_minus_A"}

    assert not paths[schema.VARIANT_A].exists()
    assert not paths[schema.VARIANT_C].exists()


def test_c_out_is_retained_when_d_has_no_run_row_at_all(tmp_path, monkeypatch):
    # The resume case, and the reason "not succeeded" cannot simply mean
    # "delete": someone may run A/B/C, diff, then add D/E and diff again.
    # D has no row here, so D - C is merely pending -- deleting c_out would
    # destroy a computable comparison rather than a dead one.
    out = tmp_path / "out"
    runs, paths = _custom_diff_runs_fixture(tmp_path, {
        schema.VARIANT_A: schema.Status.OK,
        schema.VARIANT_C: schema.Status.OK,
    })
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    rows = ts_diff[ts_diff["model_id"] == "F/custom"]
    assert set(rows["comparison"]) == {"C_minus_A"}

    assert paths[schema.VARIANT_C].exists()
    assert paths[schema.VARIANT_A].exists()


def test_d_minus_c_is_produced_even_when_the_a_run_is_unavailable(
    tmp_path, monkeypatch,
):
    # No comparison depends on another's inputs: D - C reads C and D only,
    # so a crashed A run costs the A-anchored comparisons and nothing else.
    out = tmp_path / "out"
    runs, paths = _custom_diff_runs_fixture(tmp_path, {
        schema.VARIANT_A: (schema.Status.CRASH, False),
        schema.VARIANT_C: schema.Status.OK,
        schema.VARIANT_D: schema.Status.OK,
    })
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    rows = ts_diff[ts_diff["model_id"] == "F/custom"]
    assert set(rows["comparison"]) == {"D_minus_C"}


def _mixed_engine_diff_runs(tmp_path):
    """One model whose A ran under build 1 and whose C ran under build 2,
    plus a REF row carrying the fixed `corpus-reference` sentinel."""
    rows = []
    paths = {}
    for variant, version in ((schema.VARIANT_A, "openswmm 1.0"),
                             (schema.VARIANT_C, "openswmm 2.0")):
        path = tmp_path / f"mixed_{variant}.out"
        path.write_bytes(b"not really a binary .out file")
        paths[variant] = path
        rows.append({"model_id": "F/mixed", "family": "F", "variant": variant,
                     "engine_version": version, "status": schema.Status.OK,
                     "out_path": str(path)})
    rows.append({"model_id": "F/mixed", "family": "F",
                 "variant": schema.VARIANT_REF,
                 "engine_version": cli.REF_ENGINE_VERSION,
                 "status": schema.Status.OK, "out_path": None})
    return pd.DataFrame(rows), paths


def test_diff_refuses_a_store_holding_two_engine_builds(
    tmp_path, monkeypatch, capsys,
):
    # `runs` is keyed on engine_version, so two builds legitimately coexist.
    # Taking the last out_path per variant would diff build 1's A against
    # build 2's C under a single label -- a number attributable to neither.
    out = tmp_path / "out"
    runs, paths = _mixed_engine_diff_runs(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    code = cli.stage_diff(out, abs_tol=1e-9)

    assert code != 0
    ts_diff = store.read_table(out, "ts_diff")
    assert ts_diff.empty
    # Nothing written means nothing deleted either.
    assert paths[schema.VARIANT_A].exists()
    assert paths[schema.VARIANT_C].exists()

    captured = capsys.readouterr()
    assert "openswmm 1.0" in captured.out
    assert "openswmm 2.0" in captured.out


def test_diff_does_not_count_the_reference_sentinel_as_an_engine_build(
    tmp_path, monkeypatch,
):
    # REF rows always carry `corpus-reference`; counting it would make every
    # single-build store look mixed and refuse every sweep.
    out = tmp_path / "out"
    runs, paths = _custom_diff_runs_fixture(tmp_path, {
        schema.VARIANT_A: schema.Status.OK,
        schema.VARIANT_C: schema.Status.OK,
    })
    runs["engine_version"] = "openswmm 1.0"
    runs = pd.concat([runs, pd.DataFrame([{
        "model_id": "F/custom", "family": "F", "variant": schema.VARIANT_REF,
        "engine_version": cli.REF_ENGINE_VERSION,
        "status": schema.Status.OK, "out_path": None,
    }])], ignore_index=True)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    code = cli.stage_diff(out, abs_tol=1e-9)

    assert code == 0
    ts_diff = store.read_table(out, "ts_diff")
    assert set(ts_diff["comparison"]) == {"C_minus_A"}


def _partial_diff_runs_fixture(tmp_path, present_variants):
    """One OK model whose `runs` rows exist only for `present_variants`."""
    paths = {}
    for variant in present_variants:
        path = tmp_path / f"partial_{variant}.out"
        path.write_bytes(b"not really a binary .out file")
        paths[variant] = path

    runs = pd.DataFrame([
        {"model_id": "F/partial", "family": "F", "variant": variant,
         "status": schema.Status.OK, "out_path": str(path)}
        for variant, path in paths.items()
    ])
    return runs, paths


def _fake_diff_out_files_always_succeeding(a_out, other_out, abs_tol):
    return [dict(FAKE_DIFF_ROW)]


def test_a_missing_c_run_still_yields_b_minus_a(tmp_path, monkeypatch):
    # C is the newest scheme and, across a large third-party corpus, the
    # variant most likely to fail outright (crash/timeout/parse error) --
    # its absence must not cost the model its otherwise-computable B - A.
    out = tmp_path / "out"
    runs, paths = _partial_diff_runs_fixture(
        tmp_path, [schema.VARIANT_A, schema.VARIANT_B])
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    rows = ts_diff[ts_diff["model_id"] == "F/partial"]
    assert set(rows["comparison"]) == {"B_minus_A"}


def test_a_missing_b_run_still_yields_c_minus_a(tmp_path, monkeypatch):
    out = tmp_path / "out"
    runs, paths = _partial_diff_runs_fixture(
        tmp_path, [schema.VARIANT_A, schema.VARIANT_C])
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    rows = ts_diff[ts_diff["model_id"] == "F/partial"]
    assert set(rows["comparison"]) == {"C_minus_A"}


def test_a_missing_d_run_still_yields_the_other_three_comparisons(tmp_path, monkeypatch):
    # A model missing only D (e.g. its D run crashed) must still yield
    # B - A, C - A and E - A -- D's absence costs only D - A.
    out = tmp_path / "out"
    runs, paths = _partial_diff_runs_fixture(
        tmp_path, [schema.VARIANT_A, schema.VARIANT_B, schema.VARIANT_C,
                  schema.VARIANT_E])
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    rows = ts_diff[ts_diff["model_id"] == "F/partial"]
    assert set(rows["comparison"]) == {"B_minus_A", "C_minus_A", "E_minus_A"}

    # D - A is one of a_out's four readers and it was never attempted: D has
    # no run row at all, so it is pending rather than terminally
    # unavailable. a_out must survive for that future D - A.
    assert paths[schema.VARIANT_A].exists()


def test_a_model_with_only_a_yields_nothing_and_keeps_its_out_file(
    tmp_path, monkeypatch,
):
    out = tmp_path / "out"
    runs, paths = _partial_diff_runs_fixture(tmp_path, [schema.VARIANT_A])
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")
    assert ts_diff.empty or "F/partial" not in set(ts_diff["model_id"])
    assert paths[schema.VARIANT_A].exists()


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
    # `_fake_diff_out_files_raising_for_bad` fails for every comparison of
    # this model (it keys off the baseline argument, and every baseline of
    # this model is a "bad_*" path), so all five labels must be identifiable
    # in the warning output -- naming the model alone would also be
    # satisfied by a message that dropped which comparison failed.
    assert "B_minus_A" in captured.out
    assert "C_minus_A" in captured.out
    assert "D_minus_A" in captured.out
    assert "E_minus_A" in captured.out
    assert "D_minus_C" in captured.out


def _many_model_diff_fixture(tmp_path, n_models):
    """`n_models` models, each with an `.out` for all five variants.

    All five so every comparison resolves and every file is releasable --
    the batching, not the retention rule, is what these tests are about.
    """
    paths = {}
    rows = []
    for index in range(n_models):
        model_id = f"F/m{index}"
        for variant in schema.VARIANTS:
            path = tmp_path / f"batch_{index}_{variant}.out"
            path.write_bytes(b"not really a binary .out file")
            paths[(model_id, variant)] = path
            rows.append({"model_id": model_id, "family": "F",
                         "variant": variant, "status": schema.Status.OK,
                         "out_path": str(path)})
    return pd.DataFrame(rows), paths


def test_a_batch_boundary_flushes_without_losing_rows_or_stranding_files(
    tmp_path, monkeypatch,
):
    # The boundary path is unreachable with real fixtures: every diff fixture
    # holds two models against FLUSH_BATCH_SIZE = 50, so only the final
    # remainder flush ever ran. Patching the constant crosses a real boundary
    # (three models, batches of two) instead of building fifty models.
    out = tmp_path / "out"
    runs, paths = _many_model_diff_fixture(tmp_path, n_models=3)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(cli, "FLUSH_BATCH_SIZE", 2)
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    assert cli.stage_diff(out, abs_tol=1e-9) == 0

    ts_diff = store.read_table(out, "ts_diff")
    # Nothing lost across the boundary: every model's rows are persisted.
    assert set(ts_diff["model_id"]) == {"F/m0", "F/m1", "F/m2"}
    assert set(ts_diff["comparison"]) == {
        "B_minus_A", "C_minus_A", "D_minus_A", "E_minus_A", "D_minus_C",
    }
    # Two writes, not one: the boundary flush plus the remainder. A single
    # file would mean the boundary never fired and the test proved nothing.
    assert len(list((out / "ts_diff").rglob("*.parquet"))) == 2
    # And every `.out` behind an already-flushed batch is released.
    assert not any(path.exists() for path in paths.values())


def test_a_batch_boundary_never_deletes_an_out_file_before_its_rows_are_written(
    tmp_path, monkeypatch,
):
    # The ordering the whole batching scheme rests on: rows are flushed to
    # disk BEFORE any file behind them is unlinked, at a boundary exactly as
    # in the remainder. `.out` files are regenerable by re-running the model;
    # the diff rows derived from them are not.
    out = tmp_path / "out"
    runs, paths = _many_model_diff_fixture(tmp_path, n_models=3)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(cli, "FLUSH_BATCH_SIZE", 2)
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    observed = []
    original = cli.store.write_table

    def spy(frame, out_dir, name, partition_by=None):
        if name == "ts_diff":
            observed.append((set(frame["model_id"]),
                             [p for p in paths.values() if p.exists()]))
        return original(frame, out_dir, name, partition_by=partition_by)

    monkeypatch.setattr(cli.store, "write_table", spy)
    cli.stage_diff(out, abs_tol=1e-9)

    assert len(observed) == 2
    for models, alive in observed:
        # Every file this flush's rows were derived from is still on disk at
        # the moment of the write.
        for model_id in models:
            for variant in schema.VARIANTS:
                assert paths[(model_id, variant)] in alive


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

    assert len(executed) == 10
    assert set(executed["variant"]) == {"A", "B", "C", "D", "E"}
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


def test_the_scalars_value_column_is_numeric(corpus_root, tmp_path, fake_engine):
    # `scalars` is long-format: one `value` column for every metric. Melting a
    # string-valued key into it (a version banner, a timestamp, a routing-model
    # or option echo) mixes types in one Arrow column and the write fails
    # outright, taking the whole `run` stage with it.
    #
    # `NON_METRIC_SCALARS` is a DENYLIST, so this guarantee is not structural:
    # a future string-valued addition to `rptparse.SCALAR_KEYS` that nobody
    # remembers to list there re-breaks it. Pinned here rather than left as an
    # incidental property of today's key set.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    scalars = store.read_table(out, "scalars")

    assert not scalars.empty
    assert pd.api.types.is_numeric_dtype(scalars["value"])
    # ... and stated against the key sets themselves, so the assertion still
    # bites for a key no fixture report happens to populate.
    melted = set(rptparse.SCALAR_KEYS) - cli.NON_METRIC_SCALARS
    assert set(scalars["metric"]) == melted
    assert not (melted & cli.NON_METRIC_SCALARS)


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
    # `--lang` defaults to `es`; assert via the same mapping stage_report
    # draws from, not a hardcoded copy of the translation.
    assert cli.STAGE_REPORT_STRINGS["es"]["empty_store"] in capsys.readouterr().out


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


# ---------------------------------------------------------------------------
# Engine build identity
# ---------------------------------------------------------------------------

#: An engine stand-in that answers `--version` with a FIXED banner regardless
#: of its own contents, so two of them are indistinguishable by version
#: string and distinguishable only by their bytes -- exactly the situation
#: `engine_version` cannot tell apart and `engine_build_id` must.
_ENGINE_TEMPLATE = """
    import sys, pathlib
    BUILD_MARKER = {marker!r}
    if "--version" in sys.argv:
        print("OpenSWMM 6.0")
        raise SystemExit(0)
    pathlib.Path(sys.argv[2]).write_text({report!r}, encoding="latin-1")
    pathlib.Path(sys.argv[3]).write_bytes(b"out")
"""

MISSING_ENGINE = "swmmbench-no-such-executable"


def _marked_engine(tmp_path, name, marker, report=FAKE_REPORT):
    script = tmp_path / name
    script.write_text(
        textwrap.dedent(_ENGINE_TEMPLATE).format(marker=marker, report=report),
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def test_two_engines_with_one_version_banner_have_different_build_ids(tmp_path):
    # The defect: `--version` was the engine's identity in the resume key, so
    # two executables containing completely different Anderson, continuity or
    # slot implementations were one key as long as they printed one string.
    build_a = _marked_engine(tmp_path, "engine_a.py", "build-a")
    build_b = _marked_engine(tmp_path, "engine_b.py", "build-b")

    assert cli._engine_version(build_a) == cli._engine_version(build_b)

    id_a = cli._engine_identity(build_a)["engine_build_id"]
    id_b = cli._engine_identity(build_b)["engine_build_id"]

    assert id_a != id_b
    # ... and it is a content hash, not a restatement of the argv.
    assert id_a.startswith(cli.ENGINE_BUILD_HASHED_PREFIX)
    assert id_b.startswith(cli.ENGINE_BUILD_HASHED_PREFIX)


def test_a_run_under_one_build_is_not_completed_under_another(corpus_root, tmp_path):
    out = tmp_path / "out"
    build_a = _marked_engine(tmp_path, "engine_a.py", "build-a")
    build_b = _marked_engine(tmp_path, "engine_b.py", "build-b")
    cli.stage_inventory(corpus_root, out)

    cli.stage_run(corpus_root, out, build_a, timeout_s=30.0, jobs=1, limit=None)
    cli.stage_run(corpus_root, out, build_b, timeout_s=30.0, jobs=1, limit=None)

    runs = store.read_table(out, "runs")
    executed = runs[runs["variant"].isin(schema.VARIANTS)]

    # Both builds ran everything: neither was skipped as "already done".
    assert len(executed) == 20
    assert executed["engine_build_id"].nunique() == 2
    assert executed["engine_version"].nunique() == 1  # the string agrees...
    # ... and the anchors are NOT re-read: a REF case depends on the corpus,
    # not on any build of ours, so a second engine must not duplicate them.
    assert len(runs[runs["variant"] == schema.VARIANT_REF]) == 2


def test_an_engine_that_is_not_a_readable_file_degrades_without_raising(tmp_path):
    identity = cli._engine_identity([MISSING_ENGINE])

    # Clearly marked as non-cryptographic rather than dressed up as a hash:
    # anything reachable under this name would collide with anything else.
    assert identity["engine_build_id"].startswith(cli.ENGINE_BUILD_UNHASHED_PREFIX)
    assert not identity["engine_build_id"].startswith(cli.ENGINE_BUILD_HASHED_PREFIX)
    assert identity["engine_version"]


def test_a_sweep_with_an_unrunnable_engine_records_crashes_rather_than_raising(
    corpus_root, tmp_path,
):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)

    code = cli.stage_run(corpus_root, out, [MISSING_ENGINE],
                         timeout_s=30.0, jobs=1, limit=None)

    assert code == 0  # the corpus is still clean; the engine is the problem
    runs = store.read_table(out, "runs")
    executed = runs[runs["variant"].isin(schema.VARIANTS)]
    assert set(executed["status"]) == {schema.Status.CRASH}
    assert executed["case_id"].notna().all()


def test_the_mixed_build_guard_reads_the_build_hash_not_the_version_string(
    tmp_path,
):
    # The guard exists to catch exactly the mix a version string cannot see.
    runs = pd.DataFrame([
        {"model_id": "F/m", "family": "F", "variant": variant,
         "engine_version": "OpenSWMM 6.0", "engine_build_id": build}
        for variant, build in ((schema.VARIANT_A, "sha256:aaa"),
                               (schema.VARIANT_C, "sha256:bbb"))
    ])

    assert cli.executed_engine_versions(runs) == ["sha256:aaa", "sha256:bbb"]


# ---------------------------------------------------------------------------
# The corpus dependency identity
# ---------------------------------------------------------------------------

requires_git = pytest.mark.skipif(shutil.which("git") is None,
                                  reason="git is not installed")


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args],
                   check=True, capture_output=True, text=True)


@pytest.fixture
def git_corpus(corpus_root):
    """`corpus_root` as a git repository, with an external data file.

    The data file is the point: it is what a deck references by relative
    path and what a deck-only hash cannot see change.
    """
    _git(corpus_root, "init", "-q")
    _git(corpus_root, "config", "user.email", "swmmbench@example.invalid")
    _git(corpus_root, "config", "user.name", "swmmbench")
    _git(corpus_root, "config", "commit.gpgsign", "false")
    (corpus_root / "EPA" / "DataFiles").mkdir()
    (corpus_root / "EPA" / "DataFiles" / "rainfall.dat").write_text(
        "1\n", encoding="latin-1")
    _git(corpus_root, "add", "-A")
    _git(corpus_root, "commit", "-qm", "one")
    return corpus_root


@requires_git
def test_a_new_corpus_commit_invalidates_the_resume_key(
    git_corpus, tmp_path, fake_engine,
):
    # The decks are byte-identical before and after; only an external data
    # file changed. Under the old deck-only key every case read as "already
    # done" and the sweep republished stale numbers as current.
    out = tmp_path / "out"
    cli.stage_inventory(git_corpus, out)
    cli.stage_run(git_corpus, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)
    before = len(store.read_table(out, "runs"))

    (git_corpus / "EPA" / "DataFiles" / "rainfall.dat").write_text(
        "2\n", encoding="latin-1")
    _git(git_corpus, "commit", "-qam", "two")
    cli.stage_inventory(git_corpus, out)
    cli.stage_run(git_corpus, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    runs = store.read_table(out, "runs")

    assert len(runs) == 2 * before
    assert runs["corpus_commit"].nunique() == 2
    assert runs["inp_sha256"].nunique() == 1  # the decks never moved
    assert runs["case_id"].nunique() == len(runs)


@requires_git
def test_an_unchanged_git_corpus_still_resumes(git_corpus, tmp_path, fake_engine):
    # The other half: over-invalidation is the safe direction, but it must
    # not fire on a corpus that did not move.
    out = tmp_path / "out"
    cli.stage_inventory(git_corpus, out)
    cli.stage_run(git_corpus, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)
    before = len(store.read_table(out, "runs"))

    cli.stage_run(git_corpus, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    assert len(store.read_table(out, "runs")) == before


def test_a_non_git_corpus_sweeps_with_the_dependency_marked_unpinned(
    corpus_root, tmp_path, fake_engine,
):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)

    assert cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0,
                         jobs=1, limit=None) == 0

    runs = store.read_table(out, "runs")
    assert set(runs["corpus_commit"]) == {corpus.UNPINNED_CORPUS}


def test_a_rewritten_deck_re_runs_even_when_the_corpus_is_unpinned(
    corpus_root, tmp_path, fake_engine,
):
    # `UNPINNED_CORPUS` is a CONSTANT, so with it as the whole dependency
    # identity nothing about the corpus could ever invalidate a resumed
    # sweep: an operator could rewrite a deck, re-run `inventory` and `run`,
    # and be told every case was already done while `runs` kept the stale
    # `inp_sha256`. The deck's own hash is folded in behind the sentinel.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1,
                  limit=None)

    (corpus_root / "EPA" / "m1.inp").write_text(
        DECK + "IGNORE_RAINFALL       YES\n", encoding="latin-1")
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1,
                  limit=None)

    runs = store.read_table(out, "runs")
    edited = runs[runs["model_id"] == "EPA/m1"]
    untouched = runs[runs["model_id"] == "LID/m2"]

    # The edited model re-ran all five variants and its anchor; the model
    # nobody touched was still recognised as done.
    assert len(edited) == 12
    assert edited["case_id"].nunique() == 12
    assert edited["inp_sha256"].nunique() == 2
    assert len(untouched) == 6
    # And the degraded identity still cannot masquerade as a pin.
    assert set(runs["corpus_commit"]) == {corpus.UNPINNED_CORPUS}


def test_an_unchanged_unpinned_corpus_still_resumes(
    corpus_root, tmp_path, fake_engine,
):
    # The other half: folding the deck hash in must not make every sweep
    # re-run everything.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1,
                  limit=None)
    before = len(store.read_table(out, "runs"))

    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1,
                  limit=None)

    assert len(store.read_table(out, "runs")) == before


def test_a_degraded_dependency_identity_never_equals_a_commit():
    pinned = corpus.dependency_id("git:1111", "abcd")
    unpinned = corpus.dependency_id(corpus.UNPINNED_CORPUS, "abcd")

    # A commit already covers the deck and every file it references, so
    # nothing is folded in behind it.
    assert pinned == "git:1111"
    assert unpinned.startswith(corpus.UNPINNED_CORPUS)
    assert not unpinned.startswith(corpus.GIT_COMMIT_PREFIX)
    assert unpinned != corpus.dependency_id(corpus.UNPINNED_CORPUS, "efgh")


def test_the_run_stage_itself_warns_that_a_sweep_is_unpinned(
    corpus_root, tmp_path, fake_engine, capsys,
):
    # `inventory` and `run` are separate invocations in the documented
    # workflow, and `run` prefers the identity already recorded on `models`,
    # so it never called `commit_sha` and printed nothing at all -- leaving
    # the operator kicking off a multi-hour sweep the one person not told.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    capsys.readouterr()  # discard whatever `inventory` said

    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1,
                  limit=None)

    printed = capsys.readouterr().out
    assert cli.UNPINNED_SWEEP_WARNING in printed
    # It says what is degraded and what follows from it, not merely that
    # something is.
    assert corpus.UNPINNED_CORPUS in printed
    assert "republish" in printed


@requires_git
def test_a_pinned_sweep_does_not_warn(git_corpus, tmp_path, fake_engine, capsys):
    out = tmp_path / "out"
    cli.stage_inventory(git_corpus, out)
    capsys.readouterr()

    cli.stage_run(git_corpus, out, fake_engine, timeout_s=30.0, jobs=1,
                  limit=None)

    assert cli.UNPINNED_SWEEP_WARNING not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The variant's option set is part of the case identity
# ---------------------------------------------------------------------------


def test_editing_a_variants_option_set_re_runs_that_variant_alone(
    corpus_root, tmp_path, fake_engine, monkeypatch,
):
    # `variants.OPTIONS` has been edited several times in this project's
    # life. Keyed on the letter alone, a `B` run recorded before an edit and
    # a `B` run after it are one case, so the second reads as "already done"
    # and two materially different units of work share an id.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1,
                  limit=None)
    before = len(store.read_table(out, "runs"))

    monkeypatch.setitem(variants.OPTIONS, schema.VARIANT_B,
                        {**variants.OPTIONS[schema.VARIANT_B],
                         "DPS_ALPHA": "9.0"})
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1,
                  limit=None)

    runs = store.read_table(out, "runs")

    # Exactly B re-ran, once per model: the other variants' option sets did
    # not move, so over-invalidation is not the fix either.
    assert len(runs) == before + 2
    assert runs[runs["variant"] == schema.VARIANT_B]["case_id"].nunique() == 4
    for variant in (schema.VARIANT_A, schema.VARIANT_C, schema.VARIANT_D,
                    schema.VARIANT_E, schema.VARIANT_REF):
        assert runs[runs["variant"] == variant]["case_id"].nunique() == 2, variant


def test_an_option_set_fingerprint_tracks_the_values_not_their_order():
    base = variants.options_id(schema.VARIANT_A)

    assert base == variants.options_id(schema.VARIANT_A)
    assert base != variants.options_id(schema.VARIANT_B)
    assert base.startswith(variants.OPTIONS_ID_PREFIX)
    # REF is parsed, never run with options, so nothing about `OPTIONS`
    # should ever re-read the anchors.
    assert variants.options_id(schema.VARIANT_REF) == variants.options_id("nope")


def test_reordering_an_option_set_leaves_every_existing_case_valid(monkeypatch):
    before = variants.options_id(schema.VARIANT_A)
    reordered = dict(reversed(list(variants.OPTIONS[schema.VARIANT_A].items())))
    monkeypatch.setitem(variants.OPTIONS, schema.VARIANT_A, reordered)

    assert variants.options_id(schema.VARIANT_A) == before

    monkeypatch.setitem(variants.OPTIONS, schema.VARIANT_A,
                        {**reordered, "DPS_ALPHA": "9.0"})
    assert variants.options_id(schema.VARIANT_A) != before


# ---------------------------------------------------------------------------
# `case_id` travels to every derived table
# ---------------------------------------------------------------------------


def _case_keys(table):
    return set(zip(table["model_id"], table["variant"], table["case_id"]))


def test_case_id_is_identical_across_runs_scalars_and_elements(
    corpus_with_elements, tmp_path, engine_with_elements,
):
    # `scalars` and `elements` used to be identified by model/family/variant/
    # engine_version alone, so a changed model left old and new records
    # coexisting with nothing to tell them apart and `aggfunc="first"` free
    # to pick the stale one.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_with_elements, out)
    cli.stage_run(corpus_with_elements, out, engine_with_elements,
                  timeout_s=30.0, jobs=1, limit=None)

    runs = store.read_table(out, "runs")
    scalars = store.read_table(out, "scalars")
    elements = store.read_table(out, "elements")

    for table in (runs, scalars, elements):
        assert "case_id" in table.columns
        assert table["case_id"].notna().all()

    assert _case_keys(scalars) == _case_keys(runs)
    assert _case_keys(elements) == _case_keys(runs)


def test_case_id_differs_when_the_engine_build_differs(corpus_root, tmp_path):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    cli.stage_run(corpus_root, out, _marked_engine(tmp_path, "engine_a.py", "a"),
                  timeout_s=30.0, jobs=1, limit=None)
    cli.stage_run(corpus_root, out, _marked_engine(tmp_path, "engine_b.py", "b"),
                  timeout_s=30.0, jobs=1, limit=None)

    runs = store.read_table(out, "runs")
    executed = runs[runs["variant"].isin(schema.VARIANTS)]
    per_case = executed.groupby(["model_id", "variant"])["case_id"].nunique()

    assert set(per_case) == {2}


def test_delta_rows_carry_the_case_id_of_every_source_run(
    corpus_with_elements, tmp_path, engine_with_elements,
):
    # A published number has to be traceable to the exact executable and
    # corpus state behind it; `deltas` and `element_deltas` are recomputed
    # and replaced every report, so they cannot rely on `runs` staying put.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_with_elements, out)
    cli.stage_run(corpus_with_elements, out, engine_with_elements,
                  timeout_s=30.0, jobs=1, limit=None)
    cli.stage_report(out, lang="en")

    runs = store.read_table(out, "runs")
    known = set(runs["case_id"])

    for name in ("deltas", "element_deltas"):
        table = store.read_table(out, name)
        assert not table.empty, name
        for column in report.CASE_ID_COLUMNS:
            assert column in table.columns, (name, column)
            assert set(table[column].dropna()) <= known, (name, column)
        # Each column names its OWN variant's run, not just some run.
        for variant in ("a", "e", "ref"):
            expected = set(runs.loc[runs["variant"] == variant.upper(), "case_id"])
            assert set(table[f"case_id_{variant}"].dropna()) == expected, name


# ---------------------------------------------------------------------------
# Batched writes during `run`
# ---------------------------------------------------------------------------


class _SimulatedMachineFailure(Exception):
    """Something `_run_one` cannot itself raise: a lost machine mid-sweep."""


def test_rows_reach_disk_before_the_sweep_finishes(
    corpus_root, tmp_path, fake_engine, monkeypatch,
):
    # The whole sweep used to be held in memory and written once, so a
    # machine failure near the end of ~8200 simulations lost everything
    # rather than the tail. Two models x five variants against a batch of
    # two crosses four real boundaries.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    monkeypatch.setattr(cli, "FLUSH_BATCH_SIZE", 2)

    original = cli._run_one
    durable = []

    def spy(job):
        durable.append(len(store.read_table(out, "runs")))
        return original(job)

    monkeypatch.setattr(cli, "_run_one", spy)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    # By the last of the ten simulations, the first eight are already durable.
    assert durable[0] == 0
    assert durable[-1] == 8
    assert len(store.read_table(out, "runs")) == 12  # 10 + two REF rows


def test_an_interruption_after_a_flush_leaves_the_flushed_rows_resumable(
    corpus_root, tmp_path, fake_engine, monkeypatch,
):
    out = tmp_path / "out"
    cli.stage_inventory(corpus_root, out)
    monkeypatch.setattr(cli, "FLUSH_BATCH_SIZE", 2)

    original = cli._run_one
    calls = {"n": 0}

    def failing(job):
        calls["n"] += 1
        if calls["n"] == 5:
            raise _SimulatedMachineFailure("power cut")
        return original(job)

    monkeypatch.setattr(cli, "_run_one", failing)
    with pytest.raises(_SimulatedMachineFailure):
        cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0,
                      jobs=1, limit=None)

    # Two batches had flushed; only the unflushed fifth simulation is lost.
    assert len(store.read_table(out, "runs")) == 4

    monkeypatch.setattr(cli, "_run_one", original)
    cli.stage_run(corpus_root, out, fake_engine, timeout_s=30.0, jobs=1, limit=None)

    runs = store.read_table(out, "runs")
    assert len(runs) == 12
    # Resumed, not redone: the survivors were recognised by their case_id.
    assert runs["case_id"].nunique() == 12
    assert set(store.read_table(out, "scalars")["case_id"]) == set(runs["case_id"])


# ---------------------------------------------------------------------------
# `ts_diff` carries the coverage evidence, in a stable schema
# ---------------------------------------------------------------------------


def test_the_diff_stage_persists_every_coverage_field(tmp_path, monkeypatch):
    # The evidence is only useful if it survives the write: a `ts_diff` that
    # drops `coverage_fraction` cannot tell a full comparison from one over
    # half a run, which is the whole point of collecting it.
    out = tmp_path / "out"
    runs, _bad, _good = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)

    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")

    assert set(outdiff.COVERAGE_FIELDS) <= set(ts_diff.columns)
    assert ts_diff["coverage_fraction"].notna().all()
    assert report.coverage_anomalies(ts_diff).empty


def _fake_diff_row_with_coverage(**overrides):
    def fake(a_out, other_out, abs_tol):
        return [{**FAKE_DIFF_ROW, **overrides}]
    return fake


def test_a_truncated_comparison_survives_the_store_as_an_anomaly(
    tmp_path, monkeypatch,
):
    out = tmp_path / "out"
    runs, _bad, _good = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_row_with_coverage(
                            n_periods_right=12, n_common=12,
                            coverage_fraction=0.5, end_time_match=False,
                            time_grid_match=False))

    cli.stage_diff(out, abs_tol=1e-9)

    anomalies = report.coverage_anomalies(store.read_table(out, "ts_diff"))

    assert set(anomalies["comparison"]) == {
        "B_minus_A", "C_minus_A", "D_minus_A", "E_minus_A", "D_minus_C",
    }
    assert (anomalies["min_coverage"] == 0.5).all()


def test_a_batch_with_no_coverage_at_all_still_writes_a_matching_fragment(
    tmp_path, monkeypatch,
):
    # A row set carrying none of the coverage fields (or carrying them all
    # null) must not infer Arrow's `null` type and stop matching the
    # fragments around it -- the same hazard `_run_frame` conforms `runs`
    # against.
    out = tmp_path / "out"
    runs, _bad, _good = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])

    bare = {k: v for k, v in FAKE_DIFF_ROW.items()
            if k not in outdiff.COVERAGE_FIELDS}
    monkeypatch.setattr(outdiff, "diff_out_files",
                        lambda a, b, abs_tol: [dict(bare)])
    cli.stage_diff(out, abs_tol=1e-9)

    second = tmp_path / "second"
    second.mkdir()
    runs2, _bad2, _good2 = _diff_runs_fixture(second)
    store.write_table(runs2, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)
    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")

    assert set(outdiff.COVERAGE_FIELDS) <= set(ts_diff.columns)
    # The bare batch is unknown, not incomplete: absence is no evidence.
    assert ts_diff["coverage_fraction"].isna().any()
    assert ts_diff["coverage_fraction"].notna().any()
    assert report.coverage_anomalies(ts_diff).empty


def test_an_all_agreeing_batch_does_not_poison_a_later_divergent_one(
    tmp_path, monkeypatch,
):
    # A flush in which every comparison agreed within tolerance has no first
    # divergence at all, so `first_div_period` and `first_div_time` are null
    # across the whole batch -- entirely ordinary on a corpus of small decks,
    # and certain for a single-model batch. Unconformed, that fragment infers
    # Arrow's `null` type, and a later fragment carrying an actual divergence
    # writes `int64` and `timestamp`; the DATASET then stops reading, with
    # `Unsupported cast from int64 to null`, taking the whole report stage
    # (which reads `ts_diff` unconditionally) down with it.
    out = tmp_path / "out"
    runs, _bad, _good = _diff_runs_fixture(tmp_path)
    store.write_table(runs, out, "runs", partition_by=["family"])

    # The AGREEING batch is written FIRST, deliberately: that is the order
    # that used to fail. Written the other way round the bug is invisible,
    # so a test that happened to do so would pass against it.
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_out_files_always_succeeding)
    cli.stage_diff(out, abs_tol=1e-9)

    first_batch = store.read_table(out, "ts_diff")
    assert not first_batch.empty
    assert first_batch["first_div_period"].isna().all()
    assert first_batch["first_div_time"].isna().all()

    second = tmp_path / "second"
    second.mkdir()
    runs2, _bad2, _good2 = _diff_runs_fixture(second)
    store.write_table(runs2, out, "runs", partition_by=["family"])
    monkeypatch.setattr(outdiff, "diff_out_files",
                        _fake_diff_row_with_coverage(
                            max_abs=2.5, max_rel=0.5, rmse=1.25,
                            first_div_period=3,
                            first_div_time=datetime(2002, 1, 1, 0, 30)))
    cli.stage_diff(out, abs_tol=1e-9)

    ts_diff = store.read_table(out, "ts_diff")  # must not raise

    # Both fragments read back as one table, each keeping its own values.
    assert ts_diff["first_div_period"].isna().any()
    assert set(ts_diff["first_div_period"].dropna()) == {3}
    assert ts_diff["first_div_time"].notna().any()
    assert set(ts_diff["max_abs"].dropna()) == {1.0, 2.5}


# ---------------------------------------------------------------------------
# The conformed column lists cover what the producers emit
# ---------------------------------------------------------------------------


def test_run_columns_covers_every_key_the_run_producers_emit(
    corpus_with_elements, tmp_path, engine_with_elements, monkeypatch,
):
    # `_run_frame` builds `pd.DataFrame(rows, columns=RUN_COLUMNS)`, which
    # DROPS any key the list does not name -- silently, with no error and no
    # column. So a field added to `_run_one` or `_reference_rows` later would
    # simply never reach the store, and nothing would say so. Pinned here
    # instead of noticed in an analysis six hours later.
    out = tmp_path / "out"
    cli.stage_inventory(corpus_with_elements, out)

    emitted_runs: set = set()
    emitted_elements: set = set()
    original = cli._flush_run_batch

    def spy(out_dir, run_rows, element_rows):
        for row in run_rows:
            emitted_runs.update(row)
        for row in element_rows:
            emitted_elements.update(row)
        return original(out_dir, run_rows, element_rows)

    monkeypatch.setattr(cli, "_flush_run_batch", spy)
    cli.stage_run(corpus_with_elements, out, engine_with_elements,
                  timeout_s=30.0, jobs=1, limit=None)

    # Executed rows, reference rows and element rows all pass through the
    # same flush, so one sweep covers every producer the sweep uses.
    assert emitted_runs
    assert emitted_runs <= set(cli.RUN_COLUMNS), (
        emitted_runs - set(cli.RUN_COLUMNS))
    assert emitted_elements
    assert emitted_elements <= set(cli.ELEMENT_COLUMNS), (
        emitted_elements - set(cli.ELEMENT_COLUMNS))


def test_run_columns_covers_the_lost_worker_row_too(corpus_root, tmp_path):
    # `_crash_result` is reached only when the POOL loses a process, which no
    # ordinary sweep exercises -- so its keys would drift out of RUN_COLUMNS
    # with nothing to notice, and the row that exists precisely to keep a
    # lost job from being retried forever would land half-empty.
    job = {
        "record": {"model_id": "EPA/m1", "family": "EPA",
                   "inp_sha256": "abcd"},
        "variant": schema.VARIANT_A, "case_id": "cafe",
        "engine_version": "OpenSWMM 6.0", "engine_build_id": "sha256:aaa",
        "engine_build_info": None, "corpus_commit": "git:1111",
        "timeout_s": 30.0,
    }

    row = cli._crash_result(job, RuntimeError("pool died"))["run"]

    assert set(row) <= set(cli.RUN_COLUMNS), set(row) - set(cli.RUN_COLUMNS)
