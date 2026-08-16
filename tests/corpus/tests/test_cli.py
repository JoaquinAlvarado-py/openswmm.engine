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
        return [{"element_type": "NODE", "element_id": "n1",
                 "attribute": "INVERT_DEPTH", "max_abs": 1.0, "max_rel": 1.0,
                 "rmse": 1.0, "first_div_period": None, "first_div_time": None,
                 "n_periods": 1}]
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
    return [{"element_type": "NODE", "element_id": "n1",
             "attribute": "INVERT_DEPTH", "max_abs": 1.0, "max_rel": 1.0,
             "rmse": 1.0, "first_div_period": None, "first_div_time": None,
             "n_periods": 1}]


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
    return [{"element_type": "NODE", "element_id": "n1",
             "attribute": "INVERT_DEPTH", "max_abs": 1.0, "max_rel": 1.0,
             "rmse": 1.0, "first_div_period": None, "first_div_time": None,
             "n_periods": 1}]


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
