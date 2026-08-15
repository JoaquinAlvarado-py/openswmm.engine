from pathlib import Path

from swmmbench import corpus


def _write(path: Path, text: str = "[OPTIONS]\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="latin-1")


def test_pairs_input_with_its_sibling_report(tmp_path):
    _write(tmp_path / "EPA" / "m1.inp")
    _write(tmp_path / "EPA" / "m1.rpt")

    records = corpus.discover(tmp_path)

    assert len(records) == 1
    assert records[0].family == "EPA"
    assert records[0].model_id == "EPA/m1"
    assert records[0].ref_path == "EPA/m1.rpt"


def test_decoy_reports_do_not_pair(tmp_path):
    # The real Special/ layout: three reports, one input.
    _write(tmp_path / "Special" / "many_Isolated_Nodes.inp")
    _write(tmp_path / "Special" / "many_Isolated_Nodes.rpt")
    _write(tmp_path / "Special" / "many_Isolated_Nodes.inp.rpt")
    _write(tmp_path / "Special" / "many_Isolated_Nodes_epa.rpt")

    (record,) = corpus.discover(tmp_path)

    assert record.ref_path == "Special/many_Isolated_Nodes.rpt"


def test_report_in_another_directory_does_not_pair(tmp_path):
    _write(tmp_path / "EPA" / "m1.inp")
    _write(tmp_path / "Hydrology" / "m1.rpt")

    (record,) = corpus.discover(tmp_path)

    assert record.ref_path is None


def test_v12_and_v13_are_matched_by_basename_as_historical_anchors(tmp_path):
    _write(tmp_path / "EPA" / "Example1a.inp")
    _write(tmp_path / "v12" / "Example1a.rpt")
    _write(tmp_path / "v13" / "Example1a.rpt")

    (record,) = corpus.discover(tmp_path)

    assert record.ref_path is None
    assert record.ref_v12_path == "v12/Example1a.rpt"
    assert record.ref_v13_path == "v13/Example1a.rpt"


def test_version_directories_contribute_no_models(tmp_path):
    _write(tmp_path / "v12" / "orphan.rpt")

    assert corpus.discover(tmp_path) == []


def test_hash_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.inp"
    b = tmp_path / "b.inp"
    _write(a, "[OPTIONS]\n")
    _write(b, "[OPTIONS]\nFLOW_UNITS CFS\n")

    assert corpus.sha256_of(a) == corpus.sha256_of(a)
    assert corpus.sha256_of(a) != corpus.sha256_of(b)
