import shutil
import subprocess
from pathlib import Path

import pytest

from swmmbench import corpus, variants

requires_git = pytest.mark.skipif(shutil.which("git") is None,
                                  reason="git is not installed")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args],
                   check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "swmmbench@example.invalid")
    _git(root, "config", "user.name", "swmmbench")
    _git(root, "config", "commit.gpgsign", "false")


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


def test_harness_temporary_decks_are_not_inventoried(tmp_path):
    """A deck the harness wrote is its own leftover, not a corpus model.

    A killed sweep leaves variant decks behind; inventorying them would run
    the corpus against decks we wrote and grow the model count on every
    interrupted sweep.
    """
    _write(tmp_path / "EPA" / "m1.inp")
    _write(tmp_path / "EPA" / f"{variants.TEMP_PREFIX}A_m1.inp")
    _write(tmp_path / "EPA" / f"{variants.TEMP_PREFIX}B_m1.inp")

    records = corpus.discover(tmp_path)

    assert [r.model_id for r in records] == ["EPA/m1"]


def test_hash_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.inp"
    b = tmp_path / "b.inp"
    _write(a, "[OPTIONS]\n")
    _write(b, "[OPTIONS]\nFLOW_UNITS CFS\n")

    assert corpus.sha256_of(a) == corpus.sha256_of(a)
    assert corpus.sha256_of(a) != corpus.sha256_of(b)


# ---------------------------------------------------------------------------
# The corpus dependency identity
# ---------------------------------------------------------------------------


def test_a_corpus_that_is_not_a_git_repository_degrades_with_a_warning(
    tmp_path, capsys,
):
    # Degrading beats failing: a sweep of 8200 simulations must not be
    # refused because the corpus was unpacked from a tarball. But it must
    # SAY so -- a silent fallback would let an unpinned store be read as a
    # reproducible one.
    corpus._warned_roots.discard(str(tmp_path))

    assert corpus.commit_sha(tmp_path) == corpus.UNPINNED_CORPUS
    assert "not a git repository" in capsys.readouterr().out


def test_the_unpinned_warning_is_emitted_once_per_root(tmp_path, capsys):
    # Once per corpus, not once per model: `discover` is called per stage and
    # a per-call warning would bury every other message in the sweep.
    corpus._warned_roots.discard(str(tmp_path))
    corpus.commit_sha(tmp_path)
    capsys.readouterr()

    corpus.commit_sha(tmp_path)

    assert capsys.readouterr().out == ""


def test_an_unpinned_corpus_is_never_mistakable_for_a_commit(tmp_path):
    assert not corpus.UNPINNED_CORPUS.startswith(corpus.GIT_COMMIT_PREFIX)


@requires_git
def test_a_git_corpus_records_its_commit(tmp_path):
    _write(tmp_path / "EPA" / "m1.inp")
    _init_repo(tmp_path)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "one")

    sha = corpus.commit_sha(tmp_path)

    assert sha.startswith(corpus.GIT_COMMIT_PREFIX)
    assert len(sha) > len(corpus.GIT_COMMIT_PREFIX)


@requires_git
def test_the_commit_changes_when_an_external_data_file_changes(tmp_path):
    # The defect this exists for: the deck is untouched and its `inp_sha256`
    # is unchanged, but a file it references by relative path is not. The
    # dependency identity has to move even though the deck did not.
    _write(tmp_path / "EPA" / "m1.inp")
    _write(tmp_path / "EPA" / "DataFiles" / "rainfall.dat", "1\n")
    _init_repo(tmp_path)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "one")
    before = corpus.commit_sha(tmp_path)
    deck_hash = corpus.sha256_of(tmp_path / "EPA" / "m1.inp")

    _write(tmp_path / "EPA" / "DataFiles" / "rainfall.dat", "2\n")
    _git(tmp_path, "commit", "-qam", "two")

    assert corpus.sha256_of(tmp_path / "EPA" / "m1.inp") == deck_hash
    assert corpus.commit_sha(tmp_path) != before


@requires_git
def test_every_record_is_stamped_with_the_corpus_commit(tmp_path):
    _write(tmp_path / "EPA" / "m1.inp")
    _write(tmp_path / "LID" / "m2.inp")
    _init_repo(tmp_path)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "one")

    records = corpus.discover(tmp_path)

    assert {r.corpus_commit for r in records} == {corpus.commit_sha(tmp_path)}
