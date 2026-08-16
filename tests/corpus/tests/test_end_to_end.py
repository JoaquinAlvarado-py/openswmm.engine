import os
import shutil
from pathlib import Path

import pytest

from swmmbench import cli, schema, store

ENGINE_ENV = "SWMMBENCH_ENGINE"

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE1 = REPO_ROOT / "tests" / "regression" / "data" / "Example1.inp"


pytestmark = pytest.mark.skipif(
    not os.environ.get(ENGINE_ENV),
    reason=f"set {ENGINE_ENV} to a built openswmm executable to run this",
)


@pytest.fixture
def mini_corpus(tmp_path):
    """A one-model corpus built from the repository's own regression deck."""
    root = tmp_path / "corpus"
    (root / "EPA").mkdir(parents=True)
    shutil.copy(EXAMPLE1, root / "EPA" / "Example1.inp")
    return root


def test_full_pipeline_produces_all_six_variants(mini_corpus, tmp_path):
    out = tmp_path / "results"
    engine = [os.environ[ENGINE_ENV]]

    cli.stage_inventory(mini_corpus, out)
    cli.stage_run(mini_corpus, out, engine, timeout_s=600.0, jobs=1, limit=None)
    cli.stage_report(out)

    runs = store.read_table(out, "runs")

    assert set(runs["variant"]) == {"A", "B", "C", "D", "E", "REF"}
    executed = runs[runs["variant"].isin(schema.VARIANTS)]
    assert set(executed["status"]) == {schema.Status.OK}
    assert executed["avg_iterations_per_step"].notna().all()
    assert set(executed["iteration_metric_kind"]) == {schema.ITER_PICARD}


def test_the_corpus_is_left_untouched(mini_corpus, tmp_path):
    before = sorted(p.name for p in (mini_corpus / "EPA").iterdir())

    cli.stage_inventory(mini_corpus, tmp_path / "results")
    cli.stage_run(mini_corpus, tmp_path / "results", [os.environ[ENGINE_ENV]],
                  timeout_s=600.0, jobs=1, limit=None)

    assert sorted(p.name for p in (mini_corpus / "EPA").iterdir()) == before
    clean, leftovers = cli.corpus_is_clean(mini_corpus)
    assert clean, leftovers
