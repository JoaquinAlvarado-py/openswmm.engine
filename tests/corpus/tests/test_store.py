import pandas as pd

from swmmbench import store


def _frame():
    return pd.DataFrame([
        {"model_id": "EPA/m1", "family": "EPA", "variant": "A",
         "engine_version": "6.0.0a3", "inp_sha256": "aaa", "wall_ms": 12.0},
        {"model_id": "LID/m2", "family": "LID", "variant": "B",
         "engine_version": "6.0.0a3", "inp_sha256": "bbb", "wall_ms": 34.0},
    ])


def test_round_trips_a_table(tmp_path):
    store.write_table(_frame(), tmp_path, "runs")

    result = store.read_table(tmp_path, "runs")

    assert sorted(result["model_id"]) == ["EPA/m1", "LID/m2"]
    assert result.loc[result.model_id == "EPA/m1", "wall_ms"].iloc[0] == 12.0


def test_partitioning_by_family_keeps_the_column_readable(tmp_path):
    store.write_table(_frame(), tmp_path, "runs", partition_by=["family"])

    result = store.read_table(tmp_path, "runs")

    assert set(result["family"]) == {"EPA", "LID"}


def test_reading_an_absent_table_yields_an_empty_frame(tmp_path):
    result = store.read_table(tmp_path, "nothing")

    assert result.empty


def test_completed_keys_identify_work_already_done(tmp_path):
    store.write_table(_frame(), tmp_path, "runs")

    keys = store.completed_keys(
        tmp_path, "runs", ["model_id", "variant", "engine_version", "inp_sha256"]
    )

    assert ("EPA/m1", "A", "6.0.0a3", "aaa") in keys
    assert ("EPA/m1", "B", "6.0.0a3", "aaa") not in keys


def test_a_changed_input_hash_is_not_treated_as_completed(tmp_path):
    store.write_table(_frame(), tmp_path, "runs")

    keys = store.completed_keys(
        tmp_path, "runs", ["model_id", "variant", "engine_version", "inp_sha256"]
    )

    assert ("EPA/m1", "A", "6.0.0a3", "MODIFIED") not in keys


def test_completed_keys_on_an_absent_table_is_empty(tmp_path):
    assert store.completed_keys(tmp_path, "runs", ["model_id"]) == set()


def test_a_second_write_appends_instead_of_overwriting(tmp_path):
    # pyarrow's default basename template is `part-{i}`, so a naive second
    # write reuses `part-0.parquet` and silently destroys the first sweep.
    store.write_table(_frame(), tmp_path, "runs")
    more = pd.DataFrame([
        {"model_id": "WQ/m3", "family": "WQ", "variant": "A",
         "engine_version": "6.0.0a3", "inp_sha256": "ccc", "wall_ms": 56.0},
    ])
    store.write_table(more, tmp_path, "runs")

    result = store.read_table(tmp_path, "runs")

    assert sorted(result["model_id"]) == ["EPA/m1", "LID/m2", "WQ/m3"]


def test_a_second_partitioned_write_also_appends(tmp_path):
    store.write_table(_frame(), tmp_path, "runs", partition_by=["family"])
    more = pd.DataFrame([
        {"model_id": "EPA/m9", "family": "EPA", "variant": "B",
         "engine_version": "6.0.0a3", "inp_sha256": "ddd", "wall_ms": 78.0},
    ])
    store.write_table(more, tmp_path, "runs", partition_by=["family"])

    result = store.read_table(tmp_path, "runs")

    assert len(result) == 3
    assert set(result[result.family == "EPA"]["model_id"]) == {"EPA/m1", "EPA/m9"}
