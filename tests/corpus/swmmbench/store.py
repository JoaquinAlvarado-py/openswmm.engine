"""Parquet persistence for the pipeline stages.

Each stage writes one dataset; later stages read it back. Resumability is
keyed on `case_id` alone (see `cli.RESUME_KEY`), which already folds in the
engine build and the corpus dependency identity, so a sweep can never
silently mix results from two engine builds or two corpus states.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _dataset_path(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / name


def write_table(
    frame: pd.DataFrame,
    out_dir: Path,
    name: str,
    partition_by: list[str] | None = None,
) -> Path:
    """Append `frame` to the Parquet dataset under `out_dir/name`.

    The basename template carries a fresh UUID on every call. pyarrow's
    default is `part-{i}`, which a resumed sweep would reuse and thereby
    overwrite the previous sweep's rows — silently, and only discoverable
    once the analysis came out short.
    """
    target = _dataset_path(out_dir, name)
    target.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(target),
        partition_cols=partition_by or None,
        basename_template=f"part-{uuid.uuid4().hex}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )
    return target


def read_table(out_dir: Path, name: str) -> pd.DataFrame:
    """Read a dataset, or an empty frame when it does not exist yet."""
    target = _dataset_path(out_dir, name)
    if not target.exists() or not any(target.rglob("*.parquet")):
        return pd.DataFrame()
    return pq.read_table(str(target)).to_pandas()


def completed_keys(out_dir: Path, name: str, columns: list[str]) -> set[tuple]:
    """Key tuples already present in `name`, for skipping finished work."""
    frame = read_table(out_dir, name)
    if frame.empty or not set(columns).issubset(frame.columns):
        return set()
    return set(map(tuple, frame[columns].astype(str).itertuples(index=False)))
