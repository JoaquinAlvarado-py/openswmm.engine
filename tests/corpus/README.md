# Corpus Benchmark Harness

Runs the [1729-SWMM5-Models](https://github.com/SWMMEnablement/1729-SWMM5-Models)
corpus through the engine under three option sets, against the corpus's own
EPA SWMM 5.2 reports.

| variant | `ANDERSON_ACCEL` | `NODE_CONTINUITY` | isolates |
| --- | --- | --- | --- |
| A (baseline) | `NO` | `EXPLICIT` | -- |
| B | `YES` | `EXPLICIT` | Anderson acceleration (`B - A`) |
| C | `NO` | `SEMI_IMPLICIT` | semi-implicit / Crank-Nicolson node continuity (`C - A`) |
| REF | (not run) | (not run) | the corpus's own EPA SWMM 5.2 report; parity debt (`A - REF`) |

A is the shared baseline for both feature comparisons, and every option
under study is stated explicitly in all three decks (never left to an
engine default) so a future default change cannot silently redefine it.
B and C each move exactly one option relative to A, so `B - A` and `C - A`
are each attributable to a single cause. B and C are never combined into
one variant: doing so would make an observed shift unattributable between
Anderson acceleration and Crank-Nicolson continuity.

Design: `docs/superpowers/specs/2026-08-15-corpus-benchmark-design.md`

## Getting the corpus

The corpus is ~1.7 GB and is **not** vendored here.

```bash
git clone https://github.com/SWMMEnablement/1729-SWMM5-Models.git
```

The harness only ever reads it. Temporary decks are written beside their
originals and removed; `swmmbench check` verifies nothing was left behind.

## Running a sweep

```bash
cd tests/corpus
pip install -r requirements.txt

python -m swmmbench inventory --corpus-root /path/to/1729-SWMM5-Models --out ./results
python -m swmmbench run       --corpus-root /path/to/1729-SWMM5-Models --out ./results \
                              --engine /path/to/openswmm --jobs 8 --timeout 600
python -m swmmbench diff      --out ./results --abs-tol 1e-9
python -m swmmbench report    --out ./results
python -m swmmbench check     --corpus-root /path/to/1729-SWMM5-Models --out ./results
```

`diff` and `report` read only the Parquet store, so they take no
`--corpus-root`; `inventory`, `run` and `check` require one.

| stage | reads | writes | flags |
| --- | --- | --- | --- |
| `inventory` | the corpus tree | `models` | `--corpus-root`, `--out` |
| `run` | `models`, the corpus tree | `runs`, `scalars`, `elements` | `--corpus-root`, `--out`, `--engine`, `--jobs`, `--timeout`, `--limit` |
| `diff` | `runs`, the retained A/B/C `.out` triples | `ts_diff` | `--out`, `--abs-tol` |
| `report` | `runs`, `elements` | `deltas`, `element_deltas`, `topology_status`, `summary.md` | `--out` |
| `check` | the corpus tree | nothing | `--corpus-root`, `--out` |

`--abs-tol` is an **absolute** tolerance on the time-series difference
(`|b - a|`), not a relative one. The reported `max_rel` is scaled by the
baseline but is never thresholded.

Sweeps are resumable: re-running `run` skips any (model, variant) already
recorded for the same engine version and input hash. `runs`, `scalars`,
`elements` and `ts_diff` therefore accumulate. `models` and the `report`
outputs are pure derived data and are **replaced** on every re-run, so running
`inventory` or `report` twice never doubles a row count.

Exit codes are meaningful and worth wiring into CI:

- `run` and `check` return non-zero if any harness temporary deck survives in
  the corpus. The corpus is read-only by contract; a leftover deck would be
  inventoried as a model by the next sweep.
- `report` returns non-zero, and writes nothing, if `runs` holds results from
  more than one engine build. Two builds legitimately coexist in one store
  (they are part of the resume key), but the report will not guess which to
  publish. Point `--out` at a store holding a single build.
- `run` returns non-zero if `--engine` is missing; every stage returns non-zero
  if a flag it requires is absent.

## Reading the results

```python
import pandas as pd
runs = pd.read_parquet("results/runs")

pivot = runs.pivot_table(index="model_id", columns="variant",
                         values="avg_iterations_per_step")
(pivot["B"] / pivot["A"]).describe()   # Anderson's effect on iteration count
(pivot["C"] / pivot["A"]).describe()   # Crank-Nicolson's effect on iteration count
```

`iteration_metric_kind` must be checked before comparing iteration counts:
`fv_substeps` rows count explicit substeps, not Picard iterations.

`ts_diff` carries both time-series comparisons in one table, distinguished by
its `comparison` column (`B_minus_A` or `C_minus_A`):

```python
ts_diff = pd.read_parquet("results/ts_diff")
ts_diff[ts_diff["comparison"] == "C_minus_A"]   # Crank-Nicolson's effect on state
```
