# Corpus Benchmark Harness

Runs the [1729-SWMM5-Models](https://github.com/SWMMEnablement/1729-SWMM5-Models)
corpus through the engine under five option sets, against the corpus's own
EPA SWMM 5.2 reports.

| variant | `ANDERSON_ACCEL` | `NODE_CONTINUITY` | `SURCHARGE_METHOD` | isolates |
| --- | --- | --- | --- | --- |
| A (baseline) | `NO` | `EXPLICIT` | `EXTRAN` | -- |
| B | `YES` | `EXPLICIT` | `EXTRAN` | Anderson acceleration under EXPLICIT/EXTRAN (`B - A`) |
| C | `NO` | `SEMI_IMPLICIT` | `EXTRAN` | semi-implicit / Crank-Nicolson node continuity (`C - A`) |
| D | `YES` | `SEMI_IMPLICIT` | `EXTRAN` | incremental Anderson on top of Crank-Nicolson (`D - C`); `D - A` is the joint effect |
| E | `NO` | `EXPLICIT` | `DYNAMIC_SLOT` | the Dynamic Preissmann Slot (`E - A`) |
| REF | (not run) | (not run) | (not run) | the corpus's own EPA SWMM 5.2 report; parity debt (`A - REF`) |

A is the shared baseline every comparison reads, and every option under
study is stated explicitly in all five decks (never left to an engine
default) so a future default change cannot silently redefine it. B, C and E
each move exactly one option relative to A, so `B - A`, `C - A` and `E - A`
are each attributable to a single cause.

D is the one deliberate exception to that one-feature-per-variant rule.
`DWSolver::computeAASkipFlags` (`DynamicWave.cpp:2934`) disables Anderson
acceleration on every surcharged node whenever `surcharge_method == EXTRAN`
and `node_continuity == EXPLICIT` -- exactly B's configuration -- so `B - A`
only ever measures Anderson with its most valuable case (surcharged
junctions, where Picard iteration counts explode) switched off. Under
`SEMI_IMPLICIT` the unified Crank-Nicolson update is C1-smooth through the
free-surface/surcharge transition, so that skip no longer applies and
surcharged junctions become Anderson-eligible. D turns on both
`ANDERSON_ACCEL` and `NODE_CONTINUITY` so that regime can be exercised at
all; its value comes from `D - C` (incremental Anderson on top of
Crank-Nicolson), not from `D - A` alone, which conflates two causes. Do not
otherwise combine two of these features into one variant: doing so would
make an observed shift unattributable between their causes.

`VIRTUAL_JUNCTION_MOMENTUM BASIC`, `DPS_CELERITY 25.0`, `DPS_ALPHA 3.0` and
`DPS_DECAY_TIME 0.5` are pinned identically in every variant (the engine's
own defaults, `SimulationOptions.hpp:248-262`). They are inert under EXTRAN
(A, B, C, D), so pinning them there is harmless -- and deliberate: it stops
a future default change from silently drifting E's baseline out from under
it. Note the keyword is `DPS_CELERITY`, not `DPS_TARGET_CELERITY` (the
latter is only the internal field name).

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
| `diff` | `runs`, the retained A/B/C/D/E `.out` sets | `ts_diff` | `--out`, `--abs-tol` |
| `report` | `runs`, `elements` | `deltas`, `element_deltas`, `topology_status`, `summary.md` | `--out`, `--lang` |
| `check` | the corpus tree | nothing | `--corpus-root`, `--out` |

`--lang` (`es` or `en`, default `es`) selects the language of `summary.md`'s
prose -- section headings, table column headers, and the Caveats bullets --
and of the `report` stage's own console messages. Metric names, status
values, family names and the `B - A` / `C - A` / `A - REF` axis notation are
data, not prose, and are identical in both languages so the report stays
cross-referenceable against the Parquet columns and the code regardless of
`--lang`. The default is `es` for this harness's own operator; pass
`--lang en` for an English-language report:

```bash
python -m swmmbench report --out ./results --lang en
```

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

`ts_diff` carries all four time-series comparisons in one table, distinguished
by its `comparison` column (`B_minus_A`, `C_minus_A`, `D_minus_A` or
`E_minus_A`). Every comparison is anchored on A -- it is the shared baseline
`.out` each one reads -- but that does not mean `D - A` is a one-cause
delta: it is a joint Anderson + Crank-Nicolson effect, and `D_minus_A` /
`C_minus_A` together are what let you form the attributable `D - C` pairing
yourself from this table:

```python
ts_diff = pd.read_parquet("results/ts_diff")
ts_diff[ts_diff["comparison"] == "C_minus_A"]   # Crank-Nicolson's effect on state
```
