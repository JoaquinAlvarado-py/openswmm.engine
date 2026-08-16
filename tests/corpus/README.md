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

A is the baseline four of the five comparisons read (`D - C` is the
exception; it reads C and D only), and every option under
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
values, family names, variant letters, the hardness bucket labels
(`<= 2`, `2-4`, `> 4`), the surcharge strata (`active`, `inactive`) and the
`B - A` / `C - A` / `D - C` / `D - A` / `E - A` / `A - REF` axis notation are
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
- `diff` and `report` return non-zero, and write nothing, if `runs` holds
  results from more than one engine build. Two builds legitimately coexist in
  one store (they are part of the resume key), but the report will not guess
  which to publish, and `diff` will not compare one build's `.out` against
  another's under a single label. Point `--out` at a store holding a single
  build. The `REF` rows' `corpus-reference` sentinel is not a build and never
  triggers either refusal.
- `run` returns non-zero if `--engine` is missing; every stage returns non-zero
  if a flag it requires is absent.

## Reading the results

```python
import pandas as pd
runs = pd.read_parquet("results/runs")

dynwave = runs[runs["reported_routing_model"] == "DYNWAVE"]
pivot = dynwave.pivot_table(index="model_id", columns="variant",
                            values="avg_iterations_per_step")
(pivot["B"] / pivot["A"]).describe()   # Anderson under EXPLICIT/EXTRAN
(pivot["C"] / pivot["A"]).describe()   # Crank-Nicolson's effect
(pivot["D"] / pivot["C"]).describe()   # incremental Anderson under Crank-Nicolson
(pivot["E"] / pivot["A"]).describe()   # Dynamic Preissmann Slot
```

`scalars` is the long-format view of the numeric metrics only. The string
valued keys -- `reported_version`, the dates, `iteration_metric_kind` and the
four `reported_*` option/routing echoes -- stay one column each in `runs`;
melting them into `scalars`'s numeric `value` column would mix types in a
single Arrow column and the write would fail outright.

`iteration_metric_kind` must be checked before comparing iteration counts:
`fv_substeps` rows count explicit substeps, not Picard iterations. It is
`None` -- not `picard` -- for KINWAVE and STEADY runs: the engine prints
`Average Iterations per Step` for every routing model and relabels it only
under FV, so the label alone does not mean the counter has any Picard
meaning. Check `reported_routing_model` too; the pivot above pools routing
models and only makes sense filtered to `DYNWAVE`.

### `deltas` and `element_deltas`

Both tables carry one value column per variant and one column per
comparison axis:

| column | meaning |
| --- | --- |
| `value_a` .. `value_e`, `value_ref` | the raw per-variant value |
| `delta_b_minus_a` | Anderson under EXPLICIT/EXTRAN |
| `delta_c_minus_a` | semi-implicit (Crank-Nicolson) node continuity |
| `delta_d_minus_c` | **incremental Anderson under the C1-smooth operator** |
| `delta_d_minus_a` | joint Anderson + Crank-Nicolson (two causes, not attributable) |
| `delta_e_minus_a` | Dynamic Preissmann Slot |
| `delta_a_minus_ref` | parity debt against EPA SWMM 5.2 |

A row is always emitted, never dropped. An individual delta is null when
either operand is missing, or when the pairing is not commensurable:

- **iteration counter kind** -- FV substeps are not Picard iterations, so a
  delta is dropped when both sides' `iteration_metric_kind` are known and
  differ. A missing kind is *not* a mismatch: a variant that crashed must
  not cost the model its other deltas.
- **routing model** -- an iteration delta (`avg_iterations_per_step`,
  `pct_steps_not_converging`) is formed only when *both* sides are known
  `DYNWAVE`. This is checked explicitly rather than left to the kind guard,
  because two KINWAVE runs would agree on their (meaningless) label and pass
  straight through it.
- **surcharge method, against the anchor only** -- `X - REF` is dropped when
  the two sides echo different `SURCHARGE_METHOD` values. EPA SWMM 5.2 has
  no Dynamic Preissmann Slot, so this catches E automatically, and being
  written against the echoes it also catches a corpus reference run under a
  surcharge method other than ours. It deliberately does not apply to
  `E - A`: that delta moving the surcharge method is the point of variant E.

`summary.md` publishes a per-metric table for each of the five feature axes,
plus iteration-shift tables stratified two ways and a D/E surcharge-activity
section:

- **routing model** -- the iteration-shift sections show `DYNWAVE` rows only.
- **withheld pairings** -- the routing-model guard is deliberately stricter
  than the kind guard, so the report itemises what it cost, counted per
  (model, comparison) pairing: `not_dynwave` (expected and healthy -- a
  KINWAVE/STEADY/FV run has no Picard count to compare) and
  `routing_unknown` (actionable -- no `reported_routing_model` was recorded
  for one side, typically a store written before that column existed or an
  unreadable report). Only pairings that had both operand values and would
  otherwise have been computed are counted. If the iteration sections come
  out entirely empty the report says so outright, because four bare empty
  tables read as "the feature has no effect" when what happened is that
  nothing was measured.
- **hardness** -- split by variant A's `avg_iterations_per_step` into
  `<= 2`, `2-4` and `> 4` (closed on the right; the three partition the
  reals). Anderson cannot help a model that already converges in two
  iterations, and this corpus is dominated by small decks, so a single
  unstratified mean understates the feature exactly where it should pay off.
- **surcharge activity** -- D and E act on the free-surface/surcharge
  transition, so their rows are split by whether the model actually
  surcharges. The proxy is variant A's `pct_steps_not_converging > 0`, or
  any node with `node_total_flood_volume > 0` or `node_hours_flooded > 0`
  in `elements`; both are read from variant A only, so the stratum does not
  depend on the feature being measured. `node_max_hgl` is parsed but
  deliberately unused: separating surcharge from ordinary free-surface depth
  needs each node's crown elevation, which the harness does not parse. When
  no model (or fewer than three) is surcharge-active, the report **says so
  in words** rather than printing a near-zero `active` mean -- "the feature
  was not exercised" and "the feature has no effect" are different
  conclusions, and a markdown table cannot tell them apart on its own.

`ts_diff` carries all five time-series comparisons in one table, distinguished
by its `comparison` column (`B_minus_A`, `C_minus_A`, `D_minus_A`,
`E_minus_A` and `D_minus_C`):

```python
ts_diff = pd.read_parquet("results/ts_diff")
ts_diff[ts_diff["comparison"] == "C_minus_A"]   # Crank-Nicolson's effect on state
ts_diff[ts_diff["comparison"] == "D_minus_C"]   # incremental Anderson under Crank-Nicolson
```

`D_minus_C` is stored directly, and it has to be: **do not try to form it by
subtracting `C_minus_A` from `D_minus_A`.** A `ts_diff` row holds `max_abs`,
`max_rel`, `rmse` and `first_div_period` -- non-linear reductions over the
pointwise difference of two whole series. `max_abs(D, C)` is not
`max_abs(D, A) - max_abs(C, A)`; the triangle inequality bounds it, it does
not give its value. `D_minus_A` remains a joint Anderson + Crank-Nicolson
effect and is not attributable to one cause; `D_minus_C` is the attributable
measurement, and only the row labelled `D_minus_C` carries it.

The scalar tables behave differently, and the contrast is worth keeping
straight: a scalar metric is a single number per (model, variant), so at that
level `D - C` is a plain linear subtraction, `value_d - value_c`. That
linearity is exactly what fails for the time-series reductions above.
