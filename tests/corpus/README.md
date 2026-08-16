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

Because D completes the grid, A/B/C/D is a full **2x2 factorial** over
`ANDERSON_ACCEL` x `NODE_CONTINUITY`, and the harness publishes its
interaction, `(D - C) - (B - A)`, alongside the simple effects -- see
`delta_interaction` below.

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
| `report` | `runs`, `elements`, `ts_diff` | `deltas`, `element_deltas`, `topology_status`, `summary.md` | `--out`, `--lang` |
| `check` | the corpus tree | nothing | `--corpus-root`, `--out` |

`--lang` (`es` or `en`, default `es`) selects the language of `summary.md`'s
prose -- section headings, table column headers, and the Caveats bullets --
and of the `report` stage's own console messages. Metric names, status
values, family names, variant letters, the hardness bucket labels
(`<= 2`, `2-4`, `> 4`), the surcharge strata (`active`, `inactive`) and the
`B - A` / `C - A` / `D - C` / `D - A` / `E - A` / `A - REF` /
`(D - C) - (B - A)` axis notation are
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

## Case identity, and what makes a sweep resume

A **case** is one unit of work: this model, under this variant, executed by
*this engine build*, against *this state of the corpus*. It is hashed into a
single `case_id`, carried on every row of `runs`, `scalars`, `elements`,
`ts_diff`, `deltas` and `element_deltas`, and it is the whole resume key.

| column | meaning |
| --- | --- |
| `case_id` | sha256 of `model_id` + `variant` + `engine_build_id` + the corpus dependency identity + the variant's resolved option set. The resume key, and the join key between the tables. |
| `engine_build_id` | `sha256:<digest>` over the engine argv, each element replaced by its file hash. **This is the engine's identity.** |
| `engine_version` | the first line of `--version`. A human-readable label only. |
| `engine_build_info` | any commit / branch / build-type lines `--version` printed, when it printed them. Never depended on. |
| `corpus_commit` | `git:<sha>` of the corpus root, read at `inventory` time. The identity of everything a deck depends on. |
| `inp_sha256` | the deck's own hash. Retained as data; on a **pinned** corpus it is not part of the identity (the commit subsumes it), and on an unpinned one it is folded in as the fallback (see below). |
| `options_applied` | the option set the deck was written with, as text. Its hash is part of `case_id`. |

Three things that look like identities are deliberately not used as such:

- **The version string is not the engine.** Two executables that both print
  `OpenSWMM 6.0` can contain completely different Anderson, continuity or
  slot implementations. `engine_build_id` hashes the file. It hashes the
  whole argv (each element that names a readable file by content) so that a
  launcher invocation -- `[python, engine.py]`, `[wine, openswmm.exe]` --
  identifies the real program and not just the launcher. If argv's first
  element is not a readable file at all, the id is still deterministic but
  is prefixed `argv:` instead of `sha256:`, marking it as **not** a
  cryptographic build pin. A sweep degrades; it never crashes on this.
- **The deck is not the input.** Corpus decks reference external data by
  relative path -- `DataFiles/*.dat`, loose `.txt` series, interface files.
  If `Example.inp` is untouched but `DataFiles/rainfall.dat` changes, a
  deck-only hash says "already done" and the sweep republishes stale numbers.
  The dependency identity is therefore the corpus's **git commit**, not a
  parse of the deck's file references: enumerating those means covering
  `[RAINGAGES] FILE`, `[TIMESERIES] FILE`, `[TEMPERATURE] FILE`, the
  `[FILES]` interface section and more, and any section type missed makes
  the hash *lie*. The commit covers every referenced file exactly, for free.
  It over-invalidates when the corpus moves -- rare, and the safe direction.
- **The unpinned sentinel is not a dependency identity on its own.** A corpus
  that is not a git repository (or a machine with no git) records
  `unpinned:not-a-git-repo` rather than failing the sweep or pretending it is
  pinned -- but that value is a *constant*, so used as the whole dependency
  identity nothing about the corpus could ever invalidate a resumed sweep:
  rewrite a deck, re-run `inventory` and `run`, and every case reads as
  already done while `runs` keeps the stale `inp_sha256`. The deck's own hash
  is therefore folded in behind the sentinel whenever the corpus is unpinned,
  so **a changed deck still re-runs**. What that fallback still cannot see is
  exactly what only a commit can: a change to a file the deck *references*
  rather than to the deck itself, and a change to a corpus `.rpt` anchor. The
  value keeps its `unpinned:` prefix, so a degraded identity can never compare
  equal to a real `git:` pin. Both `inventory` and `run` say so on the
  console -- `run` because the documented workflow makes it a separate
  invocation, and the operator starting a multi-hour sweep is the one who
  needs to know.
- **The variant letter is not the option set.** `B` means whatever
  `variants.OPTIONS["B"]` says it means *today*, and that table has been
  edited more than once. `options_applied` records the resolved set on every
  row, and its hash is part of `case_id`, so editing a variant's options
  re-runs that variant -- and only that variant -- instead of colliding with
  the pre-edit rows under one id. The hash is taken over the sorted
  key/value pairs, so reordering the table (or moving a key into the shared
  block) leaves every existing case valid.

`REF` rows carry the `corpus-reference` sentinel as their `engine_build_id`:
no build of ours produced them, so a second engine does not re-read the
anchors, but a new corpus commit does.

Sweeps are resumable: re-running `run` skips any case already recorded in
`runs`. `runs`, `scalars`, `elements` and `ts_diff` therefore accumulate.
`models` and the `report` outputs are pure derived data and are **replaced**
on every re-run, so running `inventory` or `report` twice never doubles a row
count. A store written before `case_id` existed has no such column; every
case then re-runs, which is the safe direction, since those rows' engine
build cannot be established after the fact.

**Known limitation: `--timeout` is not part of `case_id`.** A case recorded as
`timeout` under `--timeout 60` is therefore *not* retried by re-running with
`--timeout 3600`; the resume key sees one case, already done. Putting the
timeout in the identity was considered and rejected, because it would
invalidate every **successful** result in the store the moment the flag moved
-- re-running ~8200 simulations to retry the handful that timed out, and again
on the next adjustment. To retry the timed-out cases, either point `--out` at
a fresh store, or delete the `runs` rows whose `status` is `timeout` (with
their `scalars` and `elements` rows) and re-run: `runs` is the resume
authority, so a case absent from it is simply recomputed.

`run` and `diff` write Parquet in batches of `FLUSH_BATCH_SIZE` (50)
completed units of work -- a simulation in `run`, a model in `diff` -- rather
than once at the end. A full sweep is ~8200 simulations over several hours;
a machine failure now costs at most one unflushed batch instead of
everything. Each stage writes derived rows before the thing they are derived
from is destroyed or claimed: `diff` flushes `ts_diff` before unlinking any
`.out`, and `run` writes `elements` and `scalars` before the `runs` rows that
would let a resume skip them.

Exit codes are meaningful and worth wiring into CI:

- `run` and `check` return non-zero if any harness temporary deck survives in
  the corpus. The corpus is read-only by contract; a leftover deck would be
  inventoried as a model by the next sweep.
- `diff` and `report` return non-zero, and write nothing, if `runs` holds
  results from more than one engine build. Two builds legitimately coexist in
  one store (the build is part of every `case_id`), but the report will not
  guess which to publish, and `diff` will not compare one build's `.out`
  against another's under a single label. Point `--out` at a store holding a
  single build. The check reads `engine_build_id`, falling back per row to
  `engine_version` only for rows written before the hash existed -- a guard
  reading the printed string alone would wave through exactly the mix it
  exists to catch. The `REF` rows' `corpus-reference` sentinel is not a build
  and never triggers either refusal.
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
| `case_id_a` .. `case_id_e`, `case_id_ref` | the `case_id` of the run each value came from |
| `delta_b_minus_a` | Anderson under EXPLICIT/EXTRAN |
| `delta_c_minus_a` | semi-implicit (Crank-Nicolson) node continuity |
| `delta_d_minus_c` | **incremental Anderson under the C1-smooth operator** |
| `delta_d_minus_a` | joint Anderson + Crank-Nicolson (two causes, not attributable) |
| `delta_e_minus_a` | Dynamic Preissmann Slot |
| `delta_a_minus_ref` | parity debt against EPA SWMM 5.2 |
| `delta_interaction` | **`(D - C) - (B - A)`** -- the 2x2 factorial interaction |

A/B/C/D is a full **2x2 factorial** over `ANDERSON_ACCEL` x
`NODE_CONTINUITY`, and `delta_interaction` is the one measurement the five
simple effects cannot give: *does switching to semi-implicit continuity
change how effective Anderson acceleration is?* `B - A` measures Anderson
under `EXPLICIT`, `D - C` measures it under `SEMI_IMPLICIT`, and only their
difference -- `(D - C) - (B - A)`, equivalently `D - C - B + A` -- says
whether the operator changed the answer.

**Sign convention.** It is a signed change in the metric itself, in that
metric's own units, not a score. It is **negative when Anderson moves the
metric further down under `SEMI_IMPLICIT` than under `EXPLICIT`** -- so for
`avg_iterations_per_step`, `pct_steps_not_converging` and `wall_ms`, where
lower is better, a negative interaction means **Anderson helps more under
Crank-Nicolson**, and a positive one means it helps less. Negative is the
expected direction, because `DWSolver::computeAASkipFlags` disables Anderson
at every surcharged node under `EXPLICIT`/`EXTRAN` and not under
`SEMI_IMPLICIT`: `B - A` measures Anderson with its most valuable case
switched off. For the `continuity_error_*` metrics the sign is directional in
the error, not in its magnitude. A near-zero interaction is a finding too --
evidence that the skip flags do not matter on this corpus -- which is why it
is quantified rather than assumed.

`delta_interaction` is computed from `delta_d_minus_c` and `delta_b_minus_a`,
not from the four raw values, so it is null exactly when either constituent
pairing is: an operand missing, or the pairing rejected by one of the
commensurability screens below. There is no fifth screen to drift out of step
with the four, and a case the screens reject cannot re-enter through the
interaction.

Both tables are recomputed and **replaced** on every `report`, so the
`case_id_*` columns are how a published number stays traceable: each one
names the exact executable and corpus state behind the value beside it. One
per variant rather than one per row, because a delta row is a join across six
runs and a single id would have to pick a side.

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
- **echoed option value vs. the variant's own intent** -- a delta is dropped
  when either operand's `reported_node_continuity`,
  `reported_anderson_accel` or `reported_surcharge_method` contradicts what
  that variant's deck asked for. The engine silently ignores an unrecognised
  option value, so an E run that actually executed under `EXTRAN` would make
  `E - A` a subtraction of two identical configurations and publish it as
  ~0 -- "the slot has no effect" when the slot was never enabled. A missing
  echo is not a contradiction, and REF has no intent to contradict. This is
  defence in depth on the anomaly report below: same mapping, both ends.

`summary.md` publishes a per-metric table for each of the five feature axes,
plus iteration-shift tables stratified two ways and a D/E surcharge-activity
section:

- **option anomalies** -- how many runs echoed an option value contradicting
  their variant's intent, broken down by variant and option, and an explicit
  statement when there are none. The console warning is seen once by one
  operator; `summary.md` is the artifact that gets kept.
- **incomplete time-series overlap** -- how many models produced a `ts_diff`
  comparison whose two runs did not cover the same reporting grid, broken
  down by `comparison` with the worst `coverage_fraction` beside it. The
  headline counts **distinct models across comparisons**, not the sum of the
  per-comparison counts: one truncated run is flagged by all five
  comparisons, and summing them would announce a single stopped simulation
  as five models. See
  *Time-series coverage* below. A store with no `ts_diff` coverage columns is
  reported as **not assessed**, never as complete.
- **Anderson x Node Continuity interaction** -- `delta_interaction` per
  metric (iterations, non-convergence, runtime and continuity error), with
  the sign convention stated in the section itself, because a bare number
  cannot carry it. An empty table says so in words: the interaction is null
  whenever either simple effect is, and *not measured* is not *no
  interaction*.
- **routing model** -- the iteration-shift sections show `DYNWAVE` rows only.
- **withheld pairings** -- the routing-model guard is deliberately stricter
  than the kind guard, so the report itemises what it cost, counted per
  (model, comparison) pairing: `not_dynwave` (expected and healthy -- a
  KINWAVE/STEADY/FV run has no Picard count to compare) and
  `routing_unknown` (actionable -- no `reported_routing_model` was recorded
  for one side, typically a store written before that column existed or an
  unreadable report). Only pairings that had both operand values and would
  otherwise have been computed are counted. A third line,
  `no_baseline_row`, counts the opposite case: a delta that WAS computed but
  has no variant A baseline to stratify it by (a `D - C` whose A run
  crashed reads C and D only, and is perfectly valid). Those appear in the
  iteration-shift tables under the `unknown` hardness bucket, so nothing
  computed ever vanishes from both the table and the accounting. If the
  iteration sections come
  out entirely empty the report says so outright, because four bare empty
  tables read as "the feature has no effect" when what happened is that
  nothing was measured.
- **hardness** -- split by variant A's `avg_iterations_per_step` into
  `<= 2`, `2-4` and `> 4` (closed on the right; the three partition the
  reals), plus `unknown` for a pairing with no usable A baseline. Anderson
  cannot help a model that already converges in two iterations, and this
  corpus is dominated by small decks, so a single unstratified mean
  understates the feature exactly where it should pay off.
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
  conclusions, and a markdown table cannot tell them apart on its own. The
  announcement is also made **per axis**: a store can hold fifty
  surcharge-active models and still produce no `active` row for `D - C`, and
  that axis would otherwise print `inactive` rows alone with no note.

`ts_diff` carries all five time-series comparisons in one table, distinguished
by its `comparison` column (`B_minus_A`, `C_minus_A`, `D_minus_A`,
`E_minus_A` and `D_minus_C`):

```python
ts_diff = pd.read_parquet("results/ts_diff")
ts_diff[ts_diff["comparison"] == "C_minus_A"]   # Crank-Nicolson's effect on state
ts_diff[ts_diff["comparison"] == "D_minus_C"]   # incremental Anderson under Crank-Nicolson
```

A comparison has two sources, so a `ts_diff` row carries `case_id_left` and
`case_id_right` -- in the label's own `X_minus_Y` order -- rather than one
`case_id` that would have to pick a side.

#### Time-series coverage

A comparison is reduced over the timestamps its two runs **share**, and
nothing else. If run A covered 0-24 h and run B stopped at 12 h, only the
shared 12 h were compared -- and if those agreed, `max_abs` reads as an
excellent result over half a run that silently failed. Every `ts_diff` row
therefore carries the evidence of what it was computed over:

| column | meaning |
| --- | --- |
| `n_periods_left` / `n_periods_right` | distinct timestamps each side reported. Both, because *shorter* is not *truncated*: either side can be the short one. |
| `n_common` | timestamps present in **both**, i.e. the periods actually reduced over. There is no separate `n_periods`. |
| `coverage_fraction` | `n_common` divided by the number of **distinct timestamps appearing in either series** (the union) -- not by either side's own length, and not by a span in hours. `1.0` exactly when the two grids are identical; `None` when neither side reported anything. |
| `start_time_match` / `end_time_match` | whether the first / last reported instants coincide. |
| `time_grid_match` | whether the two sets of timestamps are equal. This is what separates *same span, different sampling* (both ends match, the grid does not) from *same grid* (all three hold). |

An incomplete overlap is a **benchmark anomaly**, not merely a number: it is
counted, attributed by `comparison`, and stated plainly in `summary.md`
alongside the option-echo anomalies and the withheld-iteration ledger.

Such a pairing still **publishes** its `max_abs`, `max_rel` and `rmse`.
Withholding them was the alternative, and it was rejected: an exact agreement
up to the point one run stopped is a real finding, and it is precisely the
evidence that identifies the truncation as the whole story -- deleting it
would leave an operator with a missing number and no explanation. What is
withheld instead is the *conclusion*: the numbers describe the shared
sub-span only, `coverage_fraction` sits on the same row saying how much of
one, and `summary.md` names every affected comparison so none of them can be
read as a full one by default. Check `coverage_fraction` before quoting any
row's numbers.

A missing coverage value is treated as **unknown**, never as incomplete --
the same rule a missing option echo follows. A `ts_diff` written before these
columns existed is reported as *not assessed* rather than counted as a
corpus-wide anomaly.

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
