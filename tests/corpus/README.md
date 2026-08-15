# Corpus Benchmark Harness

Runs the [1729-SWMM5-Models](https://github.com/SWMMEnablement/1729-SWMM5-Models)
corpus through the engine under two option sets, against the corpus's own
EPA SWMM 5.2 reports.

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
python -m swmmbench check     --corpus-root /path/to/1729-SWMM5-Models --out ./results
```

Sweeps are resumable: re-running `run` skips any (model, variant) already
recorded for the same engine version and input hash.

## Reading the results

```python
import pandas as pd
runs = pd.read_parquet("results/runs")

pivot = runs.pivot_table(index="model_id", columns="variant",
                         values="avg_iterations_per_step")
(pivot["B"] / pivot["A"]).describe()   # Anderson's effect on iteration count
```

`iteration_metric_kind` must be checked before comparing iteration counts:
`fv_substeps` rows count explicit substeps, not Picard iterations.
