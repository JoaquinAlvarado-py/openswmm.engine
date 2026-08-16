"""Materialise the comparison axes and a human-readable summary.

`B - A` isolates Anderson acceleration under EXPLICIT/EXTRAN. `C - A`
isolates semi-implicit (Crank-Nicolson) node continuity. `D - C` isolates
Anderson acceleration *under* that C1-smooth operator -- the regime B cannot
exercise, because `DWSolver::computeAASkipFlags` disables Anderson at every
surcharged node under EXPLICIT/EXTRAN -- and `D - A` is the joint
Anderson + Crank-Nicolson effect. `E - A` isolates the Dynamic Preissmann
Slot. `A - REF` measures parity debt against EPA SWMM 5.2. Keeping all of
them side by side rather than collapsed is the reason the A/B/C/D/E matrix
exists: folding two feature changes into one variant would make an observed
shift unattributable between their two causes.

Unlike the time-series stage (`cli.stage_diff`), `D - C` here is a plain
linear subtraction of two scalars, so it is formed from the same pivot as
every other axis rather than needing its own comparison pass.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema, variants

DEFAULT_METRICS = [
    "avg_iterations_per_step",
    "pct_steps_not_converging",
    "continuity_error_runoff",
    "continuity_error_flow",
    "continuity_error_quality",
    "avg_step",
    "wall_ms",
]

#: Metrics whose comparability depends on iteration_metric_kind matching AND
#: on both operands having been routed by the dynamic wave solver.
ITERATION_METRICS = {"avg_iterations_per_step", "pct_steps_not_converging"}

#: The only routing model under which `Average Iterations per Step` counts
#: Picard iterations of the dynamic-wave solver. The engine prints that line
#: for KINWAVE and STEADY too (DefaultReportPlugin.cpp), where the counter
#: has no Picard meaning at all -- so an iteration delta is only formed when
#: BOTH operands are known to be DYNWAVE.
DYNWAVE = schema.ROUTING_DYNWAVE

#: Every variant that can carry a value on a delta row, in report order. The
#: emitted column is `value_<lowercased variant>` -- `value_a` .. `value_e`
#: and `value_ref`.
DELTA_VARIANTS = (
    schema.VARIANT_A, schema.VARIANT_B, schema.VARIANT_C,
    schema.VARIANT_D, schema.VARIANT_E, schema.VARIANT_REF,
)

#: `(column, left, right)` for every delta emitted, computed as
#: `value_left - value_right`.
#:
#: `delta_d_minus_c` is the column the whole variant-D exercise exists to
#: produce: with NODE_CONTINUITY held at SEMI_IMPLICIT on both sides, the
#: only key that moves between C and D is ANDERSON_ACCEL, so the difference
#: is attributable to Anderson alone -- in the surcharge regime B cannot
#: reach. `delta_d_minus_a` is deliberately kept alongside it and is NOT a
#: substitute: it moves two keys at once and is a joint effect.
DELTA_SPECS = (
    ("delta_b_minus_a", schema.VARIANT_B, schema.VARIANT_A),
    ("delta_c_minus_a", schema.VARIANT_C, schema.VARIANT_A),
    ("delta_d_minus_a", schema.VARIANT_D, schema.VARIANT_A),
    ("delta_d_minus_c", schema.VARIANT_D, schema.VARIANT_C),
    ("delta_e_minus_a", schema.VARIANT_E, schema.VARIANT_A),
    ("delta_a_minus_ref", schema.VARIANT_A, schema.VARIANT_REF),
)

DELTA_COLUMNS = [column for column, _, _ in DELTA_SPECS]
VALUE_COLUMNS = [f"value_{variant.lower()}" for variant in DELTA_VARIANTS]


def _pivot(runs: pd.DataFrame, column: str) -> pd.DataFrame:
    """model_id x variant table of `column`, or an empty frame if absent."""
    if column not in runs.columns:
        return pd.DataFrame()
    try:
        return runs.pivot_table(index="model_id", columns="variant",
                                values=column, aggfunc="first")
    except (KeyError, ValueError):
        return pd.DataFrame()


def _lookup(table: pd.DataFrame, model_id, variant: str):
    """`table[model_id, variant]`, or `pd.NA` when either axis is missing."""
    if table.empty or variant not in table.columns:
        return pd.NA
    if model_id not in table.index:
        return pd.NA
    return table.loc[model_id, variant]


def _same(left, right) -> bool | None:
    """Case-insensitive echo comparison. None when either side is unknown.

    A missing echo is deliberately NOT a mismatch: a variant that crashed,
    or a report that never printed the line, must not cost the model an
    otherwise-computable delta.
    """
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return None
    return str(left).strip().upper() == str(right).strip().upper()


def _commensurable(
    metric: str,
    model_id,
    left: str,
    right: str,
    kinds: pd.DataFrame,
    routing: pd.DataFrame,
    surcharge: pd.DataFrame,
) -> bool:
    """True when `left - right` is a subtraction of two like quantities.

    Three confounds are screened, each per-delta rather than per-row so one
    bad pairing never costs an unrelated computable one:

    1. **Iteration counter kind.** Picard iterations and FV substeps are
       different counters; subtracting one from the other is meaningless.
       Only trips when BOTH kinds are known and differ.
    2. **Routing model.** The engine prints `Average Iterations per Step`
       for EVERY routing model and relabels it only under FV, so a KINWAVE
       or STEADY run carries a counter with no Picard meaning while still
       looking like one. Checked EXPLICITLY rather than left to (1): a
       missing kind is deliberately treated as "not a mismatch", and two
       KINWAVE runs agree on their (meaningless) kind anyway. Both operands
       must be known DYNWAVE.
    3. **Surcharge method, against the anchor.** Applied only when REF is
       one of the operands. E is DYNAMIC_SLOT and EPA SWMM 5.2 has no
       Dynamic Preissmann Slot, so its reports are always EXTRAN -- but the
       rule is written against the echoes, not against E, so it also catches
       a corpus reference that happens to have been run under a different
       surcharge method than ours. It must NOT apply to `E - A`: that delta
       moving the surcharge method is the entire point of variant E.
    """
    if metric in ITERATION_METRICS:
        if _same(_lookup(kinds, model_id, left),
                 _lookup(kinds, model_id, right)) is False:
            return False
        for variant in (left, right):
            model = _lookup(routing, model_id, variant)
            if model is None or pd.isna(model):
                return False
            if str(model).strip().upper() != DYNWAVE:
                return False

    if schema.VARIANT_REF in (left, right):
        if _same(_lookup(surcharge, model_id, left),
                 _lookup(surcharge, model_id, right)) is False:
            return False

    return True


def build_deltas(runs: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """One row per (model, metric) with A..E and REF values and all deltas.

    A row is always emitted, never dropped; an individual delta is null when
    either operand is missing or when `_commensurable` rejects the pairing.
    """
    if runs.empty:
        return pd.DataFrame()

    present = [m for m in metrics if m in runs.columns]
    if not present:
        return pd.DataFrame()

    kinds = _pivot(runs, "iteration_metric_kind")
    routing = _pivot(runs, "reported_routing_model")
    surcharge = _pivot(runs, "reported_surcharge_method")

    families = runs.groupby("model_id")["family"].first()
    rows: list[dict] = []

    for metric in present:
        wide = runs.pivot_table(index="model_id", columns="variant",
                                values=metric, aggfunc="first")
        for model_id, record in wide.iterrows():
            values = {variant: record.get(variant) for variant in DELTA_VARIANTS}

            row = {
                "model_id": model_id,
                "family": families.get(model_id),
                "metric": metric,
            }
            row.update({f"value_{variant.lower()}": values[variant]
                        for variant in DELTA_VARIANTS})

            for column, left, right in DELTA_SPECS:
                value_left, value_right = values[left], values[right]
                usable = (pd.notna(value_left) and pd.notna(value_right)
                          and _commensurable(metric, model_id, left, right,
                                             kinds, routing, surcharge))
                row[column] = (value_left - value_right) if usable else pd.NA

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Engine-echoed option values vs. variant intent
# ---------------------------------------------------------------------------

#: Options whose engine-reported value can be checked against the variant's
#: stated intent, keyed by the `runs` column that carries the echo
#: (DefaultReportPlugin.cpp prints all three under the DYNWAVE-only Analysis
#: Options block; rptparse.parse_scalars reads them into these columns).
#: OptionsHandler.cpp silently ignores an unrecognised value for any of them
#: -- no `else`, no warning -- so `options_applied` (what the harness's deck
#: asked for) cannot alone prove the engine actually did it. Without this
#: check, a typo in an option name or value would read as "the feature has
#: no effect" across the whole corpus instead of as a broken deck.
OPTION_ECHO_COLUMNS = {
    "NODE_CONTINUITY": "reported_node_continuity",
    "ANDERSON_ACCEL": "reported_anderson_accel",
    "SURCHARGE_METHOD": "reported_surcharge_method",
}

ANOMALY_COLUMNS = ["model_id", "variant", "option", "expected", "reported"]


def option_anomalies(runs: pd.DataFrame) -> pd.DataFrame:
    """Executed runs whose engine-reported option value contradicts intent.

    Every executed variant is checked against `variants.OPTIONS`, so D's
    `ANDERSON_ACCEL YES` / `NODE_CONTINUITY SEMI_IMPLICIT` and E's
    `SURCHARGE_METHOD DYNAMIC_SLOT` are covered by the same loop as A/B/C --
    which matters most for E, whose entire value is the Dynamic Preissmann
    Slot: an engine that silently fell back to EXTRAN would otherwise
    publish `E - A ~ 0` as "the slot has no effect".

    A row with no echo (FV/STEADY/KINWAVE routing never prints the block at
    all) is not an anomaly: there is nothing to contradict, so it is silently
    skipped rather than flagged.
    """
    if runs.empty:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    executed = runs[runs["variant"].isin(schema.VARIANTS)]
    rows: list[dict] = []

    for option, echo_column in OPTION_ECHO_COLUMNS.items():
        if echo_column not in executed.columns:
            continue
        for _, row in executed.iterrows():
            reported = row.get(echo_column)
            # `pd.isna` alone covers None, float NaN, NaT and `pd.NA` -- the
            # narrower `isinstance(reported, float)` guard missed `pd.NA`,
            # which would stringify to "<NA>" and flag a false anomaly on
            # every row whose echo column happened to hold it.
            if reported is None or pd.isna(reported):
                continue
            expected = variants.OPTIONS.get(row["variant"], {}).get(option)
            if expected is None:
                continue
            if str(reported).strip().upper() != str(expected).strip().upper():
                rows.append({
                    "model_id": row["model_id"], "variant": row["variant"],
                    "option": option, "expected": expected, "reported": reported,
                })

    return pd.DataFrame(rows, columns=ANOMALY_COLUMNS)


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

#: Hardness buckets over variant A's `avg_iterations_per_step`, as
#: `(label, predicate)` in report order. Anderson acceleration cannot help a
#: model that already converges in two Picard iterations, and this corpus is
#: dominated by small decks -- so an unstratified mean over every model
#: understates the feature exactly where it is supposed to pay off. The
#: labels are numeric notation, not prose: they stay verbatim in both
#: languages, like the `B - A` axis notation.
HARDNESS_BUCKETS = (
    ("<= 2", lambda value: value <= 2.0),
    ("2-4", lambda value: 2.0 < value <= 4.0),
    ("> 4", lambda value: value > 4.0),
)

HARDNESS_LABELS = [label for label, _ in HARDNESS_BUCKETS]


def hardness_bucket(value) -> str | None:
    """The bucket `value` falls in, or None when it is missing.

    The buckets partition the reals: `<= 2` is closed at 2, `2-4` is
    half-open `(2, 4]`, and `> 4` takes the rest, so no value lands in two
    buckets and no finite value lands in none.
    """
    if value is None or pd.isna(value):
        return None
    number = float(value)
    for label, predicate in HARDNESS_BUCKETS:
        if predicate(number):
            return label
    return None


def dynwave_models(runs: pd.DataFrame) -> set:
    """Models whose baseline (variant A) run was routed by DYNWAVE.

    The iteration-shift sections are restricted to these. A KINWAVE or
    STEADY run still prints an `Average Iterations per Step` line, but the
    number counts nothing Picard-shaped; pooling those into a median makes
    the median describe a quantity nobody named.
    """
    if runs.empty or "reported_routing_model" not in runs.columns:
        return set()
    baseline = runs[runs["variant"] == schema.VARIANT_A]
    routed = baseline[
        baseline["reported_routing_model"].astype("string").str.strip().str.upper()
        == DYNWAVE
    ]
    return set(routed["model_id"])


#: Per-element metrics that evidence flooding, i.e. a node whose hydraulic
#: grade rose above the ground and spilled. Either being positive for any
#: node of a model is taken as surcharge activity.
FLOODING_METRICS = ("node_total_flood_volume", "node_hours_flooded")

#: Below this many surcharge-active models, the D/E surcharge strata are
#: reported as anecdote rather than as a measurement. A mean over one or two
#: models reads exactly like a mean over four hundred in a markdown table,
#: and the reader cannot tell them apart without being told.
SURCHARGE_ACTIVE_MIN = 3

#: Strata labels for the surcharge-activity split. Status-like data values,
#: verbatim in both languages.
SURCHARGE_ACTIVE = "active"
SURCHARGE_INACTIVE = "inactive"


def surcharge_active_models(
    runs: pd.DataFrame, elements: pd.DataFrame | None = None,
) -> set:
    """Models whose baseline run shows evidence of surcharging.

    The proxy, and why this combination:

    * `pct_steps_not_converging > 0` on variant A. Under EXTRAN the
      free-surface/surcharge transition is the dominant source of
      non-convergence in the dynamic-wave solver, so a model that never
      fails to converge is very unlikely to be pressurising anything.
    * any node with `node_total_flood_volume > 0` or
      `node_hours_flooded > 0` on variant A. This catches the pressurised
      models that converge anyway, which the first signal alone would miss.

    `node_max_hgl` is parsed by the harness but is deliberately NOT used:
    an HGL is an elevation, and telling surcharge from ordinary free-surface
    depth needs each node's crown elevation, which the harness does not
    parse. There is no threshold to compare a bare HGL against, so including
    it would add noise rather than signal.

    Both signals are read from variant A only -- the baseline every D and E
    delta is anchored to -- so the stratum a model lands in does not depend
    on the very feature being measured.
    """
    active: set = set()

    if not runs.empty and "variant" in runs.columns:
        baseline = runs[runs["variant"] == schema.VARIANT_A]
        if "pct_steps_not_converging" in baseline.columns:
            values = pd.to_numeric(baseline["pct_steps_not_converging"],
                                   errors="coerce")
            active |= set(baseline.loc[values > 0, "model_id"])

    if elements is not None and not elements.empty:
        needed = {"variant", "metric", "value", "model_id"}
        if needed <= set(elements.columns):
            flooding = elements[
                (elements["variant"] == schema.VARIANT_A)
                & (elements["metric"].isin(FLOODING_METRICS))
            ]
            values = pd.to_numeric(flooding["value"], errors="coerce")
            active |= set(flooding.loc[values > 0, "model_id"])

    return active


#: Default report language. The harness's own author works in Spanish, but
#: the harness is being submitted upstream to an English-language project --
#: so the *code* stays English (identifiers, comments, docstrings) while only
#: the generated prose is switchable. `es` is the default because that is
#: this harness's primary operator; `en` exists for the upstream project.
DEFAULT_LANG = "es"

#: All prose that lands in the generated `summary.md`, keyed by language so
#: the two versions sit side by side and cannot drift apart -- deliberately
#: not a scatter of `if lang == "es"` branches through write_markdown's body.
#: Metric names, status values, family names, variant letters, the hardness
#: bucket labels (`<= 2`, `2-4`, `> 4`), the surcharge strata (`active`,
#: `inactive`) and the `B - A` / `C - A` / `D - C` / `D - A` / `E - A` /
#: `A - REF` axis notation are NOT here: they are Parquet column values and
#: identifiers a reader cross-references against the data and the code, and
#: must stay verbatim in both languages.
MARKDOWN_STRINGS = {
    "en": {
        "title": "# Corpus Benchmark Summary",
        "coverage_heading": "## Coverage",
        "coverage_header": "| status | runs |",
        "coverage_sep": "| --- | --- |",
        "b_minus_a_heading": "## B - A by metric (Anderson acceleration effect)",
        "c_minus_a_heading": "## C - A by metric (Crank-Nicolson continuity effect)",
        "d_minus_c_heading": "## D - C by metric (incremental Anderson under Crank-Nicolson)",
        "d_minus_a_heading": "## D - A by metric (joint Anderson + Crank-Nicolson effect)",
        "e_minus_a_heading": "## E - A by metric (Dynamic Preissmann Slot effect)",
        "metric_header": "| metric | models | mean | median | min | max |",
        "metric_sep": "| --- | --- | --- | --- | --- | --- |",
        "iter_shift_b_heading": "## Iteration shift by family and hardness (B - A)",
        "iter_shift_c_heading": "## Iteration shift by family and hardness (C - A)",
        "iter_shift_d_heading": "## Iteration shift by family and hardness (D - C)",
        "iter_shift_e_heading": "## Iteration shift by family and hardness (E - A)",
        "family_header": "| family | hardness | models | mean abs iterations |",
        "family_sep": "| --- | --- | --- | --- |",
        "surcharge_heading": "## D and E by surcharge activity",
        "surcharge_header": "| axis | surcharge | metric | models | mean | median |",
        "surcharge_sep": "| --- | --- | --- | --- | --- | --- |",
        "surcharge_proxy": [
            "A model counts as `active` when its variant A run reports",
            "`pct_steps_not_converging` greater than zero, or any of its nodes",
            "reports `node_total_flood_volume` or `node_hours_flooded` greater",
            "than zero. Both signals are read from variant A only, so the",
            "stratum does not depend on the feature being measured.",
            "`node_max_hgl` is parsed but not used here: separating surcharge",
            "from ordinary free-surface depth needs each node's crown",
            "elevation, which the harness does not parse.",
        ],
        "surcharge_none": [
            "**No model in this store is surcharge-active by that proxy.** D and",
            "E act on the free-surface/surcharge transition, so this sweep",
            "measures them where they are not designed to act. Read the",
            "`inactive` rows below as *the feature was not exercised*, never as",
            "*the feature has no effect* -- those two conclusions are not the",
            "same, and only the second one is refutable by this table.",
        ],
        "surcharge_few": [
            "**Only {n} surcharge-active model(s) in this store.** Any mean over",
            "them is anecdote, not evidence; the `active` rows below are shown",
            "for inspection, not for a conclusion.",
        ],
        "caveats_heading": "## Caveats",
        "caveat_estimate": [
            "- `total_iterations_est` is a derived **estimate**",
            "  (`avg_iterations_per_step x duration / avg_step`); the report",
            "  publishes only a two-decimal mean. Its error grows on",
            "  variable-time-step models. Use `avg_iterations_per_step` for any",
            "  conclusion that matters.",
        ],
        "caveat_kind_mismatch": [
            "- Rows whose `iteration_metric_kind` differs from the other side's",
            "  are excluded from that pairing's iteration delta (B - A, C - A,",
            "  D - A, D - C, E - A and A - REF independently): FV substeps are",
            "  not Picard iterations.",
        ],
        "caveat_routing_model": [
            "- Iteration deltas are formed only when BOTH sides are known to be",
            "  `DYNWAVE`, and the iteration-shift sections show only those rows.",
            "  The engine prints `Average Iterations per Step` for KINWAVE and",
            "  STEADY too, where the counter has no Picard meaning; those runs",
            "  would otherwise agree on their kind and pass straight through.",
        ],
        "caveat_hardness": [
            "- The iteration-shift tables are split by variant A's",
            "  `avg_iterations_per_step`. Anderson acceleration cannot help a",
            "  model that already converges in two iterations, and this corpus",
            "  is dominated by small decks, so a single unstratified mean",
            "  understates the feature exactly where it is meant to pay off.",
        ],
        "caveat_surcharge_ref": [
            "- `X - REF` is dropped when the two sides echo different",
            "  `SURCHARGE_METHOD` values. EPA SWMM 5.2 has no Dynamic",
            "  Preissmann Slot, so `E - REF` is always dropped; the rule is",
            "  written against the echoes, so it also catches a reference run",
            "  under a surcharge method other than ours.",
        ],
        "caveat_time_series": [
            "- Time series are compared only between our own runs. The external",
            "  anchor is summary-level.",
        ],
    },
    "es": {
        "title": "# Resumen del Benchmark del Corpus",
        "coverage_heading": "## Cobertura",
        "coverage_header": "| estado | corridas |",
        "coverage_sep": "| --- | --- |",
        "b_minus_a_heading": "## B - A por métrica (efecto de la aceleración de Anderson)",
        "c_minus_a_heading": "## C - A por métrica (efecto de continuidad de Crank-Nicolson)",
        "d_minus_c_heading": "## D - C por métrica (Anderson incremental sobre Crank-Nicolson)",
        "d_minus_a_heading": "## D - A por métrica (efecto conjunto de Anderson + Crank-Nicolson)",
        "e_minus_a_heading": "## E - A por métrica (efecto de la ranura de Preissmann dinámica)",
        "metric_header": "| métrica | modelos | media | mediana | mínimo | máximo |",
        "metric_sep": "| --- | --- | --- | --- | --- | --- |",
        "iter_shift_b_heading": "## Cambio de iteraciones por familia y dificultad (B - A)",
        "iter_shift_c_heading": "## Cambio de iteraciones por familia y dificultad (C - A)",
        "iter_shift_d_heading": "## Cambio de iteraciones por familia y dificultad (D - C)",
        "iter_shift_e_heading": "## Cambio de iteraciones por familia y dificultad (E - A)",
        "family_header": "| familia | dificultad | modelos | media de iteraciones absolutas |",
        "family_sep": "| --- | --- | --- | --- |",
        "surcharge_heading": "## D y E por actividad de sobrecarga",
        "surcharge_header": "| axis | surcharge | métrica | modelos | media | mediana |",
        "surcharge_sep": "| --- | --- | --- | --- | --- | --- |",
        "surcharge_proxy": [
            "Un modelo cuenta como `active` cuando su corrida de la variante A",
            "reporta `pct_steps_not_converging` mayor que cero, o alguno de sus",
            "nodos reporta `node_total_flood_volume` o `node_hours_flooded`",
            "mayor que cero. Ambas señales se leen solo de la variante A, de",
            "modo que el estrato no depende de la característica que se mide.",
            "`node_max_hgl` se parsea pero no se usa aquí: separar la sobrecarga",
            "de la profundidad ordinaria de superficie libre requiere la cota de",
            "clave de cada nodo, que el arnés no parsea.",
        ],
        "surcharge_none": [
            "**Ningún modelo de este almacén tiene sobrecarga activa según ese",
            "criterio.** D y E actúan sobre la transición entre superficie libre",
            "y sobrecarga, así que este barrido las mide donde no están",
            "diseñadas para actuar. Lea las filas `inactive` de abajo como *la",
            "característica no fue ejercitada*, nunca como *la característica no",
            "tiene efecto*: no son la misma conclusión, y solo la segunda sería",
            "refutable por esta tabla.",
        ],
        "surcharge_few": [
            "**Solo {n} modelo(s) con sobrecarga activa en este almacén.**",
            "Cualquier media sobre ellos es anécdota, no evidencia; las filas",
            "`active` de abajo se muestran para inspección, no para concluir.",
        ],
        "caveats_heading": "## Advertencias",
        "caveat_estimate": [
            "- `total_iterations_est` es una **estimación** derivada",
            "  (`avg_iterations_per_step x duration / avg_step`); el informe",
            "  publica solo una media con dos decimales. Su error crece en",
            "  modelos de paso de tiempo variable. Use `avg_iterations_per_step`",
            "  para cualquier conclusión que importe.",
        ],
        "caveat_kind_mismatch": [
            "- Las filas cuyo `iteration_metric_kind` difiere del del otro lado",
            "  se excluyen del delta de iteraciones de ese par (B - A, C - A,",
            "  D - A, D - C, E - A y A - REF de forma independiente): los",
            "  subpasos de FV no son iteraciones de Picard.",
        ],
        "caveat_routing_model": [
            "- Los deltas de iteraciones se forman solo cuando AMBOS lados son",
            "  `DYNWAVE` de forma conocida, y las secciones de cambio de",
            "  iteraciones muestran solo esas filas. El motor imprime",
            "  `Average Iterations per Step` también para KINWAVE y STEADY,",
            "  donde el contador no tiene significado de Picard; esas corridas",
            "  coincidirían en su tipo y pasarían sin ser filtradas.",
        ],
        "caveat_hardness": [
            "- Las tablas de cambio de iteraciones se dividen según el",
            "  `avg_iterations_per_step` de la variante A. La aceleración de",
            "  Anderson no puede ayudar a un modelo que ya converge en dos",
            "  iteraciones, y este corpus está dominado por decks pequeños, así",
            "  que una sola media sin estratificar subestima la característica",
            "  justo donde debería rendir.",
        ],
        "caveat_surcharge_ref": [
            "- `X - REF` se descarta cuando ambos lados reportan valores",
            "  distintos de `SURCHARGE_METHOD`. EPA SWMM 5.2 no tiene la ranura",
            "  de Preissmann dinámica, así que `E - REF` siempre se descarta; la",
            "  regla se escribe contra los ecos, de modo que también detecta una",
            "  referencia corrida con un método de sobrecarga distinto al",
            "  nuestro.",
        ],
        "caveat_time_series": [
            "- Las series temporales se comparan solo entre nuestras propias",
            "  corridas. El ancla externa es a nivel de resumen.",
        ],
    },
}

#: `(delta column, heading key)` for the per-metric summary tables, in
#: report order. A/B/C first and unchanged, then the two D axes, then E.
METRIC_SECTIONS = (
    ("delta_b_minus_a", "b_minus_a_heading"),
    ("delta_c_minus_a", "c_minus_a_heading"),
    ("delta_d_minus_c", "d_minus_c_heading"),
    ("delta_d_minus_a", "d_minus_a_heading"),
    ("delta_e_minus_a", "e_minus_a_heading"),
)

#: `(delta column, heading key)` for the family x hardness iteration tables.
ITER_SECTIONS = (
    ("delta_b_minus_a", "iter_shift_b_heading"),
    ("delta_c_minus_a", "iter_shift_c_heading"),
    ("delta_d_minus_c", "iter_shift_d_heading"),
    ("delta_e_minus_a", "iter_shift_e_heading"),
)

#: `(axis label, delta column)` for the surcharge-activity strata. The axis
#: labels are notation, not prose, and stay verbatim in both languages.
SURCHARGE_SECTIONS = (
    ("D - C", "delta_d_minus_c"),
    ("D - A", "delta_d_minus_a"),
    ("E - A", "delta_e_minus_a"),
)


def _delta_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """`frame[column]` as floats, or an all-empty series when absent."""
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return frame[column].dropna().astype(float)


def write_markdown(
    deltas: pd.DataFrame,
    runs: pd.DataFrame,
    path: Path,
    lang: str = DEFAULT_LANG,
    elements: pd.DataFrame | None = None,
) -> Path:
    """Write a summary readable without opening a notebook.

    `lang` selects the language of the generated prose only (headings, table
    column headers, the Caveats bullets). Metric names, status values, family
    names, hardness buckets, surcharge strata and the `B - A` / `C - A` /
    `D - C` / `D - A` / `E - A` / `A - REF` notation are data, not prose, and
    are identical in both languages.

    `elements` is optional and used only to strengthen the surcharge-activity
    proxy with per-node flooding totals; without it the proxy falls back to
    `pct_steps_not_converging` alone.
    """
    path = Path(path)
    strings = MARKDOWN_STRINGS[lang]
    lines = [strings["title"], ""]

    if not runs.empty and "status" in runs.columns:
        lines += [strings["coverage_heading"], "",
                  strings["coverage_header"], strings["coverage_sep"]]
        for status, count in runs["status"].value_counts().items():
            lines.append(f"| {status} | {count} |")
        lines.append("")

    if not deltas.empty:
        # One section per axis rather than columns of one table, so no axis
        # reads as derived from, or secondary to, another. D gets both of
        # its axes: D - C is the attributable incremental-Anderson
        # measurement, D - A the joint effect kept alongside it.
        for delta_col, heading_key in METRIC_SECTIONS:
            lines += [strings[heading_key], "",
                      strings["metric_header"], strings["metric_sep"]]
            if delta_col in deltas.columns:
                # `metric` here is plain object dtype (built row-by-row in
                # build_deltas), so groupby's default observed=False is
                # harmless.
                grouped = deltas.dropna(subset=[delta_col]).groupby("metric")
                for metric, group in grouped:
                    column = group[delta_col].astype(float)
                    lines.append(
                        f"| {metric} | {len(column)} | {column.mean():.4f} | "
                        f"{column.median():.4f} | {column.min():.4f} | "
                        f"{column.max():.4f} |"
                    )
            lines.append("")

        # Only DYNWAVE baselines participate: a KINWAVE or STEADY run still
        # prints an iteration count, and pooling those into a median makes
        # the median describe a quantity nobody named. `build_deltas`
        # already nulls those deltas; filtering here as well keeps the
        # section's contract stated where a reader can see it.
        iterations = deltas[deltas.metric == "avg_iterations_per_step"]
        routed = dynwave_models(runs)
        iterations = iterations[iterations["model_id"].isin(routed)]
        if "value_a" in iterations.columns:
            hardness = iterations["value_a"].map(hardness_bucket)
        else:
            hardness = pd.Series(index=iterations.index, dtype="object")
        iterations = iterations.assign(_hardness=hardness)

        for delta_col, heading_key in ITER_SECTIONS:
            lines += [strings[heading_key], "",
                      strings["family_header"], strings["family_sep"]]
            # `deltas["family"]` is object dtype in the direct build_deltas ->
            # write_markdown pipeline (build_deltas reconstructs it from
            # plain Python scalars), so this groupby is unaffected either
            # way here. But `deltas` can also be re-read from its own
            # persisted dataset (stage_report writes it partitioned by
            # family), where pandas restores `family` as a `category`
            # dtype. groupby on a categorical defaults to observed=False and
            # materialises EVERY category, including ones with zero rows in
            # the (already-filtered) `iterations` frame, as spurious
            # empty-group rows. observed=True avoids that in both cases.
            for family, group in iterations.groupby("family", observed=True):
                for bucket in HARDNESS_LABELS:
                    column = _delta_column(
                        group[group["_hardness"] == bucket], delta_col).abs()
                    if column.empty:
                        continue
                    lines.append(f"| {family} | {bucket} | {len(column)} | "
                                 f"{column.mean():.4f} |")
            lines.append("")

        lines += _surcharge_section(deltas, runs, elements, strings)

    lines += [strings["caveats_heading"], ""]
    lines += strings["caveat_estimate"]
    lines += strings["caveat_kind_mismatch"]
    lines += strings["caveat_routing_model"]
    lines += strings["caveat_hardness"]
    lines += strings["caveat_surcharge_ref"]
    lines += strings["caveat_time_series"]
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _surcharge_section(
    deltas: pd.DataFrame,
    runs: pd.DataFrame,
    elements: pd.DataFrame | None,
    strings: dict,
) -> list[str]:
    """The D/E surcharge-activity strata, with their honesty note.

    An empty (or near-empty) `active` stratum is stated in words. Printing
    only an `inactive` mean would read as "the feature has no effect", when
    what actually happened is that the feature was never exercised -- and a
    markdown table cannot tell those apart on its own.
    """
    active = surcharge_active_models(runs, elements) & set(deltas["model_id"])
    lines = [strings["surcharge_heading"], ""]
    lines += strings["surcharge_proxy"]
    lines.append("")

    if not active:
        lines += strings["surcharge_none"]
        lines.append("")
    elif len(active) < SURCHARGE_ACTIVE_MIN:
        lines += [line.format(n=len(active)) for line in strings["surcharge_few"]]
        lines.append("")

    lines += [strings["surcharge_header"], strings["surcharge_sep"]]
    strata = (
        (SURCHARGE_ACTIVE, deltas[deltas["model_id"].isin(active)]),
        (SURCHARGE_INACTIVE, deltas[~deltas["model_id"].isin(active)]),
    )
    for axis, delta_col in SURCHARGE_SECTIONS:
        for stratum, subset in strata:
            if subset.empty or delta_col not in subset.columns:
                continue
            for metric, group in subset.dropna(subset=[delta_col]).groupby("metric"):
                column = group[delta_col].astype(float)
                if column.empty:
                    continue
                lines.append(
                    f"| {axis} | {stratum} | {metric} | {len(column)} | "
                    f"{column.mean():.4f} | {column.median():.4f} |"
                )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Element-level comparison
# ---------------------------------------------------------------------------

ELEMENT_KEY = ["model_id", "family", "element_type", "element_id", "metric"]


def build_element_deltas(elements: pd.DataFrame) -> pd.DataFrame:
    """Per-element A/B/C/D/E/REF values and all deltas.

    No commensurability guard applies here: element metrics are depths,
    flows and volumes in fixed units, not iteration counters, so none of the
    three confounds `build_deltas` screens for can arise.
    """
    if elements.empty:
        return pd.DataFrame()

    # `elements` is read back from a dataset partitioned by `family`
    # (store.write_table(..., partition_by=["family"])), so pandas restores
    # `family` as a `category` dtype. pivot_table's index includes `family`
    # (via ELEMENT_KEY); pivoting on a categorical index materialises the
    # full cartesian product of categories by default, which with 20+
    # families and thousands of elements is a real blow-up, not a rounding
    # error. Cast to plain str first so only the categories actually present
    # form the index -- the same hazard write_markdown documents and fixes
    # with observed=True for groupby.
    elements = elements.copy()
    if isinstance(elements["family"].dtype, pd.CategoricalDtype):
        elements["family"] = elements["family"].astype(str)

    wide = elements.pivot_table(
        index=ELEMENT_KEY, columns="variant", values="value", aggfunc="first"
    ).reset_index()

    for variant in DELTA_VARIANTS:
        if variant not in wide.columns:
            wide[variant] = pd.NA

    wide = wide.rename(columns={variant: f"value_{variant.lower()}"
                                for variant in DELTA_VARIANTS})

    for column, left, right in DELTA_SPECS:
        wide[column] = (wide[f"value_{left.lower()}"]
                        - wide[f"value_{right.lower()}"])

    return wide[ELEMENT_KEY + VALUE_COLUMNS + DELTA_COLUMNS]


def topology_status(elements: pd.DataFrame) -> pd.DataFrame:
    """Models whose reference report shares no element ids with variant A.

    Their scalars remain comparable; only the per-element join does not
    apply. Left unnamed, an empty join would read as "no differences".
    """
    if elements.empty:
        return pd.DataFrame(columns=["model_id", "status"])

    rows: list[dict] = []
    for model_id, group in elements.groupby("model_id"):
        ours = set(group.loc[group.variant == schema.VARIANT_A, "element_id"])
        theirs = set(group.loc[group.variant == schema.VARIANT_REF, "element_id"])
        if ours and theirs and not (ours & theirs):
            rows.append({
                "model_id": model_id,
                "status": schema.Status.REF_TOPOLOGY_MISMATCH,
            })
    return pd.DataFrame(rows, columns=["model_id", "status"])
