"""Vocabulary shared by every stage of the harness.

Centralised so that a status string or column name is spelled exactly one
way across the pipeline; a typo in a stage that writes Parquet is otherwise
invisible until the analysis stage silently drops rows.
"""

import hashlib


class Status:
    """Outcome of a single (model, variant) run. Every run gets exactly one."""

    OK = "ok"
    ENGINE_ERROR = "engine_error"      # non-zero exit, but a .rpt was produced
    CRASH = "crash"                    # signal or abort, no usable .rpt
    TIMEOUT = "timeout"                # exceeded the per-model wall limit
    PARSE_ERROR = "parse_error"        # .rpt produced but not parseable
    NO_REF = "no_ref"                  # no external anchor for this model
    REF_TOPOLOGY_MISMATCH = "ref_topology_mismatch"

    ALL = frozenset({
        OK, ENGINE_ERROR, CRASH, TIMEOUT,
        PARSE_ERROR, NO_REF, REF_TOPOLOGY_MISMATCH,
    })


VARIANT_A = "A"      # our engine, new options explicitly off
VARIANT_B = "B"      # our engine, Anderson acceleration on (EXPLICIT/EXTRAN)
VARIANT_C = "C"      # our engine, semi-implicit (Crank-Nicolson) node continuity on
VARIANT_D = "D"      # our engine, Anderson acceleration on SEMI_IMPLICIT/EXTRAN
VARIANT_E = "E"      # our engine, Dynamic Preissmann Slot surcharge method
VARIANT_REF = "REF"  # the corpus .rpt, parsed but never re-run
VARIANTS = (VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D, VARIANT_E)

# Which counter the "Average ... per Step" line actually reports. The report
# writer relabels it under FV routing (DefaultReportPlugin.cpp:1194-1203) and
# explicit substeps are not Picard iterations, so they must never share a
# column without this discriminator.
#
# The discriminator is NOT derivable from the label alone: the unrelabelled
# "Average Iterations per Step" is printed under KINWAVE and STEADY too,
# where the counter has no Picard meaning. `iteration_metric_kind` is
# therefore left None for those runs -- see rptparse.parse_scalars.
ITER_PICARD = "picard"
ITER_FV = "fv_substeps"

#: The value `Flow Routing Method` echoes for the dynamic wave solver -- the
#: only routing model under which a Picard iteration count means anything.
ROUTING_DYNWAVE = "DYNWAVE"


# ---------------------------------------------------------------------------
# Case identity
# ---------------------------------------------------------------------------

#: The fields a `case_id` is derived from, in order. A *case* is one unit of
#: work: this model, under this variant, executed by THIS engine build,
#: against THIS state of the corpus.
#:
#: `runs` used to be the only table carrying any of this (`engine_version` +
#: `inp_sha256`), while `scalars` and `elements` were identified by
#: model/family/variant/engine_version alone. So a model whose inputs changed
#: left old and new records coexisting in those tables with nothing to tell
#: them apart, and `pivot_table(..., aggfunc="first")` could silently pick
#: the stale one. One id, derived here and only here, carried on every row of
#: `runs`, `scalars`, `elements`, `ts_diff`, `deltas` and `element_deltas`,
#: makes that impossible: the tables cannot disagree about what a case is
#: because they never compute it themselves.
#:
#: `options_id` is the variant's RESOLVED option set, not its letter. The
#: letter names a unit of work only for as long as `variants.OPTIONS[letter]`
#: is unchanged, and this project has edited that table several times: a `B`
#: run recorded before an edit and a `B` run after it are two materially
#: different units of work that would otherwise share one id, so the second
#: would read as "already done" and never run. `options_applied` was written
#: to `runs` but never hashed, which made the drift visible after the fact
#: and preventable never.
CASE_ID_FIELDS = ("model_id", "variant", "engine_build_id", "dependency_id",
                  "options_id")

#: NUL joins the fields because it cannot occur in any of them. A printable
#: separator would let two different field tuples serialise identically --
#: ("a|b", "c") and ("a", "b|c") -- and hash to the same case.
_CASE_ID_SEPARATOR = "\x00"


def case_id(
    model_id: str,
    variant: str,
    engine_build_id: str,
    dependency_id: str,
    options_id: str,
) -> str:
    """The identity of one unit of work; see `CASE_ID_FIELDS`.

    `engine_build_id` is a content hash of the engine executable, not the
    version string it prints (two builds can print the same string);
    `dependency_id` is the corpus git commit rather than the deck hash (a
    deck's `DataFiles/*.dat` can change while the deck does not), falling
    back to the sentinel plus that deck hash when the corpus is unpinned; and
    `options_id` is a hash of the variant's resolved option set rather than
    its letter. All three are supplied by the caller; this function only
    fixes how they are combined, so the resume key and every table's
    `case_id` column are the same value by construction rather than by
    convention.

    What is deliberately NOT here: `timeout_s`. See the note in `cli` beside
    `RESUME_KEY`.
    """
    fields = (str(model_id), str(variant), str(engine_build_id),
              str(dependency_id), str(options_id))
    payload = _CASE_ID_SEPARATOR.join(fields).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
