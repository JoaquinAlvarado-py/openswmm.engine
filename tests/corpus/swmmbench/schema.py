"""Vocabulary shared by every stage of the harness.

Centralised so that a status string or column name is spelled exactly one
way across the pipeline; a typo in a stage that writes Parquet is otherwise
invisible until the analysis stage silently drops rows.
"""


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
VARIANT_B = "B"      # our engine, Anderson + continuity options on
VARIANT_REF = "REF"  # the corpus .rpt, parsed but never re-run
VARIANTS = (VARIANT_A, VARIANT_B)

# Which counter the "Average ... per Step" line actually reports. The report
# writer relabels it under FV routing (DefaultReportPlugin.cpp:1194-1203) and
# explicit substeps are not Picard iterations, so they must never share a
# column without this discriminator.
ITER_PICARD = "picard"
ITER_FV = "fv_substeps"
