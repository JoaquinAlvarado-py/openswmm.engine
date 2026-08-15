from swmmbench import schema


def test_status_vocabulary_is_exactly_the_specified_set():
    assert schema.Status.ALL == frozenset({
        "ok", "engine_error", "crash", "timeout",
        "parse_error", "no_ref", "ref_topology_mismatch",
    })


def test_run_variants_exclude_the_reference():
    assert schema.VARIANTS == ("A", "B", "C")
    assert schema.VARIANT_REF == "REF"
    assert schema.VARIANT_REF not in schema.VARIANTS


def test_iteration_kinds_are_distinct():
    assert schema.ITER_PICARD != schema.ITER_FV
