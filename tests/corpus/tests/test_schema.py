from swmmbench import schema


def test_status_vocabulary_is_exactly_the_specified_set():
    assert schema.Status.ALL == frozenset({
        "ok", "engine_error", "crash", "timeout",
        "parse_error", "no_ref", "ref_topology_mismatch",
    })


def test_run_variants_exclude_the_reference():
    assert schema.VARIANTS == ("A", "B", "C", "D", "E")
    assert schema.VARIANT_REF == "REF"
    assert schema.VARIANT_REF not in schema.VARIANTS


def test_iteration_kinds_are_distinct():
    assert schema.ITER_PICARD != schema.ITER_FV


# ---------------------------------------------------------------------------
# Case identity
# ---------------------------------------------------------------------------

BASE_CASE = ("EPA/m1", "A", "sha256:aaa", "git:1111")


def test_a_case_id_is_stable_for_the_same_inputs():
    assert schema.case_id(*BASE_CASE) == schema.case_id(*BASE_CASE)


def test_every_case_id_field_changes_the_result():
    # Each field is load-bearing: the variant because a model runs five ways,
    # the build because two executables can print one version string, and the
    # dependency because a deck's external data can change under a fixed deck.
    base = schema.case_id(*BASE_CASE)
    for index in range(len(BASE_CASE)):
        moved = list(BASE_CASE)
        moved[index] = "moved"
        assert schema.case_id(*moved) != base, schema.CASE_ID_FIELDS[index]


def test_case_id_fields_cannot_be_repartitioned_into_each_other():
    # A printable separator would let ("a|b", "c") and ("a", "b|c") hash
    # alike, so a model named after a variant could collide with another
    # case. The separator is NUL, which none of the fields can contain.
    assert schema.case_id("a", "b", "c", "d") != schema.case_id("ab", "", "c", "d")
    assert schema.case_id("a", "b", "c", "d") != schema.case_id("", "ab", "c", "d")
