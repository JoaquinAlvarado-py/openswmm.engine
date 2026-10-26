import pytest

from swmmbench import schema, variants

DECK = """[TITLE]
Test deck

[OPTIONS]
FLOW_UNITS           CFS
ROUTING_STEP         0:00:20

[JUNCTIONS]
J1  10  0
"""


def test_baseline_states_the_new_options_explicitly_off():
    result = variants.apply_options(DECK, variants.OPTIONS[schema.VARIANT_A])

    assert "ANDERSON_ACCEL       NO" in result


def test_feature_variant_turns_anderson_on():
    result = variants.apply_options(DECK, variants.OPTIONS[schema.VARIANT_B])

    assert "ANDERSON_ACCEL       YES" in result
    assert "ANDERSON_ACCEL       NO" not in result


def test_a_and_b_both_state_node_continuity_explicit():
    result_a = variants.apply_options(DECK, variants.OPTIONS[schema.VARIANT_A])
    result_b = variants.apply_options(DECK, variants.OPTIONS[schema.VARIANT_B])

    assert "NODE_CONTINUITY      EXPLICIT" in result_a
    assert "NODE_CONTINUITY      EXPLICIT" in result_b


def test_variant_c_turns_on_semi_implicit_continuity_and_keeps_anderson_off():
    result = variants.apply_options(DECK, variants.OPTIONS[schema.VARIANT_C])

    assert "NODE_CONTINUITY      SEMI_IMPLICIT" in result
    assert "ANDERSON_ACCEL       NO" in result


def test_existing_key_is_overwritten_not_duplicated():
    deck = DECK.replace("[OPTIONS]\n", "[OPTIONS]\nANDERSON_ACCEL       NO\n")

    result = variants.apply_options(deck, {"ANDERSON_ACCEL": "YES"})

    assert result.count("ANDERSON_ACCEL") == 1
    assert "ANDERSON_ACCEL       YES" in result


def test_untouched_options_survive():
    result = variants.apply_options(DECK, {"ANDERSON_ACCEL": "YES"})

    assert "FLOW_UNITS           CFS" in result
    assert "ROUTING_STEP         0:00:20" in result


def test_deck_without_an_options_section_gets_one():
    deck = "[TITLE]\nNo options here\n\n[JUNCTIONS]\nJ1  10  0\n"

    result = variants.apply_options(deck, {"ANDERSON_ACCEL": "YES"})

    assert "[OPTIONS]" in result
    assert "ANDERSON_ACCEL       YES" in result
    assert "[JUNCTIONS]" in result


def test_section_header_matching_is_case_insensitive():
    deck = DECK.replace("[OPTIONS]", "[Options]")

    result = variants.apply_options(deck, {"ANDERSON_ACCEL": "YES"})

    assert result.count("ANDERSON_ACCEL") == 1
    assert "[Options]" in result


def test_application_is_idempotent():
    once = variants.apply_options(DECK, variants.OPTIONS[schema.VARIANT_B])
    twice = variants.apply_options(once, variants.OPTIONS[schema.VARIANT_B])

    assert once == twice


def test_temp_deck_lands_beside_the_original_and_is_removed(tmp_path):
    original = tmp_path / "model.inp"
    original.write_text(DECK, encoding="latin-1")

    with variants.temp_deck(original, schema.VARIANT_B) as deck:
        assert deck.parent == original.parent
        assert deck.exists()
        assert "ANDERSON_ACCEL       YES" in deck.read_text(encoding="latin-1")

    assert not deck.exists()
    assert original.read_text(encoding="latin-1") == DECK


def test_temp_deck_is_removed_even_when_the_body_raises(tmp_path):
    original = tmp_path / "model.inp"
    original.write_text(DECK, encoding="latin-1")

    with pytest.raises(RuntimeError):
        with variants.temp_deck(original, schema.VARIANT_A) as deck:
            captured = deck
            raise RuntimeError("simulated crash")

    assert not captured.exists()


# ---------------------------------------------------------------------------
# The five-variant matrix (A, B, C, D, E)
# ---------------------------------------------------------------------------

#: The table from the spec: every variant's exact, explicit value for every
#: option under study.
EXPECTED_TABLE = {
    schema.VARIANT_A: {"ANDERSON_ACCEL": "NO", "NODE_CONTINUITY": "EXPLICIT",
                       "SURCHARGE_METHOD": "EXTRAN"},
    schema.VARIANT_B: {"ANDERSON_ACCEL": "YES", "NODE_CONTINUITY": "EXPLICIT",
                       "SURCHARGE_METHOD": "EXTRAN"},
    schema.VARIANT_C: {"ANDERSON_ACCEL": "NO", "NODE_CONTINUITY": "SEMI_IMPLICIT",
                       "SURCHARGE_METHOD": "EXTRAN"},
    schema.VARIANT_D: {"ANDERSON_ACCEL": "YES", "NODE_CONTINUITY": "SEMI_IMPLICIT",
                       "SURCHARGE_METHOD": "EXTRAN"},
    schema.VARIANT_E: {"ANDERSON_ACCEL": "NO", "NODE_CONTINUITY": "EXPLICIT",
                       "SURCHARGE_METHOD": "DYNAMIC_SLOT"},
}


def test_every_variant_states_every_studied_option_explicitly_with_the_exact_value():
    for variant, expected in EXPECTED_TABLE.items():
        stated = variants.OPTIONS[variant]
        for option, value in expected.items():
            assert stated[option] == value, (variant, option)


def test_d_differs_from_c_only_in_anderson_accel():
    c = variants.OPTIONS[schema.VARIANT_C]
    d = variants.OPTIONS[schema.VARIANT_D]

    # Equal key sets first: `for k in c` alone would silently ignore a key
    # present in D but absent from C, so a stray extra option in D would
    # read as "differs only in ANDERSON_ACCEL".
    assert set(c) == set(d)

    differing = {k for k in c if c[k] != d[k]}

    assert differing == {"ANDERSON_ACCEL"}
    assert c["ANDERSON_ACCEL"] == "NO"
    assert d["ANDERSON_ACCEL"] == "YES"


def test_e_differs_from_a_only_in_surcharge_method():
    a = variants.OPTIONS[schema.VARIANT_A]
    e = variants.OPTIONS[schema.VARIANT_E]

    assert set(a) == set(e)

    differing = {k for k in a if a[k] != e[k]}

    assert differing == {"SURCHARGE_METHOD"}
    assert a["SURCHARGE_METHOD"] == "EXTRAN"
    assert e["SURCHARGE_METHOD"] == "DYNAMIC_SLOT"


def test_shared_pins_are_identical_and_present_across_all_five_variants():
    # DPS_* is inert under EXTRAN (A, B, C, D) -- pinned everywhere anyway so
    # a future default change cannot silently drift E's baseline.
    shared_keys = ("VIRTUAL_JUNCTION_MOMENTUM", "DPS_CELERITY",
                  "DPS_ALPHA", "DPS_DECAY_TIME")
    for key in shared_keys:
        values = {variants.OPTIONS[v][key] for v in schema.VARIANTS}
        assert len(values) == 1, (key, values)

    assert variants.OPTIONS[schema.VARIANT_A]["VIRTUAL_JUNCTION_MOMENTUM"] == "BASIC"
    assert variants.OPTIONS[schema.VARIANT_A]["DPS_CELERITY"] == "25.0"
    assert variants.OPTIONS[schema.VARIANT_A]["DPS_ALPHA"] == "3.0"
    assert variants.OPTIONS[schema.VARIANT_A]["DPS_DECAY_TIME"] == "0.5"


def test_variant_e_deck_states_dynamic_slot():
    result = variants.apply_options(DECK, variants.OPTIONS[schema.VARIANT_E])

    assert "SURCHARGE_METHOD     DYNAMIC_SLOT" in result


def test_variant_d_deck_states_anderson_on_and_semi_implicit():
    result = variants.apply_options(DECK, variants.OPTIONS[schema.VARIANT_D])

    assert "ANDERSON_ACCEL       YES" in result
    assert "NODE_CONTINUITY      SEMI_IMPLICIT" in result
