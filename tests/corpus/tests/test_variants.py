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
