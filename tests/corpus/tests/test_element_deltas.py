import pandas as pd
import pytest

from swmmbench import report, schema


def _elements(rows):
    return pd.DataFrame([
        {"model_id": "EPA/m1", "family": "EPA", "element_type": "NODE", **row}
        for row in rows
    ])


def test_element_deltas_pair_by_element_and_metric():
    frame = _elements([
        {"element_id": "J1", "metric": "node_max_depth", "variant": "A", "value": 4.0},
        {"element_id": "J1", "metric": "node_max_depth", "variant": "B", "value": 4.5},
        {"element_id": "J1", "metric": "node_max_depth", "variant": "C", "value": 3.7},
        {"element_id": "J1", "metric": "node_max_depth", "variant": "REF", "value": 3.9},
    ])

    (row,) = report.build_element_deltas(frame).to_dict("records")

    assert row["element_id"] == "J1"
    assert row["delta_b_minus_a"] == pytest.approx(0.5)
    assert row["delta_c_minus_a"] == pytest.approx(-0.3)
    assert row["delta_a_minus_ref"] == pytest.approx(0.1)


def test_element_deltas_carry_the_d_and_e_axes():
    # `D - C` at the element level is the same linear subtraction as at the
    # scalar level, and it is the axis variant D exists for -- an
    # element-level table that stopped at A/B/C would publish nothing from D
    # or E at all.
    frame = _elements([
        {"element_id": "J1", "metric": "node_max_depth", "variant": "A", "value": 4.0},
        {"element_id": "J1", "metric": "node_max_depth", "variant": "C", "value": 3.7},
        {"element_id": "J1", "metric": "node_max_depth", "variant": "D", "value": 3.2},
        {"element_id": "J1", "metric": "node_max_depth", "variant": "E", "value": 4.6},
    ])

    (row,) = report.build_element_deltas(frame).to_dict("records")

    assert row["value_d"] == pytest.approx(3.2)
    assert row["value_e"] == pytest.approx(4.6)
    assert row["delta_d_minus_c"] == pytest.approx(-0.5)
    assert row["delta_d_minus_a"] == pytest.approx(-0.8)
    assert row["delta_e_minus_a"] == pytest.approx(0.6)


def test_element_deltas_expose_every_value_and_delta_column():
    frame = _elements([
        {"element_id": "J1", "metric": "node_max_depth", "variant": "A", "value": 4.0},
    ])

    columns = set(report.build_element_deltas(frame).columns)

    assert set(report.VALUE_COLUMNS) <= columns
    assert set(report.DELTA_COLUMNS) <= columns


def test_elements_present_in_only_one_variant_yield_a_null_delta():
    frame = _elements([
        {"element_id": "J1", "metric": "node_max_depth", "variant": "A", "value": 4.0},
        {"element_id": "J2", "metric": "node_max_depth", "variant": "B", "value": 9.0},
    ])

    deltas = report.build_element_deltas(frame)

    assert set(deltas["element_id"]) == {"J1", "J2"}
    assert deltas["delta_b_minus_a"].isna().all()


def test_disjoint_reference_topology_is_flagged():
    frame = _elements([
        {"element_id": "J1", "metric": "node_max_depth", "variant": "A", "value": 4.0},
        {"element_id": "OTHER", "metric": "node_max_depth", "variant": "REF", "value": 3.0},
    ])

    status = report.topology_status(frame).set_index("model_id")

    assert status.loc["EPA/m1", "status"] == schema.Status.REF_TOPOLOGY_MISMATCH


def test_overlapping_topology_is_not_flagged():
    frame = _elements([
        {"element_id": "J1", "metric": "node_max_depth", "variant": "A", "value": 4.0},
        {"element_id": "J1", "metric": "node_max_depth", "variant": "REF", "value": 3.0},
    ])

    assert report.topology_status(frame).empty


def test_topology_status_ignores_variant_c_and_still_compares_a_to_ref():
    # C rows must not participate in the A-REF topology comparison in either
    # direction: neither masking a real mismatch nor manufacturing one.
    frame = _elements([
        {"element_id": "J1", "metric": "node_max_depth", "variant": "A", "value": 4.0},
        {"element_id": "J1", "metric": "node_max_depth", "variant": "C", "value": 3.7},
        {"element_id": "OTHER", "metric": "node_max_depth", "variant": "REF", "value": 3.0},
    ])

    status = report.topology_status(frame).set_index("model_id")

    assert status.loc["EPA/m1", "status"] == schema.Status.REF_TOPOLOGY_MISMATCH


def test_a_model_with_no_reference_elements_is_not_flagged():
    frame = _elements([
        {"element_id": "J1", "metric": "node_max_depth", "variant": "A", "value": 4.0},
        {"element_id": "J1", "metric": "node_max_depth", "variant": "B", "value": 4.0},
    ])

    assert report.topology_status(frame).empty


def test_empty_elements_produce_empty_results():
    assert report.build_element_deltas(pd.DataFrame()).empty
    assert report.topology_status(pd.DataFrame()).empty
