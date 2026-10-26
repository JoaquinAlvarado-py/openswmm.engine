import pytest

from swmmbench import rptparse

EMPTY = """
  *************************
  Highest Continuity Errors
  *************************
  None


  ***************************
  Time-Step Critical Elements
  ***************************
  None


  ********************************
  Highest Flow Instability Indexes
  ********************************
  All links are stable.


  *********************************
  Most Frequent Nonconverging Nodes
  *********************************
  Convergence obtained at all time steps.
"""

POPULATED = """
  *************************
  Highest Continuity Errors
  *************************
  Node J1 (-0.36%)


  ********************************
  Highest Flow Instability Indexes
  ********************************
  Link 8060 (1)
  Link 8040 (1)


  *********************************
  Most Frequent Nonconverging Nodes
  *********************************
  Node 10208 (3.68%)
  Node 82309 (3.40%)
"""


def test_empty_sentinels_yield_no_rows():
    assert rptparse.parse_rankings(EMPTY) == []


def test_nonconverging_nodes_are_extracted_with_percentages():
    rows = rptparse.parse_rankings(POPULATED)
    nonconv = [r for r in rows if r["metric"] == "nonconverging_pct"]

    assert len(nonconv) == 2
    assert nonconv[0] == {
        "element_type": "NODE",
        "element_id": "10208",
        "metric": "nonconverging_pct",
        "value": pytest.approx(3.68),
    }


def test_instability_index_is_a_bare_integer_not_a_percentage():
    rows = rptparse.parse_rankings(POPULATED)
    unstable = [r for r in rows if r["metric"] == "flow_instability_index"]

    assert len(unstable) == 2
    assert unstable[0]["element_type"] == "LINK"
    assert unstable[0]["element_id"] == "8060"
    assert unstable[0]["value"] == pytest.approx(1.0)


def test_negative_continuity_percentages_keep_their_sign():
    rows = rptparse.parse_rankings(POPULATED)
    (row,) = [r for r in rows if r["metric"] == "continuity_error_pct"]

    assert row["element_id"] == "J1"
    assert row["value"] == pytest.approx(-0.36)


def test_real_reference_report_reports_four_nonconverging_nodes(fixtures):
    rows = rptparse.parse_rankings(rptparse.read(fixtures / "extran1.rpt"))
    nonconv = [r for r in rows if r["metric"] == "nonconverging_pct"]

    assert len(nonconv) == 4
    assert {r["element_id"] for r in nonconv} == {"10208", "82309", "80608", "80408"}


def test_junk_input_yields_no_rows_rather_than_raising():
    assert rptparse.parse_rankings("\x00 garbage \n") == []


def test_malformed_ranking_values_do_not_raise_and_terminate_block():
    """Malformed values like '1.2.3' or '.' must not raise ValueError.

    Valid entries before the malformed one are returned; the malformed entry
    and anything after it are skipped as the block terminates.
    """
    malformed_input = """
  *************************
  Highest Continuity Errors
  *************************
  Node J1 (-0.36%)
  Node X (1.2.3%)
  Node Y (3.0%)


  *********************************
  Most Frequent Nonconverging Nodes
  *********************************
  Node M (2.5%)
  Node N (.)
  Node O (1.0%)
"""
    rows = rptparse.parse_rankings(malformed_input)

    # Verify no exception was raised and valid entries are returned
    assert len(rows) == 2

    # First valid entry (before malformed in continuity block)
    continuity = [r for r in rows if r["metric"] == "continuity_error_pct"]
    assert len(continuity) == 1
    assert continuity[0]["element_id"] == "J1"
    assert continuity[0]["value"] == pytest.approx(-0.36)

    # First valid entry in nonconverging block (before malformed)
    nonconv = [r for r in rows if r["metric"] == "nonconverging_pct"]
    assert len(nonconv) == 1
    assert nonconv[0]["element_id"] == "M"
    assert nonconv[0]["value"] == pytest.approx(2.5)
