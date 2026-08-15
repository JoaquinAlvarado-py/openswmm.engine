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
