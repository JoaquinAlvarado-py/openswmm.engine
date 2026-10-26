import pytest

from swmmbench import rptparse


def test_node_depth_summary_rows_become_metrics(fixtures):
    rows = rptparse.parse_tables(rptparse.read(fixtures / "extran1.rpt"))
    by_key = {(r["element_id"], r["metric"]): r["value"] for r in rows}

    assert by_key[("82309", "node_max_depth")] == pytest.approx(22.32)
    assert by_key[("82309", "node_avg_depth")] == pytest.approx(7.66)
    assert by_key[("82309", "node_max_hgl")] == pytest.approx(134.62)


def test_link_flow_summary_rows_become_metrics(fixtures):
    rows = rptparse.parse_tables(rptparse.read(fixtures / "extran1.rpt"))
    by_key = {(r["element_id"], r["metric"]): r["value"] for r in rows}

    assert by_key[("1030", "link_max_flow")] == pytest.approx(122.35)
    assert by_key[("1030", "link_max_velocity")] == pytest.approx(5.84)
    assert by_key[("1602", "link_max_full_flow")] == pytest.approx(1.66)


def test_element_types_are_tagged_per_table(fixtures):
    rows = rptparse.parse_tables(rptparse.read(fixtures / "extran1.rpt"))

    assert {r["element_type"] for r in rows if r["metric"].startswith("node_")} == {"NODE"}
    assert {r["element_type"] for r in rows if r["metric"].startswith("link_")} == {"LINK"}


def test_time_of_max_occupies_two_tokens_and_does_not_shift_columns(fixtures):
    # If `0  00:22` were treated as one token, max_velocity would pick up the
    # clock string and the row would be dropped or mis-valued.
    rows = rptparse.parse_tables(rptparse.read(fixtures / "extran1.rpt"))
    by_key = {(r["element_id"], r["metric"]): r["value"] for r in rows}

    assert by_key[("8040", "link_max_velocity")] == pytest.approx(7.13)


def test_malformed_rows_are_skipped_not_fatal():
    text = """
  ******************
  Node Depth Summary
  ******************

  ---------------------------------------------------------------------------------
                                 Average  Maximum  Maximum  Time of Max    Reported
                                   Depth    Depth      HGL   Occurrence   Max Depth
  Node                 Type         Feet     Feet     Feet  days hr:min        Feet
  ---------------------------------------------------------------------------------
  GOOD                 JUNCTION     4.11    13.40   138.00     0  00:38       12.75
  TRUNCATED            JUNCTION     4.11
"""
    rows = rptparse.parse_tables(text)

    assert {r["element_id"] for r in rows} == {"GOOD"}


def test_absent_tables_yield_no_rows():
    assert rptparse.parse_tables("  nothing here\n") == []
