import pytest

from swmmbench import rptparse, schema


@pytest.fixture
def extran1(fixtures):
    return rptparse.read(fixtures / "extran1.rpt")


@pytest.fixture
def fv(fixtures):
    return rptparse.read(fixtures / "fv_substeps.rpt")


@pytest.fixture
def multi_continuity(fixtures):
    return rptparse.read(fixtures / "multi_continuity.rpt")


def test_reads_the_reported_engine_version(extran1):
    scalars = rptparse.parse_scalars(extran1)

    assert "VERSION 5.2" in scalars["reported_version"]


def test_reads_flow_routing_continuity_error(extran1):
    scalars = rptparse.parse_scalars(extran1)

    assert scalars["continuity_error_flow"] == pytest.approx(0.011)


def test_absent_continuity_blocks_are_none_not_zero(extran1):
    scalars = rptparse.parse_scalars(extran1)

    # extran1 is routing-only: no runoff, no quality, no 2D.
    assert scalars["continuity_error_runoff"] is None
    assert scalars["continuity_error_quality"] is None
    assert scalars["continuity_error_2d"] is None


def test_reads_time_step_summary(extran1):
    scalars = rptparse.parse_scalars(extran1)

    assert scalars["min_step"] == pytest.approx(20.0)
    assert scalars["avg_step"] == pytest.approx(20.0)
    assert scalars["max_step"] == pytest.approx(20.0)
    assert scalars["pct_steady_state"] == pytest.approx(0.0)


def test_picard_iterations_are_labelled_as_such(extran1):
    scalars = rptparse.parse_scalars(extran1)

    assert scalars["avg_iterations_per_step"] == pytest.approx(2.37)
    assert scalars["iteration_metric_kind"] == schema.ITER_PICARD
    assert scalars["pct_steps_not_converging"] == pytest.approx(3.68)


def test_fv_substeps_are_not_reported_as_picard_iterations(fv):
    scalars = rptparse.parse_scalars(fv)

    assert scalars["avg_iterations_per_step"] == pytest.approx(142.0)
    assert scalars["iteration_metric_kind"] == schema.ITER_FV


def test_a_kinwave_iteration_count_is_not_labelled_picard():
    # The engine prints `Average Iterations per Step` for EVERY routing
    # model and relabels it only under FV, so the label alone cannot mean
    # Picard. Claiming `picard` here would be worse than claiming nothing:
    # report.build_deltas's kind guard only trips when the two sides
    # DISAGREE, so two KINWAVE runs would agree on a meaningless label and
    # their difference would be published as an iteration shift.
    report_text = """
  ****************
  Analysis Options
  ****************
  Flow Routing Method ...... KINWAVE

  *************************
  Routing Time Step Summary
  *************************
  Average Time Step           :    10.00 sec
  Average Iterations per Step :     2.50
"""
    scalars = rptparse.parse_scalars(report_text)

    # The number is still recorded -- it is a real thing the engine printed.
    assert scalars["avg_iterations_per_step"] == pytest.approx(2.50)
    assert scalars["iteration_metric_kind"] is None


def test_a_steady_iteration_count_is_not_labelled_picard():
    report_text = """
  Flow Routing Method ...... STEADY
  Average Iterations per Step :     1.00
"""
    assert rptparse.parse_scalars(report_text)["iteration_metric_kind"] is None


def test_an_iteration_count_with_no_routing_echo_is_not_labelled_picard():
    # No echo means no evidence the counter is Picard-shaped. Unlike a
    # missing kind in the report layer, this absence is NOT given the
    # benefit of the doubt.
    report_text = "  Average Iterations per Step :     2.50\n"

    assert rptparse.parse_scalars(report_text)["iteration_metric_kind"] is None


def test_not_applicable_convergence_parses_to_none_never_zero(fv):
    scalars = rptparse.parse_scalars(fv)

    assert scalars["pct_steps_not_converging"] is None


def test_duration_comes_from_the_start_and_end_dates(extran1):
    scalars = rptparse.parse_scalars(extran1)

    assert scalars["duration_s"] == pytest.approx(8 * 3600)


def test_step_count_and_total_iterations_are_derived_estimates(extran1):
    scalars = rptparse.parse_scalars(extran1)

    assert scalars["n_steps_est"] == pytest.approx(8 * 3600 / 20.0)
    assert scalars["total_iterations_est"] == pytest.approx(1440 * 2.37)


def test_estimates_are_none_when_the_average_step_is_missing():
    scalars = rptparse.parse_scalars("  Nothing useful here\n")

    assert scalars["n_steps_est"] is None
    assert scalars["total_iterations_est"] is None


def test_parsing_junk_yields_nones_rather_than_raising():
    scalars = rptparse.parse_scalars("\x00 not a report at all \n")

    assert all(
        scalars[key] is None
        for key in ("avg_step", "avg_iterations_per_step", "continuity_error_flow")
    )


def test_extracts_all_three_continuity_blocks_from_multi_block_report(multi_continuity):
    scalars = rptparse.parse_scalars(multi_continuity)

    assert scalars["continuity_error_runoff"] == pytest.approx(0.000)
    assert scalars["continuity_error_flow"] == pytest.approx(-1.051)
    assert scalars["continuity_error_2d"] == pytest.approx(-0.507)


def test_quality_continuity_is_none_when_absent_in_multi_block_report(multi_continuity):
    scalars = rptparse.parse_scalars(multi_continuity)

    assert scalars["continuity_error_quality"] is None


def test_reads_the_reported_node_continuity_and_anderson_acceleration():
    # DefaultReportPlugin.cpp echoes both resolved option values back into
    # the report. OptionsHandler.cpp silently ignores an unrecognised
    # NODE_CONTINUITY value (no else/warning), so `options_applied` alone
    # (what the harness asked for) cannot prove the engine actually did it;
    # this is the only way to check what the engine actually resolved to.
    report_text = """
  ****************
  Analysis Options
  ****************
  Flow Routing Method ...... DYNWAVE
  Surcharge Method ......... EXTRAN
  Node Continuity .......... SEMI_IMPLICIT
  Anderson Acceleration .... YES
"""
    scalars = rptparse.parse_scalars(report_text)

    assert scalars["reported_node_continuity"] == "SEMI_IMPLICIT"
    assert scalars["reported_anderson_accel"] == "YES"


def test_reported_node_continuity_and_anderson_are_none_when_absent():
    # FV routing (or STEADY/KINWAVE) never prints this block at all
    # (DefaultReportPlugin.cpp gates it on the DYNWAVE routing method).
    scalars = rptparse.parse_scalars("  Nothing useful here\n")

    assert scalars["reported_node_continuity"] is None
    assert scalars["reported_anderson_accel"] is None


def test_reads_the_reported_routing_model_and_surcharge_method_from_extran1(extran1):
    # extran1.rpt is a real EPA SWMM 5.2 report (Example 1 of the Extran
    # Manual). It carries `Flow Routing Method ...... DYNWAVE` and
    # `Surcharge Method ......... EXTRAN` but -- being genuine EPA SWMM
    # output, not OpenSWMM's -- it does NOT carry `Node Continuity` or
    # `Anderson Acceleration`, which are OpenSWMM-only echoes.
    scalars = rptparse.parse_scalars(extran1)

    assert scalars["reported_routing_model"] == "DYNWAVE"
    assert scalars["reported_surcharge_method"] == "EXTRAN"
    assert scalars["reported_node_continuity"] is None
    assert scalars["reported_anderson_accel"] is None


def test_reads_the_reported_routing_model_and_surcharge_method_from_multi_continuity(
    multi_continuity,
):
    scalars = rptparse.parse_scalars(multi_continuity)

    assert scalars["reported_routing_model"] == "DYNWAVE"
    assert scalars["reported_surcharge_method"] == "EXTRAN"
    assert scalars["reported_node_continuity"] == "EXPLICIT"
    assert scalars["reported_anderson_accel"] == "NO"


def test_non_dynwave_report_yields_routing_model_but_none_for_the_other_three():
    # Flow Routing Method is printed for EVERY routing model
    # (DefaultReportPlugin.cpp:556); Surcharge Method, Node Continuity and
    # Anderson Acceleration sit inside the DYNWAVE-only `if (rm == 2)` block
    # (lines 558-567) and are legitimately absent for KINWAVE.
    report_text = """
  ****************
  Analysis Options
  ****************
  Flow Routing Method ...... KINWAVE
"""
    scalars = rptparse.parse_scalars(report_text)

    assert scalars["reported_routing_model"] == "KINWAVE"
    assert scalars["reported_surcharge_method"] is None
    assert scalars["reported_node_continuity"] is None
    assert scalars["reported_anderson_accel"] is None


def test_malformed_continuity_error_value_does_not_raise():
    """Malformed continuity error values like '1.2.3' or '.' must not raise.

    If a line matches the pattern but has a malformed value, the parser should
    continue looking for the next valid continuity error line.
    """
    report_with_malformed_continuity = """
  Flow Routing Continuity
  ***********************
  Continuity Error (%) .............. 1.2.3
  There are no discernible errors.

  Flow Routing Continuity
  ***********************
  Continuity Error (%) .............. -0.542
  There are no discernible errors.
"""
    # Should not raise ValueError when parsing malformed "1.2.3"
    scalars = rptparse.parse_scalars(report_with_malformed_continuity)

    # The parser should have skipped the malformed value and found the valid one
    assert scalars["continuity_error_flow"] == pytest.approx(-0.542)
