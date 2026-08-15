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
