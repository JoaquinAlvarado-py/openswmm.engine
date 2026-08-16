from datetime import datetime, timedelta

import pytest

from swmmbench import outdiff

T0 = datetime(2004, 1, 1)


def _series(values, step_minutes=5, start_minutes=0):
    return {T0 + timedelta(minutes=start_minutes + step_minutes * i): v
            for i, v in enumerate(values)}


def test_identical_series_have_no_divergence():
    result = outdiff.diff_series(_series([1.0, 2.0, 3.0]), _series([1.0, 2.0, 3.0]))

    assert result["max_abs"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["first_div_period"] is None
    assert result["n_common"] == 3


def test_maximum_absolute_difference_is_reported():
    result = outdiff.diff_series(_series([1.0, 2.0, 3.0]), _series([1.0, 2.0, 9.0]))

    assert result["max_abs"] == pytest.approx(6.0)


def test_relative_difference_is_scaled_by_the_baseline():
    result = outdiff.diff_series(_series([10.0]), _series([11.0]))

    assert result["max_rel"] == pytest.approx(0.1)


def test_relative_difference_is_finite_when_the_baseline_is_zero():
    result = outdiff.diff_series(_series([0.0]), _series([5.0]))

    # Baseline is zero, so the scale falls back to 1.0: the ratio is the raw
    # delta, not inf/nan.
    assert result["max_rel"] == pytest.approx(5.0)


def test_first_divergence_is_the_earliest_period_past_tolerance():
    result = outdiff.diff_series(
        _series([1.0, 1.0, 1.0, 1.0]),
        _series([1.0, 1.0, 5.0, 9.0]),
        abs_tol=1e-6,
    )

    assert result["first_div_period"] == 2
    assert result["first_div_time"] == T0 + timedelta(minutes=10)


def test_differing_lengths_compare_the_common_prefix():
    result = outdiff.diff_series(_series([1.0, 2.0, 3.0]), _series([1.0, 2.0]))

    assert result["n_common"] == 2
    assert result["max_abs"] == pytest.approx(0.0)


def test_empty_series_do_not_raise():
    result = outdiff.diff_series({}, {})

    assert result["n_common"] == 0
    assert result["max_abs"] is None


# ---------------------------------------------------------------------------
# Coverage: what the divergence statistics were actually computed OVER
# ---------------------------------------------------------------------------


def test_identical_series_report_full_coverage_and_matching_grids():
    result = outdiff.diff_series(_series([1.0, 2.0, 3.0]), _series([1.0, 2.0, 3.0]))

    assert result["n_periods_left"] == 3
    assert result["n_periods_right"] == 3
    assert result["n_common"] == 3
    assert result["coverage_fraction"] == pytest.approx(1.0)
    assert result["start_time_match"] is True
    assert result["end_time_match"] is True
    assert result["time_grid_match"] is True


def test_a_truncated_right_hand_series_is_visible_as_incomplete_coverage():
    # The failure this evidence exists for: B stopped halfway and agreed
    # perfectly over the half it managed, so `max_abs` alone reads as an
    # excellent result. Only the coverage fields say otherwise.
    result = outdiff.diff_series(
        _series([1.0, 2.0, 3.0, 4.0]), _series([1.0, 2.0]))

    assert result["max_abs"] == pytest.approx(0.0)   # looks perfect...
    assert result["n_periods_left"] == 4
    assert result["n_periods_right"] == 2
    assert result["n_common"] == 2
    # ... over half the run: 2 shared of 4 distinct timestamps in either.
    assert result["coverage_fraction"] == pytest.approx(0.5)
    assert result["coverage_fraction"] < 1.0
    assert result["start_time_match"] is True
    assert result["end_time_match"] is False
    assert result["time_grid_match"] is False


def test_the_same_span_sampled_differently_is_not_the_same_grid():
    # 0..20 min at 5-minute steps against 0..20 min at 10-minute steps: both
    # endpoints coincide, so start/end matching alone would call this a full
    # comparison. `time_grid_match` is what tells the two cases apart.
    left = _series([1.0, 2.0, 3.0, 4.0, 5.0], step_minutes=5)
    right = _series([1.0, 3.0, 5.0], step_minutes=10)

    result = outdiff.diff_series(left, right)

    assert result["start_time_match"] is True
    assert result["end_time_match"] is True
    assert result["time_grid_match"] is False
    # 3 shared of 5 distinct timestamps across both series.
    assert result["coverage_fraction"] == pytest.approx(0.6)
    assert result["n_common"] == 3


def test_coverage_fraction_denominator_is_the_union_not_either_side():
    # Neither side is a subset of the other: A covers 0-20 min, B covers
    # 10-30 min, both at 5-minute steps. 3 shared, 7 distinct in either.
    left = _series([1.0] * 5, step_minutes=5)
    right = _series([1.0] * 5, step_minutes=5, start_minutes=10)

    result = outdiff.diff_series(left, right)

    assert result["n_periods_left"] == 5
    assert result["n_periods_right"] == 5
    assert result["n_common"] == 3
    assert result["coverage_fraction"] == pytest.approx(3 / 7)
    assert result["start_time_match"] is False
    assert result["end_time_match"] is False


def test_disjoint_series_report_zero_coverage_rather_than_no_coverage():
    left = _series([1.0, 2.0], step_minutes=5)
    right = _series([1.0, 2.0], step_minutes=5, start_minutes=100)

    result = outdiff.diff_series(left, right)

    assert result["n_common"] == 0
    assert result["coverage_fraction"] == pytest.approx(0.0)
    assert result["time_grid_match"] is False
    assert result["max_abs"] is None


def test_two_empty_series_have_no_coverage_to_report_rather_than_zero():
    # 0/0 is not "zero coverage": there is no grid to compare at all, and
    # calling it 0.0 would file an absent comparison as a truncation.
    result = outdiff.diff_series({}, {})

    assert result["n_periods_left"] == 0
    assert result["n_periods_right"] == 0
    assert result["coverage_fraction"] is None
    assert result["start_time_match"] is None
    assert result["end_time_match"] is None
    assert result["time_grid_match"] is None


def test_every_coverage_field_is_present_on_an_empty_and_a_full_result():
    # The field set must not depend on whether the overlap was empty; a
    # missing key would break the ts_diff schema for a whole batch.
    for a, b in (({}, {}), (_series([1.0]), _series([1.0])),
                 (_series([1.0, 2.0]), _series([1.0]))):
        result = outdiff.diff_series(a, b)
        assert set(outdiff.COVERAGE_FIELDS) <= set(result)


def test_binding_backed_diff_is_skipped_without_the_binding(tmp_path):
    pytest.importorskip("openswmm.legacy.output")
    # With the binding present, a missing file must still not raise.
    rows = outdiff.diff_out_files(tmp_path / "a.out", tmp_path / "b.out", abs_tol=1e-6)
    assert rows == []
