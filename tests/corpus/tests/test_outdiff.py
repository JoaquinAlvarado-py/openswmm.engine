from datetime import datetime, timedelta

import pytest

from swmmbench import outdiff

T0 = datetime(2004, 1, 1)


def _series(values):
    return {T0 + timedelta(minutes=5 * i): v for i, v in enumerate(values)}


def test_identical_series_have_no_divergence():
    result = outdiff.diff_series(_series([1.0, 2.0, 3.0]), _series([1.0, 2.0, 3.0]))

    assert result["max_abs"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["first_div_period"] is None
    assert result["n_periods"] == 3


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

    assert result["n_periods"] == 2
    assert result["max_abs"] == pytest.approx(0.0)


def test_empty_series_do_not_raise():
    result = outdiff.diff_series({}, {})

    assert result["n_periods"] == 0
    assert result["max_abs"] is None


def test_binding_backed_diff_is_skipped_without_the_binding(tmp_path):
    pytest.importorskip("openswmm.legacy.output")
    # With the binding present, a missing file must still not raise.
    rows = outdiff.diff_out_files(tmp_path / "a.out", tmp_path / "b.out", abs_tol=1e-6)
    assert rows == []
