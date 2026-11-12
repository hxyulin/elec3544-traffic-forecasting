"""Subgroup weighting, missing targets, and saved-prediction alignment."""

import numpy as np
import pytest

from traffic.analysis import grouped_errors, load_predictions, verify_partition
from traffic.metrics import masked_errors


def test_group_scores_recompose_with_unequal_counts_and_empty_groups():
    true = np.array([[10.0, 0.0, 20.0], [30.0, 40.0, 0.0]])
    pred = np.array([[11.0, 999.0, 17.0], [35.0, 38.0, 999.0]])
    rows = grouped_errors(pred, true, np.array([[0, 1, 2]]), 4)
    assert [r["n"] for r in rows] == [2, 1, 1, 0]
    assert rows[0]["mae"] == 3
    assert rows[0]["rmse"] == pytest.approx(np.sqrt(13))
    assert np.isnan(rows[3]["mae"])
    verify_partition(rows, masked_errors(pred, true))
    # A mean over sensor MAEs would give 8/3, rather than the pooled 11/4.
    assert masked_errors(pred, true)["mae"] == 2.75
    by_window = grouped_errors(pred, true, np.array([[0], [1]]), 2)
    verify_partition(by_window, masked_errors(pred, true))


def test_empty_and_invalid_partitions():
    rows = grouped_errors(
        np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 1), dtype=int), 2
    )
    assert all(r["n"] == 0 and np.isnan(r["rmse"]) for r in rows)
    verify_partition(rows, {"n": 0})
    with pytest.raises(ValueError, match="integers"):
        grouped_errors(np.ones((2, 2)), np.ones((2, 2)), np.array([[0, 2]]), 2)
    with pytest.raises(ValueError, match="partition"):
        verify_partition([{"n": 1, "mae": 1, "rmse": 1, "mape": 1}], {"n": 2})


def test_saved_forecasts_reject_misalignment_and_invalid_values(tmp_path):
    path = tmp_path / "test.npz"
    starts = np.array([10, 11])
    horizons = np.array([3, 6, 12])
    pred = np.ones((2, 3, 2))
    np.savez(path, starts=starts, horizons=horizons, pred=pred)
    np.testing.assert_array_equal(
        load_predictions(path, starts, horizons, pred.shape), pred
    )
    with pytest.raises(AssertionError):
        load_predictions(path, starts[::-1], horizons, pred.shape)
    with pytest.raises(AssertionError):
        load_predictions(path, starts, horizons[::-1], pred.shape)
    with pytest.raises(ValueError, match="shape"):
        load_predictions(path, starts, horizons, (2, 3, 3))
    pred[0, 0, 0] = np.nan
    np.savez(path, starts=starts, horizons=horizons, pred=pred)
    with pytest.raises(ValueError, match="values"):
        load_predictions(path, starts, horizons, pred.shape)


def test_residual_summary_masks_missing_targets_and_keeps_additive_error():
    from traffic.analysis import residual_summary

    result = residual_summary(
        np.array([8.0, 40.0, 999.0, 29.0]), np.array([10.0, 20.0, 0.0, 30.0])
    )
    assert result["n"] == 3
    assert result["bias"] == pytest.approx(17 / 3)
    assert result["sse"] == 405
    assert result["rate_abs_ge20"] == pytest.approx(1 / 3)
    assert result["mae"] == pytest.approx(23 / 3)
    empty = residual_summary(np.ones(2), np.zeros(2))
    assert empty["n"] == 0 and empty["sse"] == 0 and np.isnan(empty["bias"])


def test_forecast_partitions_use_boundary_and_missing_origin_precedence():
    from traffic.analysis import forecast_partitions

    groups = forecast_partitions(
        np.array([0.0, 50.0, 20.0, 35.0, 60.0, 50.0, 20.0]),
        np.array([30.0, 40.0, 30.0, 35.0, 60.0, 40.001, 29.999]),
        np.array([0, 1, 3, 4, 11, 12, 0]),
        np.array([0, 1, 12, 13, 288, 289, 0]),
    )
    np.testing.assert_array_equal(groups["transition"][0], [0, 1, 2, 3, 4, 4, 3])
    np.testing.assert_array_equal(groups["input_missing"][0], [0, 1, 1, 2, 2, 3, 0])
    np.testing.assert_array_equal(groups["origin_gap_age"][0], [0, 1, 1, 2, 2, 3, 0])
