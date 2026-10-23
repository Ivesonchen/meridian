import logging

from meridian_workflows import residual_diagnostics
import numpy as np
import pandas as pd
import pytest
import xarray as xr


def fit_data(residuals, geos=None):
  residuals = np.asarray(residuals, dtype=float)
  actual = 10 + residuals
  expected = np.full((*actual.shape, 3), 10.0)
  return xr.Dataset(
      {
          "actual": (("geo", "time"), actual),
          "expected": (("geo", "time", "metric"), expected),
      },
      coords={
          "geo": geos
          if geos is not None
          else [f"Geo{i}" for i in range(len(actual))],
          "time": pd.date_range(
              "2021-01-01", periods=actual.shape[1], freq="7D"
          ),
          "metric": ["mean", "ci_lo", "ci_hi"],
      },
  )


def test_known_statistics():
  result = residual_diagnostics.compute(fit_data([[1, -1, 1, -1]]))
  assert len(result) == 1
  assert result.iloc[0].n_observations == 4
  assert result.iloc[0].durbin_watson == pytest.approx(3)
  assert result.iloc[0].jarque_bera_statistic == pytest.approx(2 / 3)
  assert result.iloc[0].jarque_bera_p_value == pytest.approx(np.exp(-1 / 3))


def test_national_series_is_summed_not_concatenated():
  result = residual_diagnostics.compute(
      fit_data([[1, -1, 1, -1], [2, 1, 0, -1]])
  )
  assert len(result) == 3
  assert result.n_observations.tolist() == [4, 4, 4]
  national = result[result.scope == "national"].iloc[0]
  assert national.durbin_watson == pytest.approx(19 / 14)
  assert national.durbin_watson != pytest.approx(24 / 10)
  assert result[result.geo == "Geo1"].iloc[0].durbin_watson == pytest.approx(
      0.5
  )


def test_sorting_dates_preserves_time_series_statistics():
  ordered = fit_data([[1, -1, 2, -2]])
  shuffled = ordered.isel(time=[2, 0, 3, 1])
  pd.testing.assert_frame_equal(
      residual_diagnostics.compute(ordered),
      residual_diagnostics.compute(shuffled),
  )


@pytest.mark.parametrize(
    "values, expected_dw, message",
    [
        ([0, 0, 0, 0], np.nan, "all-zero"),
        ([1, 1, 1, 1], 0.0, "constant"),
        ([1], np.nan, "fewer than 2"),
    ],
)
def test_undefined_statistics_are_explained(
    values, expected_dw, message, caplog
):
  with caplog.at_level(logging.WARNING):
    result = residual_diagnostics.compute(fit_data([values]))
  row = result.iloc[0]
  if np.isnan(expected_dw):
    assert np.isnan(row.durbin_watson)
  else:
    assert row.durbin_watson == expected_dw
  assert np.isnan(row.jarque_bera_p_value)
  assert message in row.notes
  assert message in caplog.text


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_observations_are_not_silently_dropped(value):
  with pytest.raises(ValueError, match="must be finite"):
    residual_diagnostics.compute(fit_data([[1, value, 2, 3]]))


def test_nonfinite_prediction_is_rejected():
  frame = fit_data([[1, 2, 3, 4]])
  frame.expected.values[0, 1, 0] = np.nan
  with pytest.raises(ValueError, match="must be finite"):
    residual_diagnostics.compute(frame)


def test_duplicate_dates_are_rejected():
  frame = fit_data([[1, 2, 3, 4]])
  frame = frame.assign_coords(time=["2021-01-01"] * 4)
  with pytest.raises(ValueError, match="unique valid dates"):
    residual_diagnostics.compute(frame)


def test_duplicate_geos_are_rejected():
  with pytest.raises(ValueError, match="unique geos"):
    residual_diagnostics.compute(
        fit_data([[1, 2, 3, 4], [2, 1, 4, 3]], geos=["same", "same"])
    )


def test_large_residuals_do_not_overflow_squared_moments():
  result = residual_diagnostics.compute(
      fit_data([[1e200, -1e200, 1e200, -1e200]])
  )
  assert result.iloc[0].durbin_watson == pytest.approx(3)
  assert result.iloc[0].jarque_bera_p_value == pytest.approx(np.exp(-1 / 3))


def test_html_and_csv_include_scale_cautions_and_escape_labels(tmp_path):
  result = residual_diagnostics.write_report(
      fit_data([[1, -1, 1, -1]], geos=["<script>alert(1)</script>"]),
      tmp_path,
      use_kpi=False,
      health_status="FAIL",
  )
  assert result.outcome_scale.tolist() == ["revenue"]
  saved = pd.read_csv(
      tmp_path / "residual_diagnostics.csv", keep_default_na=False
  )
  assert saved.jarque_bera_p_value.iloc[0] == pytest.approx(np.exp(-1 / 3))
  html = (tmp_path / "residual_diagnostics.html").read_text()
  assert "<script>alert(1)</script>" not in html
  assert "&lt;script&gt;" in html
  assert "Durbin-Watson" in html
  assert "Jarque-Bera p-value" in html
  assert "<strong>FAIL</strong>" in html
  assert "not</em> the Bayesian posterior predictive p-value" in html
  assert "holdout" in html
  assert "chi-square approximation" in html


def test_kpi_scale_is_explicit(tmp_path):
  result = residual_diagnostics.write_report(
      fit_data([[1, -1, 1, -1]]),
      tmp_path,
      use_kpi=True,
      health_status="PASS",
  )
  assert result.outcome_scale.tolist() == ["KPI"]
