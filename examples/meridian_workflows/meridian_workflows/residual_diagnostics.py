"""Descriptive residual checks, separate from Bayesian convergence diagnostics."""

from html import escape
import logging
from pathlib import Path

from meridian import constants
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from statsmodels.stats import stattools
import xarray as xr


def _statistics(
    residuals: NDArray[np.float64], scope: str, geo: str
) -> dict[str, str | int | float]:
  dw = jb = p_value = float("nan")
  notes = []
  if residuals.size < 2:
    notes.append(
        "Both statistics are undefined with fewer than 2 observations."
    )
  elif not np.any(residuals):
    notes.append("Both statistics are undefined for all-zero residuals.")
  else:
    # Both statistics are scale-invariant; rescaling avoids squared overflow.
    scaled = residuals / np.max(np.abs(residuals))
    dw = float(stattools.durbin_watson(scaled))
    if np.ptp(scaled) == 0:
      notes.append("Jarque-Bera is undefined for constant residuals.")
    else:
      jb_result = stattools.jarque_bera(scaled)
      jb, p_value = float(jb_result[0]), float(jb_result[1])
      if not np.isfinite([dw, jb, p_value]).all():
        raise ValueError(f"Non-finite residual statistics for {scope}: {geo}.")
  if notes:
    logging.warning(
        "Residual diagnostics for %s %s: %s", scope, geo, " ".join(notes)
    )
  return {
      "scope": scope,
      "geo": geo,
      "n_observations": residuals.size,
      "durbin_watson": dw,
      "jarque_bera_statistic": jb,
      "jarque_bera_p_value": p_value,
      "notes": " ".join(notes),
  }


def compute(fit_data: xr.Dataset) -> pd.DataFrame:
  """Use ModelFit's All Data posterior mean, retaining separate geo sequences."""
  required = {constants.ACTUAL, constants.EXPECTED}
  if not required.issubset(fit_data.data_vars):
    raise ValueError(
        "Model fit data must contain actual and expected outcomes."
    )
  dates = pd.DatetimeIndex(
      pd.to_datetime(fit_data.coords[constants.TIME].values, errors="raise")
  )
  if dates.empty or dates.hasnans or not dates.is_unique:
    raise ValueError(
        "Residual diagnostics require nonempty, unique valid dates."
    )
  geos = fit_data.coords[constants.GEO].to_index()
  if geos.empty or not geos.is_unique:
    raise ValueError("Residual diagnostics require nonempty, unique geos.")
  ordered = fit_data.isel({constants.TIME: np.argsort(dates.asi8)})
  actual = np.asarray(
      ordered[constants.ACTUAL].transpose(constants.GEO, constants.TIME),
      dtype=np.float64,
  )
  expected = np.asarray(
      ordered[constants.EXPECTED]
      .sel({constants.METRIC: constants.MEAN}, drop=True)
      .transpose(constants.GEO, constants.TIME),
      dtype=np.float64,
  )
  if not np.isfinite(actual).all() or not np.isfinite(expected).all():
    raise ValueError(
        "Actual and posterior-mean outcomes must be finite. Missing periods "
        "cannot be silently dropped from a time-series diagnostic."
    )
  residuals = actual - expected
  if not np.isfinite(residuals).all():
    raise ValueError("Residuals overflowed; check the outcome scale.")
  records = []
  if len(geos) > 1:
    national = residuals.sum(axis=0)
    if not np.isfinite(national).all():
      raise ValueError(
          "National residuals overflowed; check the outcome scale."
      )
    records.append(_statistics(national, "national", "All geos (summed)"))
  records.extend(
      _statistics(residuals[index], "geo", str(geo))
      for index, geo in enumerate(geos)
  )
  return pd.DataFrame.from_records(records)


def write_report(
    fit_data: xr.Dataset,
    output: Path,
    *,
    use_kpi: bool,
    health_status: str,
) -> pd.DataFrame:
  """Write local HTML/CSV without recomputing predictions or altering health."""
  table = compute(fit_data)
  scale = "KPI" if use_kpi else "revenue"
  table.insert(2, "outcome_scale", scale)
  table.insert(3, "evaluation_set", "All Data")
  output.mkdir(parents=True, exist_ok=True)
  table.to_csv(output / "residual_diagnostics.csv", index=False, na_rep="NaN")
  rendered = table.to_html(
      index=False,
      escape=True,
      na_rep="undefined",
      float_format=lambda value: f"{value:.6g}",
      border=0,
  )
  html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian residual diagnostics</title>
<style>
body {{ font-family: sans-serif; max-width: 1400px; margin: 2rem auto; padding: 0 1rem; }}
nav a {{ margin-right: 1rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 0.65rem; border-bottom: 1px solid #ddd; text-align: left; }}
th {{ background: #eef3fb; }}
.notice {{ padding: 1rem; background: #fff4d6; }}
.table-container {{ overflow-x: auto; }}
</style>
</head>
<body>
<h1>Residual diagnostics</h1>
<nav><a href="health_card.html">Model health card</a>
<a href="summary.html">Model summary</a>
<a href="residual_diagnostics.csv">Download CSV</a></nav>
<p class="notice">Meridian health status: <strong>{escape(health_status)}</strong>.
These supplementary statistics do not change the health score, establish MCMC
convergence, or make a smoke-test model suitable for business decisions.</p>
<h2>What is being measured?</h2>
<p>Residual = actual outcome minus the posterior mean of the expected outcome,
on the <strong>{escape(scale)}</strong> scale used in the model-fit chart.
This is All Data, including any holdout observations, not a separate holdout
evaluation. Each geo is ordered by date; no periods are dropped.
For multiple geos, the national row sums residuals across geos at each time
point. It never concatenates different geos into one artificial time series.</p>
<ul>
<li><strong>Durbin-Watson:</strong> a descriptive first-order autocorrelation
statistic, usually between 0 and 4. Values near 2 suggest little first-order
autocorrelation; lower/higher values suggest positive/negative autocorrelation.
This report does not apply a universal pass/fail cutoff.</li>
<li><strong>Jarque-Bera p-value:</strong> tests a normal reference distribution
using residual skewness and kurtosis. A small p-value suggests a departure
from normality; it is <em>not</em> the Bayesian posterior predictive p-value.
The p-value uses a large-sample chi-square approximation and needs caution
with short series, fitted residuals, heteroscedasticity, or serial dependence.
These reporting-scale residuals do not directly test the Bayesian likelihood.</li>
<li>Undefined statistics are shown explicitly with a reason. An undefined
statistic is not a pass.</li>
</ul>
<div class="table-container">{rendered}</div>
</body>
</html>
"""
  (output / "residual_diagnostics.html").write_text(html, encoding="utf-8")
  logging.info(
      "Residual diagnostics (%s scale, All Data): %s",
      scale,
      (output / "residual_diagnostics.html").resolve(),
  )
  return table
