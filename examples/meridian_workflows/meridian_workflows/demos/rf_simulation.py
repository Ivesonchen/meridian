"""Simulate RF data and known effects using the notebook's generative model."""

from dataclasses import dataclass
import json

from meridian import backend
from meridian.model import adstock_hill
from meridian.model import prior_distribution
from meridian.model import transformers
from meridian_workflows import config
import numpy as np
import pandas as pd


@dataclass
class Simulation:
  geo: pd.DataFrame
  national: pd.DataFrame
  raw_parameters: dict[str, np.ndarray]
  ground_truth: dict[str, np.ndarray]


def aggregate_national(frame: pd.DataFrame) -> pd.DataFrame:
  grouped = frame.groupby("time", sort=True)
  rules = {
      name: ("mean" if name.endswith("_control") else "sum")
      for name in frame.columns
      if name.endswith(("_impression", "_spend", "_reach", "_control"))
      or name == "conversions"
  }
  national = grouped.agg(rules)
  revenue = (
      (frame.conversions * frame.revenue_per_conversion)
      .groupby(frame.time)
      .sum()
  )
  national["revenue_per_conversion"] = revenue / national["conversions"]
  for name in frame.columns:
    if name.endswith("_frequency"):
      reach_name = name.removesuffix("_frequency") + "_reach"
      impressions = (frame[name] * frame[reach_name]).groupby(frame.time).sum()
      reach = national[reach_name]
      national[name] = np.divide(
          impressions, reach, out=np.zeros(len(reach)), where=reach > 0
      )
  return national.reset_index()


def simulate(
    seed: int = 1320,
    n_geos: int = 20,
    n_times: int = 156,
    n_imp_channels: int = 3,
    n_rf_channels: int = 1,
) -> Simulation:
  if min(n_geos, n_imp_channels, n_rf_channels) < 1 or n_times < 2:
    raise ValueError(
        "Use positive geos/channels and at least two time periods."
    )
  rng = np.random.default_rng(seed)
  n_channels = n_imp_channels + n_rf_channels
  population = rng.uniform(1e5, 1e6, n_geos)
  controls = rng.normal(0, 3, (n_geos, n_times, 2))

  def tensor(value):
    return backend.to_tensor(value, dtype=backend.float_dtype)

  scaled_controls = np.asarray(
      transformers.CenteringAndScalingTransformer(
          tensor=tensor(controls),
          population=tensor(population),
          population_scaling_id=None,
      ).forward(tensor(controls))
  )
  per_capita = np.maximum(
      rng.normal(1, 0.5, n_channels)
      + rng.normal(0.8, 0.5, (n_times, n_channels))
      + rng.normal(0, 0.5, (n_geos, n_times, n_channels)),
      0,
  )
  impressions = np.round(per_capita * population[:, None, None])
  rf_per_capita = per_capita[..., -n_rf_channels:]
  rf_impressions = impressions[..., -n_rf_channels:]
  mean_reach = population[:, None, None] * (1 - np.exp(-rf_per_capita))
  std_reach = np.sqrt(
      population[:, None, None]
      * (np.exp(-rf_per_capita) - np.exp(-2 * rf_per_capita))
  )
  reach = np.round(rng.normal(mean_reach, std_reach))
  # A simulated positive impression count requires at least one reached person.
  reach = np.minimum(
      np.maximum(reach, (rf_impressions > 0).astype(float)),
      np.minimum(rf_impressions, population[:, None, None]),
  )
  frequency = np.divide(
      rf_impressions, reach, out=np.zeros_like(reach), where=reach > 0
  )
  media = impressions[..., :n_imp_channels]
  scaled_media = transformers.MediaTransformer(
      media=tensor(media), population=tensor(population)
  ).forward(tensor(media))
  scaled_reach = transformers.MediaTransformer(
      media=tensor(reach), population=tensor(population)
  ).forward(tensor(reach))
  tau = rng.normal(15, 1.2, n_geos)
  mu = rng.normal(0, 2, n_times)
  spend = impressions * rng.uniform(0.011, 0.012, n_channels)
  revenue_per_kpi = rng.uniform(0.0345, 0.0355, (n_geos, n_times))
  beta = rng.normal(0.9, 0.1, n_channels)
  eta = np.abs(rng.normal(0, 0.18, n_channels))
  beta_geo = np.exp(beta + eta * rng.normal(size=(n_geos, n_channels)))
  gamma = rng.normal(3.5, 0.5, 2)
  xi = np.abs(rng.normal(0, 0.3, 2))
  gamma_geo = gamma + xi * rng.normal(size=(n_geos, 2))
  sigma = np.array([0.5])
  noise = rng.normal(0, sigma[0], (n_geos, n_times))
  priors = prior_distribution.PriorDistribution()
  handler = backend.RNGHandler(seed)
  parameters = {}
  for suffix, count in (("m", n_imp_channels), ("rf", n_rf_channels)):
    for parameter in ("alpha", "ec", "slope"):
      name = f"{parameter}_{suffix}"
      parameters[name] = getattr(priors, name).sample(
          count, seed=handler.get_next_seed()
      )
  media_effect = adstock_hill.HillTransformer(
      ec=parameters["ec_m"], slope=parameters["slope_m"]
  ).forward(
      adstock_hill.AdstockTransformer(
          alpha=parameters["alpha_m"], max_lag=8, n_times_output=n_times
      ).forward(scaled_media)
  )
  adjusted_frequency = adstock_hill.HillTransformer(
      ec=parameters["ec_rf"], slope=parameters["slope_rf"]
  ).forward(tensor(frequency))
  rf_effect = adstock_hill.AdstockTransformer(
      alpha=parameters["alpha_rf"], max_lag=8, n_times_output=n_times
  ).forward(scaled_reach * adjusted_frequency)
  transformed = np.concatenate(
      [np.asarray(media_effect), np.asarray(rf_effect)], axis=-1
  )
  baseline = np.maximum(
      tau[:, None]
      + mu[None, :]
      + np.einsum("gtc,gc->gt", scaled_controls, gamma_geo)
      + noise,
      0,
  )
  kpi = (
      baseline + np.einsum("gtm,gm->gt", transformed, beta_geo)
  ) * population[:, None]
  raw = {key: np.asarray(value) for key, value in parameters.items()}
  raw.update(
      beta_gm=beta_geo[:, :n_imp_channels],
      beta_grf=beta_geo[:, -n_rf_channels:],
      beta_m=beta[:n_imp_channels],
      beta_rf=beta[-n_rf_channels:],
      eta_m=eta[:n_imp_channels],
      eta_rf=eta[-n_rf_channels:],
      gamma_c=gamma,
      gamma_gc=gamma_geo,
      xi_c=xi,
      sigma=sigma,
      tau_g=tau,
      mu_t=mu,
      intercept_gt=tau[:, None] + mu[None, :],
  )
  kpi_scaler = transformers.KpiTransformer(
      kpi=tensor(kpi), population=tensor(population)
  )
  kpi_mean = float(kpi_scaler.population_scaled_mean)
  kpi_std = float(kpi_scaler.population_scaled_stdev)
  truth = dict(raw)
  for name in ("beta_gm", "beta_grf", "gamma_c", "gamma_gc", "sigma", "xi_c"):
    truth[name] = raw[name] / kpi_std
  for name in ("beta_m", "beta_rf"):
    truth[name] = raw[name] - np.log(kpi_std)
  truth["intercept_gt"] = (raw["intercept_gt"] - kpi_mean) / kpi_std
  roi = np.einsum(
      "g,gt,gtm,gm->m", population, revenue_per_kpi, transformed, beta_geo
  ) / spend.sum(axis=(0, 1))
  truth["roi_m"], truth["roi_rf"] = roi[:n_imp_channels], roi[-n_rf_channels:]
  times = pd.date_range("2021-01-25", periods=n_times, freq="7D").strftime(
      "%Y-%m-%d"
  )
  frame = pd.DataFrame(
      {
          "geo": np.repeat([f"Geo{i}" for i in range(n_geos)], n_times),
          "time": np.tile(times, n_geos),
          "population": np.repeat(population, n_times),
          "conversions": kpi.ravel(),
          "revenue_per_conversion": revenue_per_kpi.ravel(),
      }
  )
  for i, name in enumerate(
      ("sentiment_score_control", "competitor_activity_score_control")
  ):
    frame[name] = controls[..., i].ravel()
  for i in range(n_channels):
    frame[f"Channel{i}_impression"] = impressions[..., i].ravel()
    frame[f"Channel{i}_spend"] = spend[..., i].ravel()
  for i in range(n_rf_channels):
    frame[f"Channel{n_imp_channels + i}_reach"] = reach[..., i].ravel()
    frame[f"Channel{n_imp_channels + i}_frequency"] = frequency[..., i].ravel()
  return Simulation(frame, aggregate_national(frame), raw, truth)


def run(options: config.Options) -> None:
  if options.action == "prepare":
    print(
        json.dumps(
            {
                "geos": 20,
                "weeks": 156,
                "impression_channels": 3,
                "rf_channels": 1,
                "controls": 2,
                "next_step": "Run rf-simulation simulate; no MCMC is required.",
            },
            indent=2,
        )
    )
    return
  if options.action != "simulate":
    raise ValueError(f"Unsupported simulation action: {options.action}")
  paths = (
      "geo_rf_data.csv",
      "national_rf_data.csv",
      "ground_truth.npz",
      "raw_parameters.npz",
  )
  if not options.overwrite:
    for name in paths:
      if (options.output / name).exists():
        raise FileExistsError(
            f"{name} exists. Use --overwrite or another directory."
        )
  result = simulate(seed=options.sampling.seed)
  options.output.mkdir(parents=True, exist_ok=True)
  result.geo.to_csv(options.output / "geo_rf_data.csv", index=False)
  result.national.to_csv(options.output / "national_rf_data.csv", index=False)
  np.savez_compressed(
      options.output / "ground_truth.npz", **result.ground_truth
  )
  np.savez_compressed(
      options.output / "raw_parameters.npz", **result.raw_parameters
  )
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  axes[0].plot(result.raw_parameters["mu_t"])
  axes[0].set_title("Simulated time effect")
  axes[1].plot(result.geo.groupby("time")["conversions"].mean().to_numpy())
  axes[1].set_title("Mean KPI across geos")
  fig.tight_layout()
  fig.savefig(options.output / "simulation.png")
  plt.close(fig)
  spend_columns = [name for name in result.geo if name.endswith("_spend")]
  fig, axes = plt.subplots(
      (len(spend_columns) + 1) // 2, 2, figsize=(12, 6), squeeze=False
  )
  for axis, name in zip(axes.flat, spend_columns):
    axis.hist(result.geo[name], bins=30)
    axis.set_title(name)
  for axis in list(axes.flat)[len(spend_columns) :]:
    axis.set_visible(False)
  fig.tight_layout()
  fig.savefig(options.output / "spend_histograms.png")
  plt.close(fig)
  print(f"Saved synthetic data and ground truth to {options.output.resolve()}")
