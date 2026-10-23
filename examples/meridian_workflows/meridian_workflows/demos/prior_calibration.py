"""Three explicit calibration cases: GeoX, generic, and combined experiments."""

import datetime
import logging

from meridian import constants
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
from meridian.model.calibration import prior_builder
from meridian_workflows import config
from meridian_workflows import data
from meridian_workflows import pipeline
import numpy as np
import pandas as pd


def mock_geox_result(estimate, standard_error, spend, start, end):
  """Construct synthetic experiment evidence, never actual measured evidence."""
  from meridian_geox import api

  estimate = api.Estimate(
      point_estimate=estimate,
      lower_bound=estimate - 1.96 * standard_error,
      upper_bound=estimate + 1.96 * standard_error,
      standard_deviation=standard_error,
      p_value=0.05,
  )
  metrics = api.AnalysisMetrics(
      lift=estimate,
      percent_lift=estimate,
      icpd=estimate,
      descriptive_metrics=api.DescriptiveMetrics(estimated_bau_spend=spend),
  )
  design = api.Design(
      designs={
          "cell_1": api.PerCellDesign(
              treatment_geos={"geo_1"},
              minimum_detectable_effect=0.1,
              design_implied_cpic=1.0,
              p_value=0.05,
              budget=spend,
          )
      },
      control_geos={"geo_2"},
      excluded_geos=set(),
  )
  return api.AnalysisResult(
      results={"cell_1": metrics},
      analysis_config=api.AnalysisConfig(
          design=design,
          analysis_start_date=pd.Timestamp(start),
          analysis_end_date=pd.Timestamp(end),
      ),
  )


def build_model(options: config.Options) -> model.Meridian:
  logging.warning("Calibration demo uses SYNTHETIC experiment results.")
  frame = data.read_example("geo_all_channels.csv", options.data)
  inputs = data.build_input(frame)
  builder = prior_builder.CalibrationBuilder(data=inputs)
  start = inputs.time_coordinates.all_dates[0]
  end = start + datetime.timedelta(weeks=4)
  for channel, estimate, error, fraction in (
      ("Channel0", 2.5, 0.5, 0.10),
      ("Channel2", 3.0, 0.6, 0.05),
  ):
    spend = float(inputs.media_spend.sel(media_channel=channel).sum())
    builder.with_meridian_geox_experiment_result(
        channel_name=channel,
        geox_results=mock_geox_result(
            estimate, error, spend * fraction, start, end
        ),
        experiment_kpi_types=constants.NON_REVENUE,
    )
  for channel, estimate, error, fraction in (
      ("Channel1", 1.8, 0.4, 0.15),
      ("Channel2", 2.2, 0.5, 0.08),
  ):
    spend = float(inputs.media_spend.sel(media_channel=channel).sum())
    builder.with_incrementality_experiment_result(
        channel_name=channel,
        point_estimates=estimate,
        standard_errors=error,
        experiment_kpi_types=constants.NON_REVENUE,
        experiment_total_spends=spend * fraction,
        experiment_start_dates=start,
        experiment_end_dates=end,
    )
  calibrated = builder.build()
  return model.Meridian(
      input_data=inputs,
      model_spec=spec.ModelSpec(
          prior=prior_distribution.PriorDistribution(roi_m=calibrated.roi_m),
          media_prior_type="roi",
      ),
  )


def run(options: config.Options) -> None:
  if options.action != "calibrate":
    pipeline.run_model(options, lambda: build_model(options))
    return
  import jax
  import matplotlib.pyplot as plt

  mmm = build_model(options)
  data.describe(mmm, options.output)
  samples = np.asarray(
      mmm.model_spec.prior.roi_m.sample(
          10000, seed=jax.random.PRNGKey(options.sampling.seed)
      )
  )
  np.savez_compressed(
      options.output / "calibrated_prior_samples.npz", roi=samples
  )
  fig, axes = plt.subplots(2, 3, figsize=(14, 8))
  labels = ("GeoX", "Generic", "GeoX + generic", "Uncalibrated", "Uncalibrated")
  for i, (channel, label) in enumerate(
      zip(mmm.input_data.media_channel, labels)
  ):
    axes.flat[i].hist(samples[:, i], bins=50, density=True)
    axes.flat[i].set_title(f"{channel.item()} ({label})")
    axes.flat[i].set_xlabel("ROI")
  axes.flat[-1].set_visible(False)
  fig.tight_layout()
  fig.savefig(options.output / "calibrated_priors.png")
  plt.close(fig)
