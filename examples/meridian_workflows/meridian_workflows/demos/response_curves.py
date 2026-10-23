"""ROI, marginal ROI, credible intervals, and response curves from saved draws."""

from meridian.analysis import visualizer
from meridian.model import model
from meridian_workflows import config
from meridian_workflows import data
from meridian_workflows import pipeline
import numpy as np
import pandas as pd


def build_model(options: config.Options) -> model.Meridian:
  return data.make_model(
      data.read_example("geo_all_channels.csv", options.data),
      default_priors=True,
  )


def run(options: config.Options) -> None:
  if options.action != "analyze":
    pipeline.run_model(options, lambda: build_model(options))
    return
  mmm = pipeline.load_model(options.model_path)
  pipeline.require_convergence(mmm, allow_unconverged=options.allow_unconverged)
  options.output.mkdir(parents=True, exist_ok=True)
  analysis = pipeline.get_analyzer(mmm)
  metrics = analysis.summary_metrics()
  for name in ("roi", "mroi"):
    metrics[name].sel(distribution="posterior").to_dataframe().to_csv(
        options.output / f"{name}_credible_intervals.csv"
    )
  roi = np.asarray(analysis.roi())
  marginal_roi = np.asarray(analysis.marginal_roi())
  pd.DataFrame(
      {
          "channel": mmm.input_data.get_all_paid_channels(),
          "roi_std": roi.std(axis=(0, 1)),
          "mroi_std": marginal_roi.std(axis=(0, 1)),
          "probability_roi_at_least_one": (roi >= 1).mean(axis=(0, 1)),
      }
  ).to_csv(options.output / "posterior_uncertainty.csv", index=False)
  visualizer.MediaEffects(analyzer=analysis).plot_response_curves(
      plot_separately=True, include_ci=True
  ).save(str(options.output / "response_curves.html"))
