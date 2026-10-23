"""Train once, persist immediately, and run later analysis in fresh processes."""

from collections.abc import Callable
import dataclasses
import json
import logging
from pathlib import Path
import tempfile
import time

import arviz as az
import meridian
from meridian.analysis import analyzer
from meridian.analysis import optimizer
from meridian.analysis import summarizer
from meridian.analysis import visualizer
from meridian.analysis.review import reviewer
from meridian.model import model
from meridian.schema.serde import meridian_serde
from meridian_workflows import config
from meridian_workflows import data
from meridian_workflows import residual_diagnostics
import numpy as np


def save_model(mmm: model.Meridian, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(dir=path.parent) as directory:
    temporary = Path(directory) / path.name
    meridian_serde.save_meridian(mmm, str(temporary))
    temporary.replace(path)
  logging.info("Saved model and inference data: %s", path.resolve())


def train(mmm: model.Meridian, options: config.Options) -> model.Meridian:
  if options.model is not None:
    raise ValueError("--model is for analysis, not training or resuming MCMC.")
  path = options.output / "model.binpb"
  if path.exists() and not options.overwrite:
    raise FileExistsError(
        f"{path} already exists. Reuse it for report/analyze, choose another "
        "output directory, or explicitly pass --overwrite."
    )
  options.output.mkdir(parents=True, exist_ok=True)
  if options.profile == "smoke":
    logging.warning("SMOKE TEST ONLY: samples are not suitable for inference.")
  logging.info("Sampling parameters: %s", options.sampling)
  started = time.perf_counter()
  mmm.sample_prior(options.sampling.prior, seed=options.sampling.seed)
  prior_seconds = time.perf_counter() - started
  started = time.perf_counter()
  mmm.sample_posterior(**options.sampling.posterior_kwargs())
  posterior_seconds = time.perf_counter() - started
  save_model(mmm, path)
  metadata = {
      "profile": options.profile,
      "sampling": dataclasses.asdict(options.sampling),
      "meridian_version": meridian.__version__,
      "prior_seconds": prior_seconds,
      "posterior_seconds": posterior_seconds,
      "quality": "Not yet reviewed. Full profile does not imply convergence.",
  }
  path.with_suffix(".json").write_text(
      json.dumps(metadata, indent=2), encoding="utf-8"
  )
  logging.info("Posterior completed in %.1f seconds.", posterior_seconds)
  return mmm


def load_model(path: Path) -> model.Meridian:
  if not path.is_file():
    raise FileNotFoundError(
        f"No saved model at {path}. Run train first or provide --model."
    )
  if path.suffix != ".binpb":
    raise ValueError("Use a Meridian .binpb model, not a notebook or pickle.")
  logging.info("Loading existing model; sampling will NOT run: %s", path)
  return meridian_serde.load_meridian(str(path))


def get_analyzer(mmm: model.Meridian) -> analyzer.Analyzer:
  return analyzer.Analyzer(
      model_context=mmm.model_context, inference_data=mmm.inference_data
  )


def require_convergence(
    mmm: model.Meridian, *, allow_unconverged: bool = False
) -> None:
  if "posterior" not in mmm.inference_data.groups():
    raise ValueError("The model has no posterior samples.")
  posterior = mmm.inference_data.posterior
  if posterior.sizes["chain"] < 2 or posterior.sizes["draw"] < 4:
    message = "Too few chains/draws to assess convergence."
  else:
    diagnostics = az.rhat(posterior)
    maxima = []
    invalid = False
    for name, values in diagnostics.data_vars.items():
      rhat = values.values
      finite = np.isfinite(rhat)
      if finite.any():
        maxima.append(float(np.max(rhat[finite])))
      # Constant deterministic parameters have undefined R-hat.
      variable = posterior[name].std(dim=("chain", "draw")).values > 0
      invalid |= bool(np.any(~finite & variable))
      invalid |= not bool(np.isfinite(posterior[name].values).all())
    maximum = max(maxima) if maxima else float("nan")
    divergences = 0
    if "sample_stats" in mmm.inference_data.groups():
      if "diverging" in mmm.inference_data.sample_stats:
        divergences = int(mmm.inference_data.sample_stats.diverging.sum())
    if maxima and not invalid and maximum <= 1.01 and not divergences:
      return
    message = (
        f"Convergence not established (maximum finite R-hat={maximum:.3f}, "
        f"divergences={divergences})."
    )
  if not allow_unconverged:
    raise ValueError(
        f"{message} Generate a report and inspect the chains first. "
        "--allow-unconverged is only for exploratory demonstrations."
    )
  logging.warning("%s EXPLORATORY OUTPUT: do not use for decisions.", message)


def report(mmm: model.Meridian, output: Path) -> None:
  output.mkdir(parents=True, exist_ok=True)
  review = reviewer.ModelReviewer(
      model_context=mmm.model_context, inference_data=mmm.inference_data
  ).run()
  review.output_model_health_card("health_card.html", str(output))
  logging.info(
      "Health status: %s. %s", review.overall_status, review.summary_message
  )
  az.summary(mmm.inference_data).to_csv(output / "parameter_diagnostics.csv")
  use_kpi = mmm.input_data.revenue_per_kpi is None
  visualizer.ModelDiagnostics(mmm, use_kpi=use_kpi).plot_rhat_boxplot().save(
      str(output / "rhat.html")
  )
  model_fit = visualizer.ModelFit(mmm, use_kpi=use_kpi)
  residual_diagnostics.write_report(
      model_fit.model_fit_data,
      output,
      use_kpi=use_kpi,
      health_status=review.overall_status.name,
  )
  model_fit.plot_model_fit().save(str(output / "model_fit.html"))
  summarizer.Summarizer(mmm, use_kpi=use_kpi).output_model_results_summary(
      "summary.html", str(output)
  )
  logging.info("Open HTML reports in a browser: %s", output.resolve())


def optimize(mmm: model.Meridian, options: config.Options) -> None:
  require_convergence(mmm, allow_unconverged=options.allow_unconverged)
  options.output.mkdir(parents=True, exist_ok=True)
  result = optimizer.BudgetOptimizer(analyzer=get_analyzer(mmm)).optimize()
  result.output_optimization_summary("optimization.html", str(options.output))


def run_model(
    options: config.Options,
    build: Callable[[], model.Meridian],
) -> model.Meridian:
  if options.action == "prepare":
    mmm = build()
    data.describe(mmm, options.output)
  elif options.action == "train":
    if options.model is not None:
      raise ValueError(
          "--model is for analysis, not training or resuming MCMC."
      )
    mmm = train(build(), options)
  elif options.action in ("report", "optimize"):
    mmm = load_model(options.model_path)
    if options.action == "report":
      report(mmm, options.output)
    else:
      optimize(mmm, options)
  else:
    raise ValueError(f"Unsupported model action: {options.action}")
  return mmm
