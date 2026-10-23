"""Dispatch to ordinary Python functions; no notebook execution is involved."""

import argparse
import dataclasses
import importlib
import logging
from pathlib import Path

from meridian_workflows import config

DEMOS = {
    "getting-started": ("getting_started", "End-to-end JAX marketing model"),
    "getting-started-tf": ("getting_started_tf", "Legacy TensorFlow variant"),
    "reach-frequency": ("reach_frequency", "Reach and frequency modeling"),
    "response-curves": ("response_curves", "ROI, mROI and uncertainty"),
    "prior-calibration": ("prior_calibration", "GeoX and generic experiments"),
    "full-funnel": ("full_funnel", "Two-stage brand-to-sales modeling"),
    "scenario-planner": (
        "scenario_planner",
        "Local tables and optional Sheets",
    ),
    "mlflow": ("mlflow_demo", "Local experiment tracking"),
    "rf-simulation": ("rf_simulation", "Synthetic RF data and ground truth"),
}

MODEL_ACTIONS = ("prepare", "train", "report", "optimize")
ACTIONS = {
    "getting-started": (*MODEL_ACTIONS, "hypothetical"),
    "getting-started-tf": (*MODEL_ACTIONS, "hypothetical", "eda"),
    "reach-frequency": MODEL_ACTIONS,
    "response-curves": (*MODEL_ACTIONS, "analyze"),
    "prior-calibration": (*MODEL_ACTIONS, "calibrate"),
    "full-funnel": (
        "prepare",
        "train",
        "train-stage1",
        "train-stage2",
        "report",
        "analyze",
        "optimize",
    ),
    "scenario-planner": (*MODEL_ACTIONS, "export", "upload"),
    "mlflow": (*MODEL_ACTIONS, "runs"),
    "rf-simulation": ("prepare", "simulate"),
}


def chain_batches(value: str) -> tuple[int, ...]:
  try:
    batches = tuple(int(n) for n in value.split(","))
  except ValueError as error:
    raise argparse.ArgumentTypeError(
        "Use chain batches such as 2 or 2,2."
    ) from error
  if not batches or any(n < 1 for n in batches):
    raise argparse.ArgumentTypeError("Chain counts must be positive.")
  return batches


def notebook_sampling(demo: str) -> config.Sampling:
  if demo == "full-funnel":
    return config.Sampling((2, 2), 1000, 1000, 500, 500)
  if demo == "mlflow":
    return config.Sampling((2, 2, 2, 1), 1000, 500, 1000, 500, 1)
  adapt = 1000 if demo in ("getting-started", "prior-calibration") else 2000
  return config.Sampling((2, 2, 2, 2, 2), adapt, 500, 1000, 500)


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser(
      description="Run Meridian notebook workflows as separate Python stages."
  )
  result.add_argument("demo", choices=["list", *DEMOS])
  result.add_argument("action", nargs="?", default="prepare")
  result.add_argument("--backend", choices=("jax", "tensorflow"))
  result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
  result.add_argument("--output-dir", type=Path)
  result.add_argument("--data", type=Path, help="Override the example CSV.")
  result.add_argument(
      "--model", type=Path, help="Read an existing .binpb model."
  )
  result.add_argument("--profile", choices=("smoke", "full"), default="smoke")
  result.add_argument("--chains", type=chain_batches, help="E.g. 4 or 2,2.")
  for field in ("adapt", "burnin", "keep", "prior", "seed"):
    result.add_argument(f"--{field}", type=int)
  result.add_argument("--overwrite", action="store_true")
  result.add_argument("--allow-unconverged", action="store_true")
  result.add_argument("--quarterly", action="store_true")
  result.add_argument("--spreadsheet-name", default="Meridian Python demo")
  return result


def main(argv: list[str] | None = None) -> int:
  argument_parser = parser()
  args = argument_parser.parse_args(argv)
  if args.demo == "list":
    for name, (_, description) in DEMOS.items():
      print(f"{name:22} {description}\n  Actions: {', '.join(ACTIONS[name])}")
    return 0
  if args.action not in ACTIONS[args.demo]:
    argument_parser.error(
        f"{args.demo} supports: {', '.join(ACTIONS[args.demo])}"
    )
  if args.model is not None and args.action not in (
      "report",
      "analyze",
      "optimize",
      "hypothetical",
      "export",
  ):
    argument_parser.error("--model is for saved-model analysis, not training.")
  backend = args.backend or (
      "tensorflow" if args.demo == "getting-started-tf" else "jax"
  )
  if args.demo == "getting-started-tf" and backend != "tensorflow":
    argument_parser.error("getting-started-tf requires --backend tensorflow.")
  if args.demo == "prior-calibration" and backend != "jax":
    argument_parser.error("The prior-calibration example requires JAX.")
  sampling = (
      notebook_sampling(args.demo)
      if args.profile == "full"
      else config.Sampling()
  )
  overrides = {
      field.name: getattr(args, field.name)
      for field in dataclasses.fields(config.Sampling)
      if getattr(args, field.name) is not None
  }
  try:
    sampling = dataclasses.replace(sampling, **overrides)
  except ValueError as error:
    argument_parser.error(str(error))
  options = config.Options(
      output=args.output_dir or Path("outputs") / args.demo,
      action=args.action,
      profile=args.profile,
      sampling=sampling,
      data=args.data,
      model=args.model,
      overwrite=args.overwrite,
      allow_unconverged=args.allow_unconverged,
      quarterly=args.quarterly,
      spreadsheet_name=args.spreadsheet_name,
  )
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
  config.configure_runtime(backend, args.device)
  config.check_runtime(backend, args.device)
  module = importlib.import_module(
      f"meridian_workflows.demos.{DEMOS[args.demo][0]}"
  )
  module.run(options)
  return 0
