"""Fit and persist two stages, then combine direct and mediated effects."""

import dataclasses
import json

from meridian.analysis import optimizer
from meridian.analysis import visualizer
from meridian.data import load
from meridian.model import model
from meridian.model import spec
from meridian_workflows import config
from meridian_workflows import data
from meridian_workflows import full_funnel_analysis as funnel
from meridian_workflows import pipeline


def build_model(options: config.Options, stage: int) -> model.Meridian:
  frame = data.read_example("geo_full_funnel_data.csv", options.data)
  if stage not in (1, 2):
    raise ValueError("Full-funnel stage must be 1 or 2.")
  channels = (
      ["AwarenessVideo"]
      if stage == 1
      else ["PerformanceNative", "PerformanceDisplay", "AwarenessVideo"]
  )
  media = [f"{name}_impression" for name in channels]
  spend = [f"{name}_spend" for name in channels]
  columns = load.CoordToColumns(
      time="time",
      geo="geo",
      population="population",
      kpi="BrandedGQV_control" if stage == 1 else "conversions",
      revenue_per_kpi=None if stage == 1 else "revenue_per_conversion",
      media=media,
      media_spend=spend,
      controls=None if stage == 1 else ["GenericGQV_control"],
      organic_media=None if stage == 1 else ["BrandedGQV_control"],
  )
  inputs = load.DataFrameDataLoader(
      frame,
      kpi_type="non_revenue",
      coord_to_columns=columns,
      media_to_channel=dict(zip(media, channels)),
      media_spend_to_channel=dict(zip(spend, channels)),
  ).load()
  model_spec = (
      spec.ModelSpec(knots=36)
      if stage == 1
      else spec.ModelSpec(
          knots=36,
          saturation_spec={"BrandedGQV_control": "none"},
          population_scaled_controls=["GenericGQV_control"],
      )
  )
  return model.Meridian(input_data=inputs, model_spec=model_spec)


def stage_options(options: config.Options, stage: int) -> config.Options:
  return dataclasses.replace(
      options,
      output=options.output / f"stage{stage}",
      sampling=dataclasses.replace(
          options.sampling, seed=options.sampling.seed + stage - 1
      ),
  )


def run(options: config.Options) -> None:
  if options.model is not None:
    raise ValueError(
        "Full funnel needs two models. Use --output-dir containing "
        "stage1/model.binpb and stage2/model.binpb, not --model."
    )
  if options.action == "prepare":
    for stage in (1, 2):
      data.describe(
          build_model(options, stage), stage_options(options, stage).output
      )
    return
  if options.action in ("train", "train-stage1", "train-stage2"):
    stages = (1, 2) if options.action == "train" else (int(options.action[-1]),)
    for stage in stages:
      path = stage_options(options, stage).model_path
      if path.exists() and not options.overwrite:
        raise FileExistsError(
            f"{path} exists. Train only the missing stage, or use --overwrite."
        )
    for stage in stages:
      pipeline.train(build_model(options, stage), stage_options(options, stage))
    return
  stage1 = pipeline.load_model(stage_options(options, 1).model_path)
  stage2 = pipeline.load_model(stage_options(options, 2).model_path)
  if options.action == "report":
    for stage, mmm in ((1, stage1), (2, stage2)):
      pipeline.report(mmm, stage_options(options, stage).output)
    return
  for mmm in (stage1, stage2):
    pipeline.require_convergence(
        mmm, allow_unconverged=options.allow_unconverged
    )
  analysis = funnel.AnalyzerFullFunnel(
      stage2, mediator_models={"BrandedGQV_control": stage1}
  )
  options.output.mkdir(parents=True, exist_ok=True)
  if options.action == "analyze":
    funnel.compute_roi_decomposition(analysis).to_csv(
        options.output / "roi_decomposition.csv", index=False
    )
    funnel.plot_roi_decomposition(analysis).save(
        str(options.output / "roi_decomposition.html")
    )
    funnel.plot_compounding_adstock_decay(analysis).save(
        str(options.output / "compounding_adstock.html")
    )
    visualizer.MediaEffects(analyzer=analysis).plot_response_curves().save(
        str(options.output / "response_curves.html")
    )
    for filename, chart in (
        (
            "stage1_response_curves.html",
            visualizer.MediaEffects(stage1).plot_response_curves(),
        ),
        (
            "stage1_cpik.html",
            visualizer.MediaSummary(stage1, use_kpi=True).plot_cpik(
                include_ci=True
            ),
        ),
        (
            "stage1_contributions.html",
            visualizer.MediaSummary(
                stage1, use_kpi=True
            ).plot_contribution_waterfall_chart(),
        ),
    ):
      chart.save(str(options.output / filename))
    (options.output / "baseline_probability.json").write_text(
        json.dumps(
            {
                "negative_baseline_probability": float(
                    analysis.negative_baseline_probability()
                )
            },
            indent=2,
        ),
        encoding="utf-8",
    )
  elif options.action == "optimize":
    result = optimizer.BudgetOptimizer(analyzer=analysis).optimize()
    result.output_optimization_summary("optimization.html", str(options.output))
    result.plot_spend_delta().save(str(options.output / "spend_delta.html"))
    result.plot_incremental_outcome_delta().save(
        str(options.output / "incremental_outcome_delta.html")
    )
  else:
    raise ValueError(f"Unsupported full-funnel action: {options.action}")
