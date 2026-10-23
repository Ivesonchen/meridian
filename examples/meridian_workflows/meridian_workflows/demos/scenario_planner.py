"""Export Scenario Planner tables locally; upload only on an explicit command."""

import logging
import uuid

from meridian_workflows import config
from meridian_workflows import data
from meridian_workflows import pipeline


def export_tables(mmm, options: config.Options) -> None:
  from meridian.analysis import weekly_optimization_grid
  from meridian.schema import mmm_proto_generator
  from meridian.schema.processors import budget_optimization_processor as budget
  from meridian.schema.processors import marketing_processor as marketing
  from meridian.schema.processors import model_fit_processor
  from meridian.schema.processors import model_processor
  from meridian.schema.processors import reach_frequency_optimization_processor as rf
  from meridian.schema.utils import date_range_bucketing
  from scenarioplanner import mmm_ui_proto_generator
  from scenarioplanner.converters.dataframe import dataframe_model_converter

  pipeline.require_convergence(mmm, allow_unconverged=options.allow_unconverged)
  options.output.mkdir(parents=True, exist_ok=True)
  lower, upper = 0.9, 1.0
  constraints = [
      budget.ChannelConstraintRel(
          channel_name=channel,
          spend_constraint_lower=lower,
          spend_constraint_upper=upper,
      )
      for channel in mmm.input_data.get_all_paid_channels()
  ]
  weekly_grid = weekly_optimization_grid.WeeklyOptimizationGrid.create(
      analyzer=pipeline.get_analyzer(mmm),
      use_kpi=mmm.input_data.revenue_per_kpi is None,
      max_budget_percent_decrease=lower,
      max_budget_percent_increase=upper,
      use_optimal_frequency=True,
      max_frequency=10.0,
      multiplier_step=0.01,
  )
  periods = [(None, None, "ALL")]
  coordinates = mmm.input_data.time_coordinates
  if options.quarterly:
    dates = list(coordinates.all_dates)
    dates.append(coordinates.period_end(dates[-1]))
    generator_class = date_range_bucketing.QuarterlyDateRangeGenerator
    for start, end in generator_class(dates).generate_date_intervals():
      periods.append(
          (
              start,
              end,
              mmm_ui_proto_generator.create_tag(generator_class, start),
          )
      )
  specs = [model_fit_processor.ModelFitSpec()]
  group = str(uuid.uuid4())
  for start, end, tag in periods:
    closed_start, closed_end = (
        model_processor.DatedSpec(start_date=start, end_date=end)
        .resolver(coordinates)
        .to_closed_date_interval_tuple()
    )
    grid = weekly_grid.to_optimization_grid(
        start_date=closed_start,
        end_date=closed_end,
        spend_constraint_lower=lower,
        spend_constraint_upper=upper,
    )
    specs.append(
        marketing.MarketingAnalysisSpec(
            start_date=start,
            end_date=end,
            date_interval_tag=tag,
            media_summary_spec=marketing.MediaSummarySpec(
                include_non_paid_channels=True
            ),
        )
    )
    optimization_spec = budget.BudgetOptimizationSpec(
        start_date=start,
        end_date=end,
        date_interval_tag=tag,
        optimization_name=f"Python demo {tag}",
        grid_name=f"python_demo_{tag.replace(' ', '_')}",
        group_id=f"{group}:{tag}",
        constraints=constraints,
        use_optimal_frequency=True,
        max_frequency=10.0,
        grid=grid,
        kpi_type=(
            budget.common.KpiType.NON_REVENUE
            if mmm.input_data.revenue_per_kpi is None
            else budget.common.KpiType.REVENUE
        ),
    )
    specs.append(optimization_spec)
    if mmm.input_data.rf_channel is not None:
      specs.append(
          rf.ReachFrequencyOptimizationSpec(
              start_date=start,
              end_date=end,
              date_interval_tag=tag,
              optimization_name=f"RF Python demo {tag}",
              grid_name=f"rf_python_demo_{tag.replace(' ', '_')}",
              group_id=optimization_spec.group_id,
              confidence_level=optimization_spec.confidence_level,
              max_frequency=10.0,
              kpi_type=optimization_spec.kpi_type,
          )
      )
  proto = mmm_proto_generator.create_mmm_proto(mmm=mmm, specs=specs)
  (options.output / "scenario.binpb").write_bytes(proto.SerializeToString())
  tables = dataframe_model_converter.DataFrameModelConverter(proto)()
  table_dir = options.output / "tables"
  table_dir.mkdir(parents=True, exist_ok=True)
  for name, frame in tables.items():
    frame.to_csv(table_dir / f"{name}.csv", index=False)
  logging.info("Exported %d tables locally to %s", len(tables), table_dir)


def upload_tables(options: config.Options) -> None:
  import google.auth
  import pandas as pd
  from scenarioplanner.converters import sheets
  from scenarioplanner.linkingapi import url_generator

  paths = sorted((options.output / "tables").glob("*.csv"))
  if not paths:
    raise FileNotFoundError(
        "No tables found. Run scenario-planner export first."
    )
  credentials, _ = google.auth.default(
      scopes=[
          "https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive",
      ]
  )
  tables = {path.stem: pd.read_csv(path) for path in paths}
  spreadsheet = sheets.upload_to_gsheet(
      tables, credentials, spreadsheet_name=options.spreadsheet_name
  )
  url = url_generator.create_report_url(spreadsheet)
  (options.output / "scenario_planner_url.txt").write_text(
      f"{spreadsheet.url}\n{url}\n", encoding="utf-8"
  )
  print(f"Spreadsheet: {spreadsheet.url}\nScenario Planner: {url}")


def run(options: config.Options) -> None:
  if options.action == "upload":
    upload_tables(options)
  elif options.action == "export":
    export_tables(pipeline.load_model(options.model_path), options)
  else:
    pipeline.run_model(
        options,
        lambda: data.make_model(
            data.read_example("geo_all_channels.csv", options.data),
            non_paid=True,
        ),
    )
