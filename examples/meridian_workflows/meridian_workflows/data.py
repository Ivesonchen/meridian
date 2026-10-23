"""Data loading and model construction shared by the notebook examples."""

from importlib import resources
import json
from pathlib import Path

from meridian import backend
from meridian import constants
from meridian.data import data_frame_input_data_builder as builders
from meridian.data import input_data
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
import pandas as pd


def read_example(filename: str, override: Path | None = None) -> pd.DataFrame:
  if override is not None:
    return pd.read_csv(override)
  resource = resources.files("meridian").joinpath(
      "data", "simulated_data", "csv", filename
  )
  with resources.as_file(resource) as path:
    return pd.read_csv(path)


def build_input(
    frame: pd.DataFrame,
    *,
    reach_frequency: bool = False,
    non_paid: bool = False,
) -> input_data.InputData:
  builder = builders.DataFrameInputDataBuilder(kpi_type="non_revenue")
  competitor = (
      "competitor_activity_score_control"
      if reach_frequency
      else "competitor_sales_control"
  )
  builder = (
      builder.with_kpi(frame, kpi_col="conversions")
      .with_revenue_per_kpi(frame, revenue_per_kpi_col="revenue_per_conversion")
      .with_population(frame)
      .with_controls(
          frame, control_cols=["sentiment_score_control", competitor]
      )
  )
  channels = [f"Channel{i}" for i in range(3 if reach_frequency else 5)]
  builder = builder.with_media(
      frame,
      media_cols=[f"{channel}_impression" for channel in channels],
      media_spend_cols=[f"{channel}_spend" for channel in channels],
      media_channels=channels,
  )
  if reach_frequency:
    builder = builder.with_reach(
        frame,
        reach_cols=["Channel3_reach"],
        frequency_cols=["Channel3_frequency"],
        rf_spend_cols=["Channel3_spend"],
        rf_channels=["Channel3"],
    )
  if non_paid:
    builder = builder.with_non_media_treatments(
        frame, non_media_treatment_cols=["Promo"]
    ).with_organic_media(
        frame,
        organic_media_cols=["Organic_channel0_impression"],
        organic_media_channels=["Organic_channel0"],
    )
  return builder.build()


def make_model(
    frame: pd.DataFrame,
    *,
    reach_frequency: bool = False,
    non_paid: bool = False,
    enable_aks: bool = False,
    default_priors: bool = False,
) -> model.Meridian:
  data = build_input(frame, reach_frequency=reach_frequency, non_paid=non_paid)
  if default_priors:
    model_spec = spec.ModelSpec(enable_aks=enable_aks)
  else:
    name = constants.ROI_RF if reach_frequency else constants.ROI_M
    prior = prior_distribution.PriorDistribution(
        **{
            name: backend.tfd.LogNormal(
                backend.to_tensor(0.2, dtype=backend.float_dtype),
                backend.to_tensor(0.9, dtype=backend.float_dtype),
                name=name,
            )
        }
    )
    model_spec = spec.ModelSpec(prior=prior, enable_aks=enable_aks)
  return model.Meridian(input_data=data, model_spec=model_spec)


def describe(mmm: model.Meridian, output: Path) -> None:
  output.mkdir(parents=True, exist_ok=True)
  data = mmm.input_data
  description = {
      "geos": len(data.geo),
      "time_periods": len(data.time),
      "first_date": str(data.time.values[0]),
      "last_date": str(data.time.values[-1]),
      "paid_channels": data.get_all_paid_channels().tolist(),
      "all_channels": data.get_all_channels().tolist(),
      "kpi_type": data.kpi_type,
      "note": "Prepared only; no posterior sampling has been run.",
  }
  (output / "data_summary.json").write_text(
      json.dumps(description, indent=2), encoding="utf-8"
  )
  print(json.dumps(description, indent=2))
