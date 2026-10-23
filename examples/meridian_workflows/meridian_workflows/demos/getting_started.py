"""Getting Started: build, train, diagnose, optimize, and test a new scenario."""

from meridian import backend
from meridian.analysis import optimizer
from meridian.analysis import tensors
from meridian.model import model
from meridian_workflows import config
from meridian_workflows import data
from meridian_workflows import pipeline
import pandas as pd


def build_model(options: config.Options) -> model.Meridian:
  frame = data.read_example("geo_all_channels.csv", options.data)
  return data.make_model(frame, non_paid=True, enable_aks=True)


def hypothetical_data(
    mmm: model.Meridian, frame: pd.DataFrame
) -> tensors.DataTensors:
  """Align each tensor explicitly to the trained model's geographic order."""
  frame = frame.copy()
  frame["time"] = pd.to_datetime(frame["time"], errors="raise").dt.strftime(
      "%Y-%m-%d"
  )
  geos = mmm.input_data.geo.values.tolist()
  if set(frame["geo"]) != set(geos):
    raise ValueError("Hypothetical data must contain exactly the trained geos.")
  times = sorted(frame["time"].unique())
  indexed = frame.set_index(["geo", "time"], verify_integrity=True)
  ordered = indexed.reindex(pd.MultiIndex.from_product([geos, times]))
  if ordered.isna().any().any():
    raise ValueError("Hypothetical data has missing values or geo/time rows.")

  def array(columns: list[str]):
    values = (
        ordered[columns].to_numpy().reshape(len(geos), len(times), len(columns))
    )
    return backend.to_tensor(values, dtype=backend.float_dtype)

  channels = mmm.input_data.media_channel.values.tolist()
  return tensors.DataTensors(
      media=array([f"{ch}_impression" for ch in channels]),
      media_spend=array([f"{ch}_spend" for ch in channels]),
      non_media_treatments=array(["Promo"]),
      organic_media=array(["Organic_channel0_impression"]),
      revenue_per_kpi=array(["revenue_per_conversion"])[..., 0],
      time=times,
  )


def run(options: config.Options) -> None:
  if options.action != "hypothetical":
    pipeline.run_model(options, lambda: build_model(options))
    return
  mmm = pipeline.load_model(options.model_path)
  pipeline.require_convergence(mmm, allow_unconverged=options.allow_unconverged)
  frame = data.read_example("hypothetical_geo_all_channels.csv", options.data)
  new_data = hypothetical_data(mmm, frame)
  result = optimizer.BudgetOptimizer(
      analyzer=pipeline.get_analyzer(mmm)
  ).optimize(
      new_data=new_data,
      budget=50_000_000,
      pct_of_spend=[0.2, 0.1, 0.2, 0.2, 0.3],
  )
  options.output.mkdir(parents=True, exist_ok=True)
  result.output_optimization_summary(
      "hypothetical_optimization.html", str(options.output)
  )
