"""A deliberately tiny real CPU integration test, not a scientific fit."""

import dataclasses
import json
from unittest import mock

from meridian.model import model
from meridian.model import spec
from meridian_workflows import config
from meridian_workflows import data
from meridian_workflows import pipeline
from meridian_workflows.demos import response_curves
from meridian_workflows.demos import mlflow_demo
from meridian_workflows.demos import scenario_planner
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def sampled_model(tmp_path_factory):
  directory = tmp_path_factory.mktemp("sampled_model")
  frame = data.read_example("geo_all_channels.csv")
  frame = frame[
      frame.geo.isin(["Geo0", "Geo1", "Geo2"])
      & frame.time.isin(sorted(frame.time.unique())[:12])
  ]
  mmm = model.Meridian(
      input_data=data.build_input(frame),
      model_spec=spec.ModelSpec(knots=2, max_lag=2),
  )
  options = config.Options(
      directory,
      action="train",
      sampling=config.Sampling((2,), adapt=10, burnin=2, keep=8, prior=8),
  )
  pipeline.train(mmm, options)
  return mmm, options


def test_real_sampling_save_load_round_trip(sampled_model):
  original, options = sampled_model
  restored = pipeline.load_model(options.model_path)
  assert restored.inference_data.posterior.sizes["chain"] == 2
  assert restored.inference_data.posterior.sizes["draw"] == 8
  np.testing.assert_allclose(
      restored.inference_data.posterior.beta_gm,
      original.inference_data.posterior.beta_gm,
  )
  metadata = json.loads(options.model_path.with_suffix(".json").read_text())
  assert metadata["posterior_seconds"] > 0
  assert metadata["profile"] == "smoke"


def test_real_html_report_from_saved_draws(sampled_model):
  _, options = sampled_model
  restored = pipeline.load_model(options.model_path)
  pipeline.report(restored, options.output)
  for name in (
      "health_card.html",
      "rhat.html",
      "model_fit.html",
      "summary.html",
      "residual_diagnostics.html",
      "residual_diagnostics.csv",
  ):
    assert (options.output / name).stat().st_size > 0
  residuals = pd.read_csv(options.output / "residual_diagnostics.csv")
  assert len(residuals) == 4
  assert residuals.n_observations.tolist() == [12] * 4
  assert residuals.outcome_scale.unique().tolist() == ["revenue"]
  assert np.isfinite(residuals.durbin_watson).all()
  assert residuals.jarque_bera_p_value.between(0, 1).all()


def test_real_roi_and_response_curve_outputs(sampled_model):
  _, options = sampled_model
  response_curves.run(
      dataclasses.replace(options, action="analyze", allow_unconverged=True)
  )
  for name in (
      "roi_credible_intervals.csv",
      "mroi_credible_intervals.csv",
      "posterior_uncertainty.csv",
      "response_curves.html",
  ):
    assert (options.output / name).stat().st_size > 0


@pytest.mark.parametrize("quarterly", (False, True))
def test_real_scenario_planner_export_is_local(
    sampled_model, tmp_path, quarterly
):
  mmm, options = sampled_model
  scenario_planner.export_tables(
      mmm,
      dataclasses.replace(
          options,
          output=tmp_path,
          action="export",
          allow_unconverged=True,
          quarterly=quarterly,
      ),
  )
  assert (tmp_path / "scenario.binpb").stat().st_size > 0
  paths = list((tmp_path / "tables").glob("*.csv"))
  assert paths
  assert all(path.stat().st_size > 0 for path in paths)


def test_real_mlflow_training_saves_in_its_output_directory(
    sampled_model, tmp_path
):
  original, options = sampled_model
  training_model = model.Meridian(
      input_data=original.input_data, model_spec=original.model_spec
  )
  with mock.patch.object(
      mlflow_demo.data, "make_model", return_value=training_model
  ):
    mlflow_demo.run(
        dataclasses.replace(options, output=tmp_path, action="train")
    )
  restored = pipeline.load_model(tmp_path / "model.binpb")
  assert restored.inference_data.posterior.sizes["draw"] == 8
  assert list((tmp_path / "artifacts").rglob("model.binpb"))
