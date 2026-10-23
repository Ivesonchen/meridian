import importlib
import json
from types import SimpleNamespace
from unittest import mock

import arviz as az
from meridian.model import model
from meridian_workflows import cli
from meridian_workflows import config
from meridian_workflows import data
from meridian_workflows import pipeline
from meridian_workflows.demos import full_funnel
from meridian_workflows.demos import getting_started
from meridian_workflows.demos import rf_simulation
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "name",
    [
        "getting-started",
        "reach-frequency",
        "response-curves",
        "full-funnel",
        "scenario-planner",
        "mlflow",
        "rf-simulation",
    ],
)
def test_prepare_uses_real_data_but_never_samples(name, tmp_path):
  module = importlib.import_module(
      f"meridian_workflows.demos.{cli.DEMOS[name][0]}"
  )
  with mock.patch.object(model.Meridian, "sample_posterior") as sample:
    module.run(config.Options(tmp_path))
  sample.assert_not_called()
  if name not in ("full-funnel", "rf-simulation"):
    summary = json.loads((tmp_path / "data_summary.json").read_text())
    assert summary["geos"] > 1
    assert summary["time_periods"] > 1


def test_getting_started_preserves_non_paid_variables(tmp_path):
  mmm = getting_started.build_model(config.Options(tmp_path))
  assert mmm.input_data.media.sizes["media_channel"] == 5
  assert mmm.input_data.organic_media.sizes["organic_media_channel"] == 1
  assert mmm.input_data.non_media_treatments.sizes["non_media_channel"] == 1
  assert mmm.model_spec.enable_aks


def test_hypothetical_data_alignment_and_missing_rows(tmp_path):
  mmm = getting_started.build_model(config.Options(tmp_path))
  frame = data.read_example("hypothetical_geo_all_channels.csv")
  forward = getting_started.hypothetical_data(mmm, frame)
  shuffled = getting_started.hypothetical_data(
      mmm, frame.sample(frac=1, random_state=4)
  )
  np.testing.assert_array_equal(forward.media, shuffled.media)
  assert forward.media.shape[0] == 40
  assert forward.media.shape[-1] == 5
  with pytest.raises(ValueError, match="missing"):
    getting_started.hypothetical_data(mmm, frame.iloc[1:])
  with pytest.raises(ValueError):
    getting_started.hypothetical_data(mmm, pd.concat([frame, frame.iloc[:1]]))


def test_report_loads_without_building_or_sampling(tmp_path):
  options = config.Options(tmp_path, action="report")
  build = mock.Mock()
  with (
      mock.patch.object(pipeline, "load_model") as load,
      mock.patch.object(pipeline, "report") as report,
  ):
    pipeline.run_model(options, build)
  build.assert_not_called()
  load.assert_called_once_with(tmp_path / "model.binpb")
  report.assert_called_once_with(load.return_value, tmp_path)


def test_existing_training_result_is_not_overwritten(tmp_path):
  (tmp_path / "model.binpb").write_bytes(b"existing")
  mmm = mock.Mock()
  with pytest.raises(FileExistsError, match="already exists"):
    pipeline.train(mmm, config.Options(tmp_path, action="train"))
  mmm.sample_prior.assert_not_called()
  mmm.sample_posterior.assert_not_called()
  assert (tmp_path / "model.binpb").read_bytes() == b"existing"


def test_model_saved_before_any_optional_reporting(tmp_path):
  mmm = mock.Mock()
  with mock.patch.object(pipeline, "save_model") as save:
    pipeline.train(mmm, config.Options(tmp_path, action="train"))
  save.assert_called_once_with(mmm, tmp_path / "model.binpb")
  assert json.loads((tmp_path / "model.json").read_text())["profile"] == "smoke"


def test_save_failure_keeps_previous_model(tmp_path):
  target = tmp_path / "model.binpb"
  target.write_bytes(b"previous")

  def fail(mmm, path):
    from pathlib import Path

    Path(path).write_bytes(b"incomplete")
    raise RuntimeError("simulated serialization failure")

  with mock.patch.object(
      pipeline.meridian_serde, "save_meridian", side_effect=fail
  ):
    with pytest.raises(RuntimeError, match="serialization"):
      pipeline.save_model(mock.Mock(), target)
  assert target.read_bytes() == b"previous"
  assert list(tmp_path.iterdir()) == [target]


def test_missing_model_never_starts_training(tmp_path):
  with pytest.raises(FileNotFoundError, match="Run train first"):
    pipeline.load_model(tmp_path / "missing.binpb")


def test_convergence_gate_rejects_separated_chains():
  rng = np.random.default_rng(10)
  values = rng.normal(size=(2, 500)) + np.array([0, 10])[:, None]
  mmm = SimpleNamespace(inference_data=az.from_dict(posterior={"beta": values}))
  with pytest.raises(ValueError, match="Convergence not established"):
    pipeline.require_convergence(mmm)
  pipeline.require_convergence(mmm, allow_unconverged=True)


def test_convergence_gate_rejects_nonfinite_parameters():
  rng = np.random.default_rng(5)
  inference = az.from_dict(
      posterior={
          "finite": rng.normal(size=(4, 1000)),
          "invalid": np.full((4, 1000), np.nan),
      }
  )
  with pytest.raises(ValueError, match="Convergence not established"):
    pipeline.require_convergence(SimpleNamespace(inference_data=inference))


def test_convergence_gate_does_not_broadcast_parameter_dimensions():
  rng = np.random.default_rng(4)
  inference = az.from_dict(
      posterior={
          "beta": rng.normal(size=(4, 2000, 3)),
          "mu": rng.normal(size=(4, 2000, 10)),
      }
  )
  pipeline.require_convergence(SimpleNamespace(inference_data=inference))
  inference.add_groups(
      sample_stats={"diverging": np.ones((4, 2000), dtype=bool)}
  )
  with pytest.raises(ValueError, match="divergences"):
    pipeline.require_convergence(SimpleNamespace(inference_data=inference))


def test_full_funnel_persists_stages_separately(tmp_path):
  options = config.Options(tmp_path, action="train")
  with (
      mock.patch.object(full_funnel, "build_model") as build,
      mock.patch.object(pipeline, "train") as train,
  ):
    full_funnel.run(options)
  assert build.call_count == train.call_count == 2
  assert train.call_args_list[0].args[1].output == tmp_path / "stage1"
  assert train.call_args_list[1].args[1].output == tmp_path / "stage2"
  assert (
      train.call_args_list[0].args[1].sampling.seed
      != train.call_args_list[1].args[1].sampling.seed
  )


def test_full_funnel_can_train_only_missing_stage(tmp_path):
  (tmp_path / "stage1").mkdir()
  (tmp_path / "stage1" / "model.binpb").write_bytes(b"stage1")
  with (
      mock.patch.object(full_funnel, "build_model"),
      mock.patch.object(pipeline, "train") as train,
  ):
    full_funnel.run(config.Options(tmp_path, action="train-stage2"))
  assert train.call_count == 1
  assert train.call_args.args[1].output == tmp_path / "stage2"


def test_simulation_is_reproducible_and_preserves_aggregation():
  result = rf_simulation.simulate(seed=7, n_geos=3, n_times=16)
  repeated = rf_simulation.simulate(seed=7, n_geos=3, n_times=16)
  pd.testing.assert_frame_equal(result.geo, repeated.geo)
  assert result.geo.shape[0] == 48
  assert np.isfinite(result.geo.select_dtypes("number")).all().all()
  assert (result.geo.Channel3_reach >= 0).all()
  np.testing.assert_allclose(
      result.geo.Channel3_reach * result.geo.Channel3_frequency,
      result.geo.Channel3_impression,
  )
  np.testing.assert_allclose(
      result.national.Channel3_reach * result.national.Channel3_frequency,
      result.geo.groupby("time").Channel3_impression.sum(),
  )
  np.testing.assert_allclose(
      result.national.conversions * result.national.revenue_per_conversion,
      (result.geo.conversions * result.geo.revenue_per_conversion)
      .groupby(result.geo.time)
      .sum(),
  )
  inputs = data.build_input(result.geo, reach_frequency=True)
  assert inputs.reach.shape == (3, 16, 1)
  assert result.ground_truth["roi_m"].shape == (3,)
  assert result.ground_truth["roi_rf"].shape == (1,)
