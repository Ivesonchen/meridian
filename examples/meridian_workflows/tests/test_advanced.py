"""Check advanced integrations without cloud access or a full training run."""

import dataclasses
from unittest import mock

import arviz as az
from meridian_workflows import config
from meridian_workflows import full_funnel_analysis
from meridian_workflows import pipeline
from meridian_workflows.demos import full_funnel
from meridian_workflows.demos import mlflow_demo
from meridian_workflows.demos import prior_calibration
from meridian_workflows.demos import scenario_planner
import numpy as np
import pytest


def test_calibrated_priors_can_be_saved_and_loaded(tmp_path):
  mmm = prior_calibration.build_model(config.Options(tmp_path))
  mmm.sample_prior(4, seed=3)
  pipeline.save_model(mmm, tmp_path / "calibrated.binpb")
  restored = pipeline.load_model(tmp_path / "calibrated.binpb")
  assert restored.inference_data.prior.sizes["draw"] == 4
  np.testing.assert_allclose(
      restored.inference_data.prior.roi_m, mmm.inference_data.prior.roi_m
  )


def test_full_funnel_equations_and_plots_use_standalone_helpers(tmp_path):
  options = config.Options(tmp_path)
  stage1 = full_funnel.build_model(options, 1)
  stage2 = full_funnel.build_model(options, 2)
  stage1.sample_prior(4, seed=1)
  stage2.sample_prior(4, seed=2)
  # Test algebra using prior draws, not a claim of a fitted posterior.
  analysis = full_funnel_analysis.AnalyzerFullFunnel(
      stage2,
      {"BrandedGQV_control": stage1},
      inference_data=az.InferenceData(posterior=stage2.inference_data.prior),
      inference_data_mediators={
          "BrandedGQV_control": az.InferenceData(
              posterior=stage1.inference_data.prior
          )
      },
  )
  remaining = analysis.incremental_outcome(
      effect_type="remaining", include_non_paid_channels=False
  )
  indirect = analysis.incremental_outcome(
      effect_type="indirect", include_non_paid_channels=False
  )
  total = analysis.incremental_outcome(
      effect_type="total", include_non_paid_channels=False
  )
  np.testing.assert_allclose(total, remaining + indirect, rtol=1e-8, atol=1e-6)
  assert total.shape == (1, 4, 3)
  assert np.isfinite(total).all()
  decomposition = full_funnel_analysis.compute_roi_decomposition(analysis)
  np.testing.assert_allclose(
      decomposition.total_roi_mean,
      decomposition.remaining_roi_mean + decomposition.indirect_roi_mean,
  )
  full_funnel_analysis.plot_roi_decomposition(analysis).save(
      str(tmp_path / "decomposition.html")
  )
  full_funnel_analysis.plot_compounding_adstock_decay(analysis).save(
      str(tmp_path / "adstock.html")
  )
  assert 0 <= float(analysis.negative_baseline_probability()) <= 1


def test_mlflow_tracks_locally_without_resampling_for_runs(tmp_path):
  import mlflow

  options = config.Options(tmp_path, action="train")

  def fake_train(mmm, opts):
    (opts.output / "model.binpb").write_bytes(b"test artifact")
    (opts.output / "model.json").write_text('{"profile": "smoke"}')

  with (
      mock.patch.object(mlflow_demo.data, "make_model"),
      mock.patch.object(
          mlflow_demo.pipeline, "train", side_effect=fake_train
      ) as train,
  ):
    mlflow_demo.run(options)
    mlflow_demo.run(dataclasses.replace(options, action="runs"))
  assert train.call_count == 1
  assert (tmp_path / "runs.csv").stat().st_size > 0
  experiment = mlflow.get_experiment_by_name("meridian-python-examples")
  runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
  assert runs.iloc[0]["tags.sampling_profile"] == "smoke"
  assert runs.iloc[0]["artifact_uri"].startswith(
      (tmp_path / "artifacts").as_uri()
  )
  assert list((tmp_path / "artifacts").rglob("model.binpb"))


def test_scenario_upload_requires_exported_tables(tmp_path):
  with pytest.raises(FileNotFoundError, match="export first"):
    scenario_planner.upload_tables(config.Options(tmp_path))


def test_scenario_upload_is_explicit_and_uses_adc(tmp_path):
  from scenarioplanner.converters import sheets
  from scenarioplanner.linkingapi import url_generator

  table_dir = tmp_path / "tables"
  table_dir.mkdir()
  (table_dir / "ModelFit.csv").write_text("time,actual\n2021-01-25,1\n")
  with (
      mock.patch(
          "google.auth.default", return_value=(object(), "test-project")
      ) as credentials,
      mock.patch.object(sheets, "upload_to_gsheet") as upload,
      mock.patch.object(
          url_generator,
          "create_report_url",
          return_value="https://example.test/report",
      ),
  ):
    upload.return_value.url = "https://example.test/sheet"
    scenario_planner.run(config.Options(tmp_path, action="upload"))
  assert credentials.call_count == upload.call_count == 1
  assert "ModelFit" in upload.call_args.args[0]
  assert (tmp_path / "scenario_planner_url.txt").exists()
