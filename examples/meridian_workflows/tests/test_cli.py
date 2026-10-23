import dataclasses
import importlib
import subprocess
import sys
from unittest import mock

from meridian_workflows import cli
from meridian_workflows import config
import pytest


def test_list_does_not_import_computational_backends():
  code = (
      "import sys; from meridian_workflows.cli import main; main(['list']); "
      "assert 'jax' not in sys.modules; assert 'tensorflow' not in sys.modules"
  )
  result = subprocess.run(
      [sys.executable, "-c", code], capture_output=True, text=True, check=True
  )
  for name in cli.DEMOS:
    assert name in result.stdout


@pytest.mark.parametrize("name", cli.DEMOS)
def test_every_demo_has_a_callable_entry_point(name):
  module = importlib.import_module(
      f"meridian_workflows.demos.{cli.DEMOS[name][0]}"
  )
  assert callable(module.run)


def test_dispatch_explicit_sampling(tmp_path):
  demo = mock.Mock()
  with (
      mock.patch.object(cli.config, "configure_runtime") as setup,
      mock.patch.object(cli.config, "check_runtime"),
      mock.patch.object(cli.importlib, "import_module", return_value=demo),
  ):
    assert (
        cli.main(
            [
                "getting-started",
                "train",
                "--device",
                "cuda",
                "--output-dir",
                str(tmp_path),
                "--chains",
                "2,2",
                "--keep",
                "55",
            ]
        )
        == 0
    )
  setup.assert_called_once_with("jax", "cuda")
  options = demo.run.call_args.args[0]
  assert options.sampling.chains == (2, 2)
  assert options.sampling.keep == 55
  assert options.profile == "smoke"


@pytest.mark.parametrize(
    "arguments",
    [
        ["getting-started", "upload"],
        ["getting-started", "--chains", "0"],
        ["getting-started", "--keep", "-1"],
        ["getting-started-tf", "--backend", "jax"],
        ["prior-calibration", "--backend", "tensorflow"],
        ["mlflow", "train", "--model", "existing.binpb"],
    ],
)
def test_invalid_arguments_fail_before_initializing_gpu(arguments):
  with mock.patch.object(config, "configure_runtime") as setup:
    with pytest.raises(SystemExit) as error:
      cli.main(arguments)
    assert error.value.code == 2
  setup.assert_not_called()


def test_sampling_preserves_total_chains_and_fresh_kwargs():
  sampling = cli.notebook_sampling("getting-started")
  assert sum(sampling.chains) == 10
  assert sampling.keep == 1000
  kwargs = sampling.posterior_kwargs()
  kwargs["n_chains"].append(99)
  assert sum(sampling.chains) == 10
  with pytest.raises(ValueError):
    dataclasses.replace(sampling, adapt=0)


def test_options_does_not_create_outputs(tmp_path):
  options = config.Options(tmp_path / "new")
  assert options.model_path == tmp_path / "new" / "model.binpb"
  assert not options.output.exists()


def test_runtime_must_be_configured_before_backend_import(monkeypatch):
  monkeypatch.setitem(sys.modules, "meridian.backend", object())
  with pytest.raises(RuntimeError, match="before importing"):
    config.configure_runtime()


def test_native_windows_cuda_is_rejected(monkeypatch):
  monkeypatch.setattr(config.sys, "platform", "win32")
  with pytest.raises(RuntimeError, match="WSL"):
    config.configure_runtime(device="cuda")


def test_tensorflow_prepare_in_separate_cpu_process(tmp_path):
  result = subprocess.run(
      [
          sys.executable,
          "-m",
          "meridian_workflows",
          "getting-started-tf",
          "prepare",
          "--device",
          "cpu",
          "--output-dir",
          str(tmp_path),
      ],
      capture_output=True,
      text=True,
  )
  assert result.returncode == 0, result.stderr
  assert (tmp_path / "data_summary.json").exists()


def test_tensorflow_eda_in_separate_cpu_process(tmp_path):
  result = subprocess.run(
      [
          sys.executable,
          "-m",
          "meridian_workflows",
          "getting-started-tf",
          "eda",
          "--device",
          "cpu",
          "--prior",
          "4",
          "--output-dir",
          str(tmp_path),
      ],
      capture_output=True,
      text=True,
  )
  assert result.returncode == 0, result.stderr
  assert (tmp_path / "eda.html").stat().st_size > 0
