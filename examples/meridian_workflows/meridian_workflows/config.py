"""Explicit runtime and sampling settings, independent of notebook state."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys


@dataclass(frozen=True)
class Sampling:
  chains: tuple[int, ...] = (2,)
  adapt: int = 200
  burnin: int = 100
  keep: int = 200
  prior: int = 200
  seed: int = 0

  def __post_init__(self):
    if not self.chains or any(n < 1 for n in self.chains):
      raise ValueError("Every chain batch must contain at least one chain.")
    if self.adapt < 1 or self.keep < 1 or self.prior < 1 or self.burnin < 0:
      raise ValueError("adapt, keep and prior must be positive; burnin >= 0.")

  def posterior_kwargs(self) -> dict:
    return dict(
        n_chains=list(self.chains),
        n_adapt=self.adapt,
        n_burnin=self.burnin,
        n_keep=self.keep,
        seed=self.seed,
    )


@dataclass(frozen=True)
class Options:
  output: Path
  action: str = "prepare"
  profile: str = "smoke"
  sampling: Sampling = Sampling()
  data: Path | None = None
  model: Path | None = None
  overwrite: bool = False
  allow_unconverged: bool = False
  quarterly: bool = False
  spreadsheet_name: str = "Meridian Python demo"

  @property
  def model_path(self) -> Path:
    return self.model or self.output / "model.binpb"


def configure_runtime(backend: str = "jax", device: str = "cpu") -> None:
  """Configure once, before importing any Meridian or GPU computation module."""
  if backend not in ("jax", "tensorflow") or device not in ("cpu", "cuda"):
    raise ValueError("Use backend jax/tensorflow and device cpu/cuda.")
  if device == "cuda" and sys.platform != "linux":
    raise RuntimeError(
        "This CUDA example requires Linux. On Windows, use WSL Ubuntu; "
        "otherwise select --device cpu."
    )
  if "meridian.backend" in sys.modules:
    raise RuntimeError(
        "Configure the runtime before importing Meridian. Use a new Python "
        "process to change backend or device."
    )
  os.environ["MERIDIAN_BACKEND"] = backend
  os.environ["JAX_PLATFORMS"] = "cuda" if device == "cuda" else "cpu"
  if device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  os.environ.setdefault(
      "JAX_COMPILATION_CACHE_DIR",
      str(Path.home() / ".cache" / "meridian-workflows" / "jax"),
  )
  os.environ.setdefault("MPLBACKEND", "Agg")


def check_runtime(backend_name: str, device: str) -> None:
  """Fail explicitly if the requested device is unavailable."""
  from meridian import backend

  if backend_name == "jax":
    import jax

    devices = jax.devices()
    if device == "cuda" and not any(d.platform == "gpu" for d in devices):
      raise RuntimeError("CUDA was requested but JAX has no GPU device.")
    logging.info(
        "JAX devices: %s; float dtype: %s", devices, backend.float_dtype
    )
  else:
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    if device == "cuda" and not devices:
      raise RuntimeError("CUDA was requested but TensorFlow has no GPU device.")
    for gpu in devices:
      tf.config.experimental.set_memory_growth(gpu, True)
    logging.info("TensorFlow GPU devices: %s", devices)
