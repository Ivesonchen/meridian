"""Tests never select the user's GPU."""

import os

os.environ["MERIDIAN_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
