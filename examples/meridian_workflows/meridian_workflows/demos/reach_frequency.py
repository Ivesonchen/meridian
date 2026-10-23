"""Three impression-based channels plus one reach/frequency channel."""

from meridian.model import model
from meridian_workflows import config
from meridian_workflows import data
from meridian_workflows import pipeline


def build_model(options: config.Options) -> model.Meridian:
  frame = data.read_example("geo_media_rf.csv", options.data)
  return data.make_model(frame, reach_frequency=True)


def run(options: config.Options) -> None:
  pipeline.run_model(options, lambda: build_model(options))
