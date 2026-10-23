"""Track a training run locally; inspecting runs never repeats sampling."""

import dataclasses

from meridian_workflows import config
from meridian_workflows import data
from meridian_workflows import pipeline


def run(options: config.Options) -> None:
  def build():
    return data.make_model(
        data.read_example("geo_all_channels.csv", options.data)
    )

  if options.action not in ("train", "runs"):
    pipeline.run_model(options, build)
    return
  import mlflow

  options.output.mkdir(parents=True, exist_ok=True)
  database = (options.output / "mlflow.db").resolve().as_posix()
  mlflow.set_tracking_uri(f"sqlite:///{database}")
  experiment_name = "meridian-python-examples"
  experiment = mlflow.get_experiment_by_name(experiment_name)
  if experiment is None:
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=(options.output / "artifacts").resolve().as_uri(),
    )
  else:
    experiment_id = experiment.experiment_id
  experiment = mlflow.set_experiment(experiment_id=experiment_id)
  if options.action == "runs":
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    runs.to_csv(options.output / "runs.csv", index=False)
    print(runs.to_string(index=False))
    return
  from meridian.mlflow import autolog

  autolog.autolog(log_metrics=True)
  try:
    with mlflow.start_run(run_name="python-demo"):
      mlflow.log_params(dataclasses.asdict(options.sampling))
      mlflow.set_tag("sampling_profile", options.profile)
      pipeline.train(build(), options)
      mlflow.log_artifact(str(options.output / "model.binpb"))
      mlflow.log_artifact(str(options.output / "model.json"))
  finally:
    autolog.autolog(disable=True)
