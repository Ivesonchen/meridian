"""Legacy TensorFlow variant, including the notebook's pre-modeling EDA."""

from meridian import backend
from meridian_workflows import config
from meridian_workflows.demos import getting_started


def run(options: config.Options) -> None:
  if backend.config.get_backend().value != "tensorflow":
    raise ValueError(
        "Configure the TensorFlow backend before running this demo."
    )
  if options.action == "eda":
    from meridian.model.eda import meridian_eda

    mmm = getting_started.build_model(options)
    options.output.mkdir(parents=True, exist_ok=True)
    meridian_eda.MeridianEDA(
        mmm,
        n_draws_prior=options.sampling.prior,
        seed=options.sampling.seed,
    ).generate_and_save_report("eda.html", str(options.output))
  else:
    getting_started.run(options)
