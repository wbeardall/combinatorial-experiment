import logging
import os

import click


@click.command()
@click.argument("experiment_dir")
def resume_experiment(experiment_dir):
    """Resume an experiment."""
    from combinatorial_experiment.combinatorial_experiment import (
        CombinatorialExperiment,
        experiment_complete,
    )

    if not os.path.exists(experiment_dir):
        raise RuntimeError(f"Experiment directory does not exist: '{experiment_dir}'")
    if not os.path.isdir(experiment_dir):
        raise RuntimeError(f"'{experiment_dir}' is not a directory.")
    if experiment_complete(experiment_dir):
        logging.info(f"Experiment is already completed: '{experiment_dir}'")
        return

    experiment = CombinatorialExperiment(
        experiment_dir=experiment_dir, resume=True, disable_safe_save=True
    )
    experiment.run()


if __name__ == "__main__":
    resume_experiment()
