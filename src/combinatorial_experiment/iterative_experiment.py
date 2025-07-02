import copy
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Union

# NOTE: This is NOT the standard multiprocessing library, which uses pickle;
# it's the one from Pathos, which uses Dill
from combinatorial_experiment.combinatorial_experiment import (
    CombinatorialExperiment,
    experiment_complete,
)
from combinatorial_experiment.variables import VariableCollection


def iteration_dir(experiment_dir: str, iteration: int) -> str:
    return f"{experiment_dir}/iteration_{iteration}"


@dataclass
class Generated:
    variables: VariableCollection
    experiment_dir: str


@dataclass
class LastIteration:
    iteration: int
    path: str


class IterativeExperimentHook(ABC):
    experiment_dir: str
    variables: VariableCollection

    def __init__(self, experiment_dir: str, variables: VariableCollection):
        self.experiment_dir = experiment_dir
        self.variables = copy.deepcopy(variables)

    def get_last_completed_iteration(self) -> Optional[LastIteration]:
        i = 0
        last_iteration = None
        while True:
            cand = iteration_dir(self.experiment_dir, i)
            zip_cand = f"{cand}.zip"
            if os.path.exists(cand):
                if experiment_complete(cand):
                    last_iteration = LastIteration(iteration=i, path=cand)
            elif os.path.exists(zip_cand):
                last_iteration = LastIteration(iteration=i, path=zip_cand)
            else:
                break
            i += 1
        return last_iteration

    @abstractmethod
    def generate_variables(self) -> Union[Generated, None]:
        """
        Generate a new set of variables for the experiment.

        Must return a Generated object, or None when the experiment is complete.
        """
        ...

    def __iter__(self) -> Iterator[Generated]:
        while True:
            r = self.generate_variables()
            if r is None:
                break
            yield r


class IteratedExperiment:
    hook: IterativeExperimentHook
    experiment_function: Callable
    database_path: str
    resume: bool
    base_config: dict
    additional_config: dict
    job_timeout: int
    autoname: bool
    repeats: int
    serialize: bool
    run_in_band: bool
    dry_run: bool
    continue_on_failure: bool
    archive_on_complete: bool

    def __init__(
        self,
        hook: IterativeExperimentHook,
        experiment_function: Callable = None,
        variables: VariableCollection = None,
        database_path: str = None,
        experiment_dir: str = os.getcwd(),
        resume: bool = True,
        base_config: dict = {},
        additional_config: dict = {},
        job_timeout: int = None,
        autoname: bool = False,
        repeats: int = 1,
        serialize: bool = False,
        run_in_band: bool = False,
        dry_run: bool = False,
        continue_on_failure: bool = False,
        archive_on_complete: bool = False,
    ):
        self.hook = hook
        self.experiment_function = experiment_function
        self.variables = variables
        self.database_path = database_path
        self.experiment_dir = experiment_dir
        self.resume = resume
        self.base_config = base_config
        self.additional_config = additional_config
        self.job_timeout = job_timeout
        self.autoname = autoname
        self.repeats = repeats
        self.serialize = serialize
        self.run_in_band = run_in_band
        self.dry_run = dry_run
        self.continue_on_failure = continue_on_failure
        self.archive_on_complete = archive_on_complete

    def run(self):
        # Check if we should resume a previous experiment
        if self.resume:
            last_experiment_dir = None
            i = 0
            while True:
                cand = iteration_dir(self.experiment_dir, i)
                if os.path.exists(cand):
                    if not experiment_complete(cand):
                        last_experiment_dir = cand
                else:
                    break
                i += 1
            if last_experiment_dir is not None:
                CombinatorialExperiment.resume(last_experiment_dir)

        for g in self.hook:
            experiment = CombinatorialExperiment(
                experiment_function=self.experiment_function,
                variables=g.variables,
                database_path=self.database_path,
                experiment_dir=g.experiment_dir,
                resume=self.resume,
                base_config=self.base_config,
                additional_config=self.additional_config,
                job_timeout=self.job_timeout,
                autoname=self.autoname,
                repeats=self.repeats,
                serialize=self.serialize,
                run_in_band=self.run_in_band,
                dry_run=self.dry_run,
                continue_on_failure=self.continue_on_failure,
                archive_on_complete=self.archive_on_complete,
            )
            experiment.run()
