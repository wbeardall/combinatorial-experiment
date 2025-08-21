import copy
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Type, Union

import yaml

# NOTE: This is NOT the standard multiprocessing library, which uses pickle;
# it's the one from Pathos, which uses Dill
from combinatorial_experiment.combinatorial_experiment import (
    CombinatorialExperiment,
    MissingCacheError,
    experiment_complete,
)
from combinatorial_experiment.utils import camel_to_snake_case
from combinatorial_experiment.variables import (
    VariableCollection,
    deserialize_experiment_config,
)

logger = logging.getLogger(__name__)


def iteration_dir(experiment_dir: str, iteration: int) -> str:
    return f"{experiment_dir}/_i_{iteration:03d}_"


def iteration_of_dir(experiment_dir: str) -> int:
    basename = os.path.basename(experiment_dir)
    return int(re.match(r"_i_(\d+)_", basename).group(1))


@dataclass
class Generated:
    variables: VariableCollection
    experiment_dir: str
    iteration: int
    base_config: Optional[dict] = None


@dataclass
class LastIteration:
    iteration: int
    path: str


def hook_type(o):
    if isinstance(o, type):
        name = o.__name__
    else:
        name = o.__class__.__name__
    return camel_to_snake_case(name.replace("Hook", ""))


class IterativeExperimentHook(ABC):
    __registry__: Dict[str, Type["IterativeExperimentHook"]] = {}
    experiment_dir: str
    variables: VariableCollection

    def __init_subclass__(cls, /, *, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register:
            IterativeExperimentHook.__registry__[hook_type(cls)] = cls

    @property
    def hook_type(self) -> str:
        return hook_type(self)

    def __init__(self, experiment_dir: str, variables: VariableCollection):
        self.experiment_dir = str(experiment_dir)
        self.variables = copy.deepcopy(variables)

    @classmethod
    @abstractmethod
    def _from_config_and_variables(
        cls, *, experiment_dir: str, variables: VariableCollection, config: dict
    ): ...

    @classmethod
    def _find_hook_type_and_create(
        cls, *, experiment_dir: str, variables: VariableCollection, config: dict
    ):
        ht = config.get("hook_type", None)
        if ht == hook_type(cls) or ht is None:
            return cls._from_config_and_variables(
                experiment_dir=experiment_dir, variables=variables, config=config
            )
        cls = IterativeExperimentHook.__registry__.get(ht, None)
        if cls is None:
            raise ValueError(f"Unknown hook type: {ht}")
        return cls._from_config_and_variables(
            experiment_dir=experiment_dir, variables=variables, config=config
        )

    @abstractmethod
    def serialize_hook_config(self) -> dict: ...

    def to_config(self) -> dict:
        return {
            "variables": self.variables.serialize(),
            "hook_config": self.serialize_hook_config(),
        }

    @classmethod
    def from_config(
        cls, *, experiment_dir: str, config: Union[str, dict]
    ) -> "IterativeExperimentHook":
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        assert isinstance(config, dict)
        if "variables" not in config:
            raise KeyError("'variables' not found in config")
        if "hook_config" not in config:
            raise KeyError("'hook_config' not found in config")

        variables = deserialize_experiment_config(config["variables"])
        return cls._find_hook_type_and_create(
            experiment_dir=experiment_dir,
            variables=variables,
            config=config["hook_config"],
        )

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


class IterativeExperiment:
    hook: IterativeExperimentHook
    experiment_function: Callable
    resume: bool
    base_config: dict
    additional_config: dict
    job_timeout: int
    serialize: bool
    run_in_band: bool
    dry_run: bool
    continue_on_failure: bool
    archive_on_complete: bool

    def __init__(
        self,
        hook: IterativeExperimentHook,
        experiment_function: Callable = None,
        resume: bool = True,
        base_config: dict = {},
        additional_config: dict = {},
        job_timeout: int = None,
        serialize: bool = False,
        run_in_band: bool = False,
        dry_run: bool = False,
        continue_on_failure: bool = False,
        archive_on_complete: bool = False,
    ):
        self.hook = hook
        self.experiment_function = experiment_function
        self.resume = resume
        self.base_config = base_config
        self.additional_config = additional_config
        self.job_timeout = job_timeout
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
                cand = iteration_dir(self.hook.experiment_dir, i)
                if os.path.exists(cand):
                    if not experiment_complete(cand):
                        last_experiment_dir = cand
                else:
                    break
                i += 1
            if last_experiment_dir is not None:
                logger.info(f"Resuming experiment from '{last_experiment_dir}'")
                try:
                    CombinatorialExperiment.resume(last_experiment_dir)
                except MissingCacheError:
                    logger.warning(f"Cache file not found in {last_experiment_dir}")
                    logger.warning("Continuing with new experiment")

        for g in self.hook:
            i = g.iteration
            logger.info(f"Running iteration {g.iteration} in '{g.experiment_dir}'")
            # NOTE: We don't pass a database path here, because the database
            # must be in the experiment directory for proper result selection!
            if g.base_config is None:
                base_config = self.base_config
            else:
                base_config = OmegaConf.to_object(
                    OmegaConf.merge(
                        OmegaConf.create(self.base_config),
                        OmegaConf.create(g.base_config),
                    )
                )

            experiment = CombinatorialExperiment(
                experiment_function=self.experiment_function,
                variables=g.variables,
                experiment_dir=g.experiment_dir,
                resume=self.resume,
                base_config=base_config,
                additional_config=self.additional_config,
                job_timeout=self.job_timeout,
                autoname=False,
                repeats=1,
                serialize=self.serialize,
                run_in_band=self.run_in_band,
                dry_run=self.dry_run,
                continue_on_failure=self.continue_on_failure,
                archive_on_complete=self.archive_on_complete,
                # Disable safe save because actual directories are handled by the
                # IterativeExperiment
                disable_safe_save=True,
            )
            experiment.run()
        logger.info(f"Completed iterative experiment ({i + 1} iterations)")
