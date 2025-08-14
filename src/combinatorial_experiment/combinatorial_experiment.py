import argparse
import atexit
import copy
import datetime
import gc
import importlib
import inspect
import logging
import os
import shutil
import sqlite3
import sys
import time
import warnings
from functools import cached_property, partial
from typing import Callable

import dill as pickle

# NOTE: This is NOT the standard multiprocessing library, which uses pickle;
# it's the one from Pathos, which uses Dill
import multiprocess as mp
import pandas as pd
import yaml
from tqdm import tqdm

from .compression import zip_experiment
from .experiment import Experiment, ExperimentSet, ExperimentStatus
from .safe_unpickle import safe_load
from .tracking import update_job_state
from .utils import NestedDict, get_last, safe_save
from .variables import VariableCollection, deserialize_experiment_config

use_style = tuple(int(el) for el in pd.__version__.split(".")) > (1, 3, 0)
allowed_time_symbols = ["T", "t", "Time", "time"]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
)


def multiprocess_wrap(func, serialize: bool = True, dry_run: bool = False):
    def wrapper(q, config):
        try:
            out = func(config, dry_run=dry_run)
            if serialize:
                # For safety, serialize the output
                out = pickle.dumps(out)
            q.put(out)
        except Exception as e:
            out = {"error": e}
            if serialize:
                out = pickle.dumps(out)
            q.put(out)

    return wrapper


def experiment_complete(directory: str) -> bool:
    return os.path.exists(os.path.join(directory, ".run_complete"))


class MissingCacheError(FileNotFoundError):
    pass


# TODO: Consider rewriting to cast variables to list at start of run and store.
# This would allow much easier indexing, iterating, error checking, returning to
# failed runs, etc.
class CombinatorialExperiment(object):
    """Combinatorial Test object.

    This class runs arbitrary tests defined in experiment_function, with the
    variables provided. This class supports caching through Dill, which means
    that it is not secure. Never run a .tst object from an untrustworthy source,
    as the experiment_function can contain arbitrary (and possibly malicious) code.

    Args:
        experiment_function: The function to run.
        variables: The variables to run.
        database_path: The path to the database. Defaults to `experiment.db` in the experiment directory.
        experiment_dir: The directory to run the experiment in.
        resume: Whether to resume the experiment.
        base_config: The base config.
        additional_config: The additional config.
    """

    initialized = False
    _is_resuming = False
    _conn = None

    def __init__(
        self,
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
        disable_safe_save: bool = False,
    ):
        self.initialized = False
        update_job_state(state="running", on_fail="warn")
        # NOTE: cache is deprecated.
        if experiment_function is None or variables is None:
            assert (
                resume
            ), "Cannot resume experiment with no experiment function or variables"
        if experiment_function is not None:
            self._function_source = os.path.abspath(
                inspect.getfile(experiment_function)
            )
            self._experiment_fn_name = experiment_function.__name__
            if run_in_band:
                self.experiment_function = partial(experiment_function, dry_run=dry_run)
            else:
                self.experiment_function = multiprocess_wrap(
                    experiment_function, serialize=serialize, dry_run=dry_run
                )
        else:
            self._experiment_source = None
            self._experiment_fn_name = None
            self.experiment_function = None
        """
        if not isinstance(variables,(Variable, VariableCollection)):
            raise TypeError("Variables must be experiment.Variable or "
                            "experiment.VariableCollection. Did you mean to "
                            "use CombinatorialExperiment.from_config()?")"""
        self._serialize = serialize
        self._run_in_band = run_in_band
        self._dry_run = dry_run
        self._variables = variables
        self._resume = resume
        self._database_path = database_path
        self._continue_on_failure = continue_on_failure
        self._archive_on_complete = archive_on_complete
        if isinstance(base_config, str):
            with open(base_config, "r") as f:
                base_config = yaml.safe_load(f)
        base_config.update(additional_config)
        self.base_config = NestedDict(base_config)
        self.autoname = autoname
        self.job_timeout = job_timeout
        self.repeats = repeats
        self.disable_safe_save = disable_safe_save
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

        self.set_directories(experiment_dir, resume)

    @property
    def cache(self):
        return os.path.join(self._experiment_dir, ".cache")

    @classmethod
    def run_experiment(cls, experiment_function, add_cmd_args=[], argv=None):
        parser = cls.parser()
        if not isinstance(add_cmd_args, (list, tuple)):
            add_cmd_args = [add_cmd_args]
        for el in add_cmd_args:
            parser.add_argument(f"--{el}")
        if argv is None:
            argv = sys.argv[1:]
        args = parser.parse_args(argv)

        d = vars(args)
        experiment = cls.from_config(
            experiment_function=experiment_function,
            experiment_config=args.experiment_config,
            experiment_dir=args.output,
            database_path=args.database_path,
            base_config=args.base_config,
            additional_config={el: d[el] for el in add_cmd_args},
            job_timeout=args.job_timeout,
            autoname=args.autoname,
            repeats=args.repeats,
            dry_run=args.dry_run,
            run_in_band=args.run_in_band,
            serialize=args.serialize,
            continue_on_failure=args.continue_on_failure,
            archive_on_complete=args.archive_on_complete,
            disable_safe_save=args.disable_safe_save,
        )
        return experiment.run()

    @classmethod
    def from_config(
        cls,
        experiment_function: Callable,
        experiment_config: str,
        experiment_dir: str = os.getcwd(),
        base_config: dict = {},
        additional_config: dict = {},
        job_timeout: int = None,
        autoname: bool = False,
        database_path: str = None,
        repeats: int = 1,
        dry_run: bool = False,
        run_in_band: bool = False,
        serialize: bool = False,
        continue_on_failure: bool = False,
        archive_on_complete: bool = False,
        disable_safe_save: bool = False,
    ):
        variables = deserialize_experiment_config(experiment_config)
        return cls(
            experiment_function=experiment_function,
            variables=variables,
            experiment_dir=experiment_dir,
            database_path=database_path,
            base_config=base_config,
            additional_config=additional_config,
            job_timeout=job_timeout,
            autoname=autoname,
            repeats=repeats,
            dry_run=dry_run,
            run_in_band=run_in_band,
            serialize=serialize,
            continue_on_failure=continue_on_failure,
            archive_on_complete=archive_on_complete,
            disable_safe_save=disable_safe_save,
        )

    @classmethod
    def parser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--base_config")
        parser.add_argument("--experiment_config")
        parser.add_argument("--output", default=None)
        parser.add_argument("--repeats", type=int, default=1)
        parser.add_argument("--job_timeout", type=int, default=None)
        parser.add_argument("--autoname", action="store_true")
        parser.add_argument("--serialize", action="store_true")
        parser.add_argument("--run_in_band", action="store_true")
        parser.add_argument("--database_path", default=None)
        parser.add_argument("--dry_run", action="store_true")
        parser.add_argument("--continue_on_failure", action="store_true")
        parser.add_argument("--archive_on_complete", action="store_true")
        parser.add_argument("--disable_safe_save", action="store_true")
        return parser

    @classmethod
    def resume(cls, path):
        if not os.path.exists(os.path.join(path, "cache")):
            raise MissingCacheError(f"Cache file not found in {path}")
        experiment = CombinatorialExperiment(
            variables=[], experiment_dir=path, resume=True
        )
        experiment.run()

    def initialize(self):
        if self.initialized:
            return

        if self._database_path is None:
            self._database_path = os.path.join(self._experiment_dir, "experiment.db")

        database_dir = os.path.dirname(self._database_path)

        if not os.path.exists(database_dir):
            os.makedirs(database_dir)

        self._conn = sqlite3.connect(self._database_path)
        self._experiment_set = ExperimentSet(
            conn=self._conn, table_name=self.name, logger=self.logger
        )

        atexit.register(self._conn.close)
        if self._is_resuming:
            self._experiment_set.pull()
            logger.info(
                f"Resuming experiment ({self._experiment_set.count_completed} / {self._experiment_set.count} complete)"
            )
        else:
            # Only update experiment set if not resuming
            configs = self._variables.to_configs(self.base_config or {})
            self._experiment_set.update_experiments(configs, repeats=self.repeats)

    def add_variable(self, variable):
        self._variables.update(variable)

    def check_run_complete(self, directory=None):
        if directory is None:
            directory = self._experiment_dir
        return experiment_complete(directory)

    def maybe_archive_complete(self):
        if self._archive_on_complete:
            zip_experiment(self._experiment_dir)
            ephemeral = os.environ.get("EPHEMERAL", None)
            if ephemeral is not None:
                dest = os.path.join(ephemeral, os.path.basename(self._experiment_dir))
                logger.info(f"Archiving results to ephemeral directory '{dest}'.")
                shutil.move(self._experiment_dir, dest)

    def mark_run_complete(self):
        if self._experiment_set.complete:
            with open(os.path.join(self._experiment_dir, ".run_complete"), "w") as f:
                f.write(str(datetime.datetime.now()))
            update_job_state(state=ExperimentStatus.COMPLETED, on_fail="warn")
            self.maybe_archive_complete()
        else:
            warnings.warn("Experiment set is not complete. Not marking run complete.")

    def make_gitignore(self):
        gitignore = ["cache", "experiments", "input"]
        with open(os.path.join(self._experiment_dir, ".gitignore"), "w") as f:
            f.writelines(gitignore)

    def dump_configs(self):
        with open(os.path.join(self._experiment_dir, "base_config.yml"), "w") as f:
            yaml.dump(dict(self.base_config), f)
        with open(
            os.path.join(self._experiment_dir, "experiment_config.yml"), "w"
        ) as f:
            yaml.dump(self._variables.serialize(), f)

    def set_directories(self, experiment_dir, resume):
        experiment_dir = os.path.abspath(experiment_dir)
        if self.autoname:
            tail, head = os.path.split(experiment_dir)
            experiment_dir = os.path.join(
                tail, head + "_" + "_".join(self._variables.flattened_name)
            )
        if self.disable_safe_save:
            self._experiment_dir = experiment_dir
            return
        if resume:
            # Attempt to load last experiment
            try:
                experiment_dir = get_last(
                    experiment_dir, zero_ext=False, ignore_ext=True
                )
                if self.check_run_complete(experiment_dir):
                    experiment_dir = safe_save(
                        experiment_dir, zero_ext=False, ignore_ext=True
                    )
                    self._experiment_dir = experiment_dir
                    logger.info(
                        "\n\nStarting new experiment in {}\n\n".format(experiment_dir)
                    )
                else:
                    self._experiment_dir = experiment_dir
                    if os.path.exists(self.cache):
                        logger.info(
                            "\n\nResuming experiment from {}\n\n".format(experiment_dir)
                        )
                        self.deserialize()
                    # In case the experiment directory was already created (but empty)
                    else:
                        logger.info(
                            "\n\nStarting new experiment in {}\n\n".format(
                                experiment_dir
                            )
                        )
            except FileNotFoundError:
                experiment_dir = safe_save(
                    experiment_dir, zero_ext=False, ignore_ext=True
                )
                self._experiment_dir = experiment_dir
                logger.info(
                    "\n\nStarting new experiment in {}\n\n".format(experiment_dir)
                )
        else:
            experiment_dir = safe_save(experiment_dir, zero_ext=False, ignore_ext=True)
            self._experiment_dir = experiment_dir
            logger.info("\n\nStarting new experiment in {}\n\n".format(experiment_dir))
        if self._dry_run:
            atexit.register(partial(shutil.rmtree, self._experiment_dir))

    @property
    def name(self):
        return os.path.basename(self._experiment_dir)

    def setup(self):
        self.initialize()
        self.serialize()
        self.make_gitignore()

    @cached_property
    def logger(self):
        # Ensure the experiment directory exists
        if not os.path.exists(self._experiment_dir):
            os.makedirs(self._experiment_dir)
        # Set logging to experiment directory
        # Get logger for CombinatorialExperiment
        logger = logging.getLogger(str(self.__class__))
        # set the log level to INFO, DEBUG as the default is ERROR
        logger.setLevel(logging.INFO)
        filehandler = logging.FileHandler(
            os.path.join(self._experiment_dir, "logfile.log"), "a"
        )
        formatter = logging.Formatter(
            "%(asctime)-15s::%(levelname)s::%(filename)s::%(funcName)s::%(lineno)d::%(message)s"
        )
        filehandler.setFormatter(formatter)
        for hdlr in logger.handlers[:]:  # remove the existing file handlers
            if isinstance(hdlr, logging.FileHandler):
                logger.removeHandler(hdlr)
        logger.addHandler(filehandler)  # set the new handler
        return logger

    @property
    def _cache_dir(self):
        cache_dir = os.path.join(self._experiment_dir, "cache")
        return cache_dir

    @property
    def _cache_loc(self):
        cache_loc = os.path.join(
            self._cache_dir, f"{self._cache_base}{self._cache_ext}"
        )
        return cache_loc

    def run(self) -> pd.DataFrame:
        self.setup()
        self.dump_configs()
        to_run = self._experiment_set.get_incomplete()
        run_length = len(to_run)
        for i, experiment in enumerate(tqdm(to_run, total=run_length)):
            self.run_iteration(experiment, i, run_length)
            if self._resume:
                self.serialize()
        self.save_results(False)
        self.mark_run_complete()
        return self.records()

    def run_iteration(self, experiment: Experiment, i: int, run_length: int):
        config = copy.deepcopy(experiment.config)
        experiment_archive = os.path.join(self._experiment_dir, "experiments")
        savedir = os.path.join(experiment_archive, experiment.id)

        try:
            os.makedirs(savedir)
        except FileExistsError:
            # Resuming from this directory
            pass

        # Save config
        with open(os.path.join(savedir, "experiment_config.yml"), "w") as f:
            yaml.dump(config, f)

        experiment.update_metadata(
            metadata={"relpath": os.path.relpath(savedir, self._experiment_dir)}
        )
        config.update({"savedir": savedir})

        try:
            # TODO: Consider making this more flexible, i.e. constructing
            # kwargs from signature as reqd.
            outer_start = time.time()
            if self._run_in_band:
                output = self.experiment_function(config)
            else:
                q = mp.Queue()

                p = mp.Process(target=self.experiment_function, args=(q, config))

                p.start()

                # Deserialize
                output = q.get(timeout=self.job_timeout)
                if self._serialize:
                    output = pickle.loads(output)
                p.join()
            if "error" in output:
                raise RuntimeError(f"Experiment error: {output['error']}") from output[
                    "error"
                ]
            outer_walltime = time.time() - outer_start

            gc.collect()
            if "metrics" in output.keys():
                metrics = output["metrics"]
            else:
                metrics = output
            experiment.update(metrics=metrics, status=ExperimentStatus.COMPLETED)
        except Exception as e:
            outer_walltime = time.time() - outer_start
            self.logger.exception(
                "Fatal error in experiment #{} ({}):\n{}".format(i, savedir, e)
            )
            warnings.warn(str(e))
            experiment.update(status=ExperimentStatus.FAILED, error=str(e))
            if not (self._continue_on_failure):
                update_job_state(state=ExperimentStatus.FAILED, on_fail="warn")
                raise e
            # Ensure we aren't saving metrics from a partial run
            metrics = {}

        experiment.update_metadata(metadata={"walltime": outer_walltime})
        self.save_results(True)

    def records(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                el.as_record()
                for el in self._experiment_set.experiments.values()
                if el.status == ExperimentStatus.COMPLETED
            ]
        )

    def save_results(self, intermediate=False):
        records = self.records()
        if intermediate:
            prepend = "intermediate_"
        else:
            prepend = ""
        records.to_csv(
            os.path.join(self._experiment_dir, "{}results.csv".format(prepend))
        )

        with open(
            os.path.join(self._experiment_dir, "{}results.md".format(prepend)), "w"
        ) as f:
            if use_style:
                f.write(records.style.to_latex())
            else:
                f.write(records.to_latex(index=False, float_format="%0.3f"))

    def serialize(self):
        old_loc = self.cache + ".old"
        if os.path.exists(self.cache):
            os.rename(self.cache, old_loc)

        unpicklable_attrs = ["experiment_function", "_conn", "_experiment_set"]
        attrs = {k: v for k, v in self.__dict__.items() if k not in unpicklable_attrs}

        with open(self.cache, "wb") as f:
            pickle.dump(attrs, f)
        if os.path.exists(old_loc):
            os.remove(old_loc)

    def deserialize(self):
        old_loc = self.cache + ".old"
        if os.path.exists(self.cache):
            file = self.cache
        elif os.path.exists(old_loc):
            file = old_loc
            warnings.warn("Cache file not found. Using old cache file.")
        else:
            raise MissingCacheError("Cache file '{}' not found.".format(self.cache))

        with open(file, "rb") as f:
            self.__dict__.update(safe_load(f))
        fn_path, fn_file = os.path.split(self._function_source)
        sys.path.append(fn_path)
        module = importlib.__import__(os.path.splitext(fn_file)[0])
        experiment_function = getattr(module, self._experiment_fn_name)
        if self._run_in_band:
            self.experiment_function = experiment_function
        else:
            self.experiment_function = multiprocess_wrap(
                experiment_function, serialize=self._serialize
            )

        # Reset initialized state to allow re-initialization
        self._is_resuming = True
        self.initialized = False

        self.logger.info("Test object loaded from file: {}".format(self.cache))
