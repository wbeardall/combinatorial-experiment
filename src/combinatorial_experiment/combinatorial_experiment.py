import argparse
import atexit
import copy
import datetime
import gc
import importlib
import inspect
import logging
import os
import random
import re
import string
import sys
import time
import shutil
from typing import Callable
import warnings

import dill as pickle

# NOTE: This is NOT the standard multiprocessing library, which uses pickle;
# it's the one from Pathos, which uses Dill
import multiprocess as mp
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from functools import partial

from .safe_unpickle import safe_load
from .utils import NestedDict, get_last, get_matching, safe_save
from .variables import Parameter, VariableCollection, deserialize_experiment_config

use_style = tuple(int(el) for el in pd.__version__.split(".")) > (1, 3, 0)
allowed_time_symbols = ["T", "t", "Time", "time"]


def multiprocess_wrap(func, serialize: bool = True, dry_run: bool = False):
    def wrapper(q, config):
        try:
            out = func(config, dry_run=dry_run)
            if serialize:
                # For safety, serialize the output
                out = pickle.dumps(out)
            q.put(out)
        except Exception as e:
            out = {"error":e}
            if serialize:
                out = pickle.dumps(out)
            q.put(out)

    return wrapper


# TODO: Consider rewriting to cast variables to list at start of run and store.
# This would allow much easier indexing, iterating, error checking, returning to
# failed runs, etc.
class CombinatorialExperiment(object):
    """Combinatorial Test object.

    This class runs arbitrary tests defined in experiment_function, with the
    variables provided. This class supports caching through Dill, which means
    that it is not secure. Never run a .tst object from an untrustworthy source,
    as the experiment_function can contain arbitrary (and possibly malicious) code.

    """

    _cache_base = "test_iter"
    _cache_ext = ".tst"

    def __init__(
        self,
        experiment_function: Callable = None,
        variables: VariableCollection = None,
        experiment_dir: str = os.getcwd(),
        resume: bool = True,
        base_config: dict = {},
        additional_config: dict = {},
        job_timeout: int = None,
        autoname: bool = False,
        heirarchical_dirs: bool = False,
        repeats: int = 1,
        serialize: bool = False,
        run_in_band: bool = False,
        dry_run: bool = False,
    ):
        # NOTE: cache is deprecated.
        if experiment_function is None or variables is None:
            assert resume == True
        if experiment_function is not None:
            self._function_source = os.path.abspath(
                inspect.getfile(experiment_function)
            )
            self._experiment_fn_name = experiment_function.__name__
            if run_in_band:
                self.experiment_function = partial(experiment_function, dry_run=dry_run)
            else:
                self.experiment_function = multiprocess_wrap(experiment_function, serialize=serialize, dry_run=dry_run)
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
        self._records = None
        self._resume = resume
        self._cache_iter = 0
        if isinstance(base_config, str):
            with open(base_config, "r") as f:
                base_config = yaml.safe_load(f)
        base_config.update(additional_config)
        self.base_config = NestedDict(base_config)
        self.autoname = autoname
        self.job_timeout = job_timeout
        self.heirarchical_dirs = heirarchical_dirs
        self.repeats = repeats
        self.set_directories(experiment_dir, resume)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

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
            experiment_function,
            args.experiment_config,
            experiment_dir=args.output,
            base_config=args.base_config,
            additional_config={el: d[el] for el in add_cmd_args},
            job_timeout=args.job_timeout,
            autoname=args.autoname,
            heirarchical_dirs=args.heirarchical_dirs,
            repeats=args.repeats,
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
        heirarchical_dirs: bool = False,
        repeats: int = 1,
        dry_run: bool = False,
        run_in_band: bool = False,
        serialize: bool = False,
    ):
        variables = deserialize_experiment_config(experiment_config)
        return cls(
            experiment_function,
            variables,
            experiment_dir,
            base_config=base_config,
            additional_config=additional_config,
            job_timeout=job_timeout,
            autoname=autoname,
            heirarchical_dirs=heirarchical_dirs,
            repeats=repeats,
            dry_run=dry_run,
            run_in_band=run_in_band,
            serialize=serialize,
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
        parser.add_argument("--heirarchical_dirs", action="store_true")
        parser.add_argument("--serialize", action="store_true")
        parser.add_argument("--run_in_band", action="store_true")
        return parser

    @classmethod
    def resume(cls, path):
        assert os.path.exists(os.path.join(path, "cache"))
        experiment = CombinatorialExperiment(
            variables=[], experiment_dir=path, resume=True
        )
        experiment.run()

    def add_variable(self, variable):
        self._variables.update(variable)

    def check_run_complete(self, directory=None):
        if directory is None:
            directory = self._experiment_dir
        return os.path.exists(os.path.join(directory, ".run_complete"))

    def mark_run_complete(self):
        with open(os.path.join(self._experiment_dir, ".run_complete"), "w") as f:
            f.write(str(datetime.datetime.now()))

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
                    print(
                        "\n\nStarting new experiment in {}\n\n".format(experiment_dir)
                    )
                else:
                    self._experiment_dir = experiment_dir
                    if os.path.exists(os.path.join(self._experiment_dir, "cache")):
                        print(
                            "\n\nResuming experiment from {}\n\n".format(experiment_dir)
                        )
                        self.auto_deserialize(self._cache_dir)
                    # In case the experiment directory was already created (but empty)
                    else:
                        print(
                            "\n\nStarting new experiment in {}\n\n".format(
                                experiment_dir
                            )
                        )
            except FileNotFoundError:
                experiment_dir = safe_save(
                    experiment_dir, zero_ext=False, ignore_ext=True
                )
                self._experiment_dir = experiment_dir
                print("\n\nStarting new experiment in {}\n\n".format(experiment_dir))
        else:
            experiment_dir = safe_save(experiment_dir, zero_ext=False, ignore_ext=True)
            self._experiment_dir = experiment_dir
            print("\n\nStarting new experiment in {}\n\n".format(experiment_dir))
        if self._dry_run:
            atexit.register(partial(shutil.rmtree, self._experiment_dir))

    def setup(self):
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)
        self.make_gitignore()
        # Set logging to experiment directory
        # Get logger for CombinatorialExperiment
        self.logger = logging.getLogger(str(self.__class__))
        # set the log level to INFO, DEBUG as the default is ERROR
        self.logger.setLevel(logging.INFO)
        filehandler = logging.FileHandler(
            os.path.join(self._experiment_dir, "logfile.log"), "a"
        )
        formatter = logging.Formatter(
            "%(asctime)-15s::%(levelname)s::%(filename)s::%(funcName)s::%(lineno)d::%(message)s"
        )
        filehandler.setFormatter(formatter)
        for hdlr in self.logger.handlers[:]:  # remove the existing file handlers
            if isinstance(hdlr, logging.FileHandler):
                self.logger.removeHandler(hdlr)
        self.logger.addHandler(filehandler)  # set the new handler

    @property
    def _cache_dir(self):
        cache_dir = os.path.join(self._experiment_dir, "cache")
        return cache_dir

    def run(self):
        self.setup()
        self.dump_configs()
        variables = list(self._variables)
        run_length = len(variables)
        for i, parameters in enumerate(tqdm(variables, total=run_length)):
            if i < self._cache_iter:
                if i == self._cache_iter - 1:
                    print(
                        "\n\n Continuing from iteration {}{}.\n\n".format(
                            i + 1, "/{}".format(run_length) if run_length else ""
                        )
                    )
                continue
            self.run_iteration(parameters, i, run_length)
            if self._resume:
                self._cache_iter = i + 1
                cache_loc = os.path.join(
                    self._cache_dir,
                    "{}_{}{}".format(self._cache_base, i, self._cache_ext),
                )
                self.serialize(cache_loc)
                # Remove previous cache to save space
                prev_cache = os.path.join(
                    self._cache_dir,
                    "{}_{}{}".format(self._cache_base, i - 1, self._cache_ext),
                )
                try:
                    os.remove(prev_cache)
                except Exception as e:
                    pass
        self.save_results(False)
        self.mark_run_complete()
        return self.records

    def run_iteration(self, parameters, i, run_length):
        if isinstance(parameters, Parameter):
            parameters = [parameters]
        for j in range(self.repeats):
            config = copy.deepcopy(self.base_config)
            for p in parameters:
                config.update(p.dict)
            # Get record dictionary; replace with parameter value if Boolean
            # True, otherwise leave as-is
            records = copy.deepcopy(self.base_config)
            savedir = os.path.join(self._experiment_dir, "experiments")
            variables_len_gt_one = self._variables.flattened_name
            dir_dict = {}
            for p in parameters:
                records.update(p.record)
                if p.name in variables_len_gt_one:
                    dir_dict[p.name] = str(p)

            # Flatten records to dict and prepend CFG to all
            records = {f"CFG_{k}": v for k, v in NestedDict.flatten(records).items()}

            if self.heirarchical_dirs:
                if self.repeats > 1:
                    # Repeats always at head of directory tree
                    savedir = os.path.join(savedir, "repeat--{}".format(j))
                # K-Fold always at tail of directory tree, External, then Internal
                ef_key = None
                if_key = None
                for k in dir_dict.keys():
                    if re.match("ext[0-9]+fold", k):
                        ef_key = k
                    if re.match("int[0-9]+fold", k):
                        if_key = k
                if ef_key:
                    efold_dir = dir_dict.pop(ef_key)
                else:
                    efold_dir = None
                if if_key:
                    ifold_dir = dir_dict.pop(if_key)
                else:
                    ifold_dir = None
                savedir = os.path.join(
                    savedir, "/".join([dir_dict[k] for k in sorted(dir_dict)])
                )
                for el in [efold_dir, ifold_dir]:
                    if el:
                        savedir = os.path.join(savedir, el)
            else:
                while True:
                    cand = os.path.join(
                        savedir,
                        "".join(
                            random.choice(string.ascii_lowercase) for i in range(15)
                        ),
                    )
                    if not os.path.exists(cand):
                        savedir = cand
                        break

            try:
                os.makedirs(savedir)
            except FileExistsError:
                # Resuming from this directory
                pass

            # Save config
            with open(os.path.join(savedir, "experiment_config.yml"), "w") as f:
                yaml.dump(config, f)

            records["CFG_relpath"] = os.path.relpath(savedir, self._experiment_dir)
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
                    raise RuntimeError("Child process encountered error") from output["error"]
                outer_walltime = time.time() - outer_start

                gc.collect()
                if "metrics" in output.keys():
                    metrics = output["metrics"]

                else:
                    metrics = output

            except Exception as e:
                outer_walltime = time.time() - outer_start
                self.logger.exception(
                    "Fatal error in experiment #{} ({}):\n{}".format(i, savedir, e)
                )
                warnings.warn(str(e))
                if (i == 0):
                    # If the first run fails, we want to raise the error so we can debug
                    raise e
                # Ensure we aren't saving metrics from a partial run
                metrics = {}

            records.update(
                {
                    k
                    if k.startswith("METRIC_") or k.startswith("ENG_")
                    else f"METRIC_{k}": v
                    for k, v in metrics.items()
                }
            )
            records.update({"ENG_walltime": outer_walltime})

            if self._records is None:
                self._records = pd.DataFrame(data=records, index=[0])
            else:
                self._records = pd.concat([self._records,pd.DataFrame(data=records, index=[0])], ignore_index=True)
            self.save_results(True)

    def save_results(self, intermediate=False):
        if intermediate:
            prepend = "intermediate_"
        else:
            prepend = ""
        self._records.to_csv(
            os.path.join(self._experiment_dir, "{}results.csv".format(prepend))
        )

        with open(
            os.path.join(self._experiment_dir, "{}results.md".format(prepend)), "w"
        ) as f:
            if use_style:
                f.write(self._records.style.to_latex())
            else:
                f.write(self._records.to_latex(index=False, float_format="%0.3f"))

    @property
    def records(self):
        return self._records

    def auto_deserialize(self, directory):
        filename = os.path.join(directory, self._cache_base + self._cache_ext)
        try:
            matching_files = get_matching(filename, zero_ext=True)
            descending_idx = np.sort(list(matching_files.keys()))[::-1]
            for idx in descending_idx:
                file = matching_files[idx]
                try:
                    self.deserialize(file)
                    break
                except EOFError:
                    self.logger.warn(
                        "Cache file {} corrupted. Consider deleting.".format(file)
                    )
                    # os.remove(file)
        except FileNotFoundError:
            print("Cache file not found.")

    def serialize(self, filename):
        self_copy = copy.deepcopy(self)
        _ = self_copy.__dict__.pop("experiment_function")
        with open(filename, "wb") as f:
            pickle.dump(self_copy, f)

    def deserialize(self, filename):
        with open(filename, "rb") as f:
            self.__dict__.update(safe_load(f).__dict__)
        fn_path, fn_file = os.path.split(self._function_source)
        sys.path.append(fn_path)
        module = importlib.__import__(os.path.splitext(fn_file)[0])
        experiment_function = getattr(module, self._experiment_fn_name)
        if self._run_in_band:
            self.experiment_function = experiment_function
        else:
            self.experiment_function = multiprocess_wrap(experiment_function, serialize=self._serialize)
        print("Test object loaded from file: {}".format(filename))
