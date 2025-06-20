import json
import os
import re
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import pytest

from combinatorial_experiment import CombinatorialExperiment, variables


def experiment_function(config, dry_run=False):
    return {"called": json.dumps(config), "run_ok": True}


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_run_experiment(temp_dir):
    argv = [
        "--base_config",
        os.path.join(Path(__file__).parent.absolute(), "configs", "base_config.yml"),
        "--experiment_config",
        os.path.join(
            Path(__file__).parent.absolute(), "configs", "experiment_config.yml"
        ),
        "--output",
        temp_dir,
    ]
    records = CombinatorialExperiment.run_experiment(experiment_function, argv=argv)
    var = variables.deserialize_experiment_config(
        os.path.join(
            Path(__file__).parent.absolute(), "configs", "experiment_config.yml"
        )
    )
    assert len(records) == len(var)
    assert records["METRIC_run_ok"].all()


def fail_experiment_function(config, dry_run=False):
    raise ValueError("This is an expected experimental failure.")


def test_fail_experiment(temp_dir):
    argv = [
        "--base_config",
        os.path.join(Path(__file__).parent.absolute(), "configs", "base_config.yml"),
        "--experiment_config",
        os.path.join(
            Path(__file__).parent.absolute(), "configs", "experiment_config.yml"
        ),
        "--continue_on_failure",
        "--output",
        temp_dir,
    ]
    records = CombinatorialExperiment.run_experiment(
        fail_experiment_function, argv=argv
    )

    assert len(records) == 0
    experiment_name = os.path.basename(str(temp_dir))
    with sqlite3.connect(os.path.join(temp_dir, "experiment.db")) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(f"SELECT * FROM {experiment_name}")
        rows = cursor.fetchall()
        assert len(rows) == 4
        for row in rows:
            assert row["status"] == "failed"
            assert "This is an expected experimental failure." in row["error"]


def test_archive_on_complete(temp_dir):
    experiment_dir = os.path.join(temp_dir, "experiment")
    argv = [
        "--base_config",
        os.path.join(Path(__file__).parent.absolute(), "configs", "base_config.yml"),
        "--experiment_config",
        os.path.join(
            Path(__file__).parent.absolute(), "configs", "experiment_config.yml"
        ),
        "--archive_on_complete",
        "--output",
        experiment_dir,
    ]
    records = CombinatorialExperiment.run_experiment(experiment_function, argv=argv)
    archive = os.path.join(temp_dir, "experiment.zip")
    if not os.path.exists(archive):
        raise AssertionError("Experiment zip file not found")
    with zipfile.ZipFile(archive, "r") as zip_ref:
        experiments = [
            el
            for el in zip_ref.namelist()
            if re.match(r"experiments/[\w-]+/experiment_config.yml", el)
        ]
        print(experiments)
        if len(experiments) != len(records):
            raise AssertionError("Experiment zip file does not contain all experiments")
