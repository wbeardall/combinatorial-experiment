import json
import os
import sqlite3
import tempfile

from combinatorial_experiment.experiment import Experiment, ExperimentSet, ExperimentStatus
import pytest

@pytest.fixture
def conn():
    with tempfile.TemporaryDirectory() as temp_dir:
        conn = sqlite3.connect(os.path.join(temp_dir, "test.db"))
        # Ensure that the row factory is set to sqlite3.Row
        # to enable dict-like access to rows
        conn.row_factory = sqlite3.Row
        yield conn
        conn.close()

def test_ensure_table(conn):
    ExperimentSet.ensure_table(conn, "test_experiments")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    assert "test_experiments" in [row[0] for row in cursor]


# Test creating an experiment from config
def test_experiment_from_config():
    config = {"param1": "value1", "param2": 42}
    experiment = Experiment.from_config(config)
    
    assert experiment.config == {"param1": "value1", "param2": 42}
    assert experiment.status == ExperimentStatus.PENDING
    assert experiment.metrics == {}
    assert experiment.error is None
    assert experiment.id is not None

# Test experiment serialization and deserialization
def test_experiment_serialization():
    data = {
        "id": "test-id-123",
        "status": "completed",
        "config": {"param1": "value1"},
        "metrics": {"accuracy": 0.95},
        "metadata": {"meta1": "value1"},
        "error": None
    }
    
    experiment = Experiment.deserialize(data)
    
    assert experiment.id == "test-id-123"
    assert experiment.status == ExperimentStatus.COMPLETED
    assert experiment.config == {"param1": "value1"}
    assert experiment.metrics == {"accuracy": 0.95}
    assert experiment.metadata == {"meta1": "value1"}
    assert experiment.error is None

# Test experiment database operations
def test_experiment_db_operations(conn):
    
    # Create experiment
    experiment = Experiment.from_config(config={"param1": "value1"})
    
    # Create table and insert experiment
    ExperimentSet.ensure_table(conn, "test_experiments")
    experiment.upsert(conn=conn, table_name="test_experiments")
    
    # Retrieve experiment
    retrieved = Experiment.from_conn(conn=conn, table_name="test_experiments", id=experiment.id)
    
    assert retrieved.id == experiment.id
    assert retrieved.status == experiment.status
    assert retrieved.config == experiment.config
    
    # Update status
    experiment.update_status(conn=conn, table_name="test_experiments", status=ExperimentStatus.RUNNING)
    retrieved = Experiment.from_conn(conn=conn, table_name="test_experiments", id=experiment.id)
    assert retrieved.status == ExperimentStatus.RUNNING
    
    # Update metrics
    metrics = {"accuracy": 0.98, "loss": 0.02}
    experiment.update_metrics(conn=conn, table_name="test_experiments", metrics=metrics)
    retrieved = Experiment.from_conn(conn=conn, table_name="test_experiments", id=experiment.id)
    assert retrieved.metrics == metrics
    
    # Update error
    error_msg = "Test error message"
    experiment.update_error(conn=conn, table_name="test_experiments", error=error_msg)
    retrieved = Experiment.from_conn(conn=conn, table_name="test_experiments", id=experiment.id)
    assert retrieved.error == error_msg
    
    # Test combined update
    experiment.update(
        conn=conn, 
        table_name="test_experiments", 
        status="completed", 
        metrics={"final_accuracy": 0.99}, 
        error='new error'
    )
    retrieved = Experiment.from_conn(conn=conn, table_name="test_experiments", id=experiment.id)
    assert retrieved.status == ExperimentStatus.COMPLETED
    assert retrieved.metrics == {"final_accuracy": 0.99}
    assert retrieved.error == 'new error'

# Test ExperimentSet operations
def test_experiment_set_operations(conn):
    
    # Create experiments
    config1 = {"param1": "value1", "param2": 42}
    config2 = {"param1": "value2", "param2": 43}
    config3 = {"param1": "value3", "param2": 44}
    
    exp1 = Experiment.from_config(config1)
    exp2 = Experiment.from_config(config2)
    exp3 = Experiment.from_config(config3)
    
    # Set different statuses
    exp1.status = "completed"
    exp2.status = "running"
    exp3.status = "failed"
    
    # Create experiment set
    experiment_set = ExperimentSet(
        conn=conn, 
        table_name="test_exp_set", 
        experiments=[exp1, exp2, exp3]
    )
    
    # Test push to database
    experiment_set.push_all()
    
    # Create new empty set and pull from database
    new_set = ExperimentSet(conn=conn, table_name="test_exp_set")
    new_set.pull()
    
    # Verify all experiments were retrieved
    assert len(new_set.experiments) == 3
    
    # Test filtering methods
    completed = new_set.get_by_status("completed")
    assert len(completed) == 1
    assert completed[0].id == exp1.id
    
    running = new_set.get_by_status("running")
    assert len(running) == 1
    assert running[0].id == exp2.id
    
    failed = new_set.get_failed()
    assert len(failed) == 1
    assert failed[0].id == exp3.id
    
    incomplete = new_set.get_incomplete()
    assert len(incomplete) == 2
    assert {exp.id for exp in incomplete} == {exp2.id, exp3.id}
    
    all_exps = new_set.get_all()
    assert len(all_exps) == 3
    assert {exp.id for exp in all_exps} == {exp1.id, exp2.id, exp3.id}

# Test ExperimentSet with empty initialization
def test_experiment_set_empty_init(conn):
    # Create empty set
    experiment_set = ExperimentSet(conn=conn, table_name="empty_set")
    assert len(experiment_set.experiments) == 0
    
    # Add experiments programmatically
    config = {"test": True}
    exp = Experiment.from_config(config)
    
    experiment_set.experiments[exp.id] = exp
    experiment_set.push_all()
    
    # Verify experiment was saved
    new_set = ExperimentSet(conn=conn, table_name="empty_set")
    new_set.pull()
    
    assert len(new_set.experiments) == 1
    retrieved_exp = list(new_set.experiments.values())[0]
    assert retrieved_exp.config == {"test": True}

def test_experiment_set_update_experiments(conn):
    experiment_set = ExperimentSet(conn=conn, table_name="test_exp_set")
    orig = {"param1": "value1", "param2": 42}
    second = {"param1": "value2", "param2": {'f1': 1, 'f2': 2}}
    count = experiment_set.update_experiments(configs=[orig], repeats=1)
    assert count == 1
    assert list(experiment_set.experiments.values())[0].config == orig

    count = experiment_set.update_experiments(configs=[orig, second], repeats=1)
    assert count == 1
    assert list(experiment_set.experiments.values())[0].config == orig
    assert list(experiment_set.experiments.values())[1].config == second

    count = experiment_set.update_experiments(configs=[orig, second], repeats=2)
    assert count == 2
    assert len(experiment_set.experiments) == 4
