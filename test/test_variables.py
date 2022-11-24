import os
from pathlib import Path

from combinatorial_experiment import variables


def make_dummy_variable():
    kwargs = {"key": "key", "value": range(5), "name": "name"}
    var = variables.Variable(**kwargs)
    return var, kwargs


def test_variable():
    var, kwargs = make_dummy_variable()
    for k, v in kwargs.items():
        assert getattr(var, k) == v
    assert len(var) == len(kwargs["value"])


def test_chain_variable():
    var, kwargs = make_dummy_variable()
    var = variables.ChainVariable([var, var])
    assert len(var) == 2 * len(kwargs["value"])


def test_product_variable():
    var, kwargs = make_dummy_variable()
    var = variables.ProductVariable([var, var])
    assert len(var) == len(kwargs["value"]) ** 2


def test_repeat_variable():
    repeats = 3
    var, kwargs = make_dummy_variable()
    var = variables.RepeatVariable(var, repeats)
    assert len(var) == repeats * len(kwargs["value"])


def test_zipped_variable():
    var, kwargs = make_dummy_variable()
    var = variables.ZippedVariable([var, var])
    assert len(var) == len(kwargs["value"])


def test_deserialization():
    var = variables.deserialize_experiment_config(
        os.path.join(Path(__file__).parent.absolute(), "configs", "variable_config.yml")
    )
    var2 = variables.deserialize_experiment_config(var.serialize())
    assert len(var) == len(var2)
