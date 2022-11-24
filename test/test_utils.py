import os
from pathlib import Path
from shutil import rmtree

import pytest

from combinatorial_experiment import utils


def test_attr_dict():
    d = utils.AttrDict({"a": 1, "b": 2})
    assert d.a == 1
    assert d.b == 2


def test_nested_dict():
    d = utils.NestedDict({"a": {"aa": 1, "ab": {"aba": 2, "abb": 3}}, "b": 4})
    d["c.a.a"] = 5
    assert d["a.aa"] == 1
    assert d["a.ab.aba"] == 2
    assert d["a.ab.abb"] == 3
    assert d["b"] == 4
    assert d["c.a.a"] == 5


@pytest.fixture
def directory():
    file_dir = os.path.join(Path(__file__).parent.absolute(), "__files__")
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    Path(os.path.join(file_dir, f"file.dum")).touch()
    for i in range(3):
        Path(os.path.join(file_dir, f"file_{i}.dum")).touch()
    yield file_dir
    rmtree(file_dir)


def test_get_last(directory):
    name = utils.get_last(os.path.join(directory, "file.dum"), zero_ext=True)
    assert name == os.path.join(directory, "file_2.dum")
    name = utils.get_last(os.path.join(directory, "file.dum"), zero_ext=False)
    assert name == os.path.join(directory, "file_2.dum")


def test_get_matching(directory):
    names = utils.get_matching(os.path.join(directory, "file.dum"), zero_ext=True)
    assert len(names) == 3
    name = utils.get_matching(os.path.join(directory, "file.dum"), zero_ext=False)
    assert len(names) == 3


def test_safe_save(directory):
    name = utils.safe_save(os.path.join(directory, "file.dum"), zero_ext=True)
    assert name == os.path.join(directory, "file_3.dum")
    name = utils.safe_save(os.path.join(directory, "file.dum"), zero_ext=False)
    assert name == os.path.join(directory, "file_3.dum")
