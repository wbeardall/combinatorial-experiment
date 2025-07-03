import copy
import itertools
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

import numpy as np
import six
import yaml

from .utils import SizeUnknownError


class Parameter(object):
    def __init__(self, key, value, i, name=None, record=True, dict={}):
        self._key = key
        self._value = value
        self._name = name
        self._i = i
        self._record = record
        self._dict = dict

    def __str__(self):
        if self._record:
            if self._record == True:  # noqa: E712
                val = self._value
            else:
                val = self._record
        else:
            val = self._i
        return "{}--{}".format(self.name, val)

    __repr__ = __str__

    @property
    def value(self):
        return self._value

    @property
    def name(self):
        if self._name:
            return self._name
        return self.key

    @property
    def key(self):
        return self._key

    @property
    def dict(self):
        d = copy.deepcopy(self._dict)
        d.update({self.key: self.value})
        return d

    @property
    def record(self):
        if self._record:
            if isinstance(self._record, bool):
                return {self.key: self.value}
            else:
                return {self.key: self._record}
        else:
            return {}


class Variable(object):
    key: str
    value: List[Any]
    name: Optional[str]
    record: bool
    dict: Dict[str, Any]

    def __init__(self, key, value, name=None, record=True, dict={}):
        self._key = key
        if not isinstance(value, Iterable):
            value = [value]
        self._value = value
        self._name = name
        self._record = record
        self._dict = dict

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return len(self.value)

    def __next__(self):
        if self._i < len(self):
            try:
                value = self.value[self._i]
            except:
                print(self._key)
                print(self._value)
                raise
            i = self._i
            record = self.record
            self._i += 1
            return Parameter(self.key, value, i, self._name, record, self._dict)
        else:
            raise StopIteration

    def serialize(self):
        return {self.key: self.value}

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    @property
    def name(self):
        if self._name:
            return self._name
        return self.key

    @property
    def record(self):
        if self._record:
            if isinstance(self._record, Iterable) and not isinstance(
                self._record, six.string_types
            ):
                return self._record[self._i]
        return self._record

    def zip(self, *variables):
        return ZippedVariable(self, *variables)

    def product(self, *variables):
        return ProductVariable(self, *variables)

    @property
    def flattened_name(self):
        return [self.name] if len(self.value) > 1 else []

    @property
    def flattened_key(self):
        return [self.key] if len(self.value) > 1 else []


class EKFoldVariable(Variable):
    def __init__(self, fold):
        super().__init__(
            "efold", range(fold), f"ext{fold}fold", record=True, dict={"ekfold": fold}
        )

    def serialize(self):
        return {self.key: self._dict["ekfold"]}


class IKFoldVariable(Variable):
    def __init__(self, fold):
        super().__init__(
            "ifold", range(fold), f"int{fold}fold", record=True, dict={"ikfold": fold}
        )

    def serialize(self):
        return {self.key: self._dict["ikfold"]}


class VariableCollection(object):
    _variables: List[Variable]

    def __init__(self, iterfn, variables):
        self._iterfn = iterfn
        self._variables = variables

    def __iter__(self):
        self._iterator = self._iterfn(*self._variables)
        return self

    def __len__(self):
        raise SizeUnknownError("Length of arbitrary iteration collection unknown.")

    def __next__(self):
        output = next(self._iterator)
        flattened = []
        for el in output:
            if isinstance(el, Iterable):
                flattened.extend(el)
            else:
                flattened.append(el)
        return flattened

    @property
    def flattened_name(self):
        names = [el for var in self._variables for el in var.flattened_name]
        names.sort()
        return names

    @property
    def flattened_key(self):
        keys = [el for var in self._variables for el in var.flattened_key]
        keys.sort()
        return keys

    def serialize(self):
        vs = {}
        for v in self._variables:
            vs.update(v.serialize())
        return {type(self).__name__: vs}

    def to_configs(self, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = []
        for parameters in self:
            config = copy.deepcopy(base_config)
            for p in parameters:
                config.update(p.dict)
            configs.append(config)
        return configs


class ZippedVariable(VariableCollection):
    def __init__(self, variables):
        super(ZippedVariable, self).__init__(zip, variables)

    def __len__(self):
        return min(len(var) for var in self._variables)


class ProductVariable(VariableCollection):
    def __init__(self, variables, experiments=None, experiment_seed=None):
        super(ProductVariable, self).__init__(itertools.product, variables)
        self._experiments = experiments
        self._seed = experiment_seed

    def __iter__(self):
        if self._experiments is not None:
            if self._seed:
                np.random.seed(self._seed)
            it = list(self._iterfn(*self._variables))
            if self._experiments < len(it):
                self._iterator = iter(
                    [
                        it[el]
                        for el in np.random.choice(
                            len(it), size=self._experiments, replace=False
                        )
                    ]
                )
            else:
                self._iterator = iter(it)
            return self
        else:
            return super().__iter__()

    def __len__(self):
        lengths = [len(var) for var in self._variables]
        return (
            np.prod(lengths)
            if not self._experiments
            else min(np.prod(lengths), self._experiments)
        )

    def serialize(self):
        ser = super().serialize()
        ser[type(self).__name__].update(
            {"experiments": self._experiments, "experiment_seed": self._seed}
        )
        return ser


class RepeatVariable(VariableCollection):
    """Wraps a Variable to repeat it, following itertools.repeat"""

    def __init__(self, variables, times=None):
        if times is None:
            raise ValueError("repeat must be specified as an int.")
        self.times = int(times)
        super(RepeatVariable, self).__init__(
            lambda v: itertools.chain.from_iterable(
                itertools.repeat(x, self.times) for x in v
            ),
            variables,
        )

    def __len__(self):
        return self.times * len(self._variables)

    def serialize(self):
        ser = super().serialize()
        ser[type(self).__name__].update({"times": self.times})
        return ser


class ChainVariable(VariableCollection):
    def __init__(self, variables):
        super(ChainVariable, self).__init__(itertools.chain, variables)

    def __len__(self):
        lengths = [len(var) for var in self._variables]
        return np.sum(lengths)


deserializable_vars = dict(
    **dict.fromkeys(["variable", "Variable"], Variable),
    **dict.fromkeys(
        [
            "efold",
            "extfold",
            "ExtFold",
            "ExtFoldVariable",
            "ekfold",
            "extkfold",
            "ExtKFold",
            "ExtKFoldVariables",
            "fold",
            "Fold",
            "FoldVariable",
            "KFold",
            "KFoldVariable",
        ],
        EKFoldVariable,
    ),
    **dict.fromkeys(
        [
            "ifold",
            "intfold",
            "IntFold",
            "IntFoldVariable",
            "ikfold",
            "intkfold",
            "IntKFold",
            "IntKFoldVariable",
        ],
        IKFoldVariable,
    ),
    **dict.fromkeys(
        ["zip", "Zip", "zipped", "Zipped", "ZippedVariable"], ZippedVariable
    ),
    **dict.fromkeys(["product", "Product", "ProductVariable"], ProductVariable),
    **dict.fromkeys(["chain", "Chain", "ChainVariable"], ChainVariable),
)


reserved_kwargs = ["experiments", "experiment_seed", "times"]


def deserialize_experiment_config(config):
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    assert isinstance(config, dict)
    _, kwargs = _deserialize_experiment_layer(config)
    if len(kwargs["variables"]) > 1:
        return ProductVariable(**kwargs)
    elif len(kwargs["variables"]):
        return kwargs["variables"][0]
    else:
        return []


def _deserialize_experiment_layer(config):
    if not isinstance(config, dict):
        if not isinstance(config, Iterable):
            config = [config]
        return config, {}
    variables = []
    kwargs = {k: config.pop(k) for k in reserved_kwargs if k in config}
    for k, v in config.items():
        if k in deserializable_vars.keys():
            a_l, kw_l = _deserialize_experiment_layer(v)
            variable = deserializable_vars[k](*a_l, **kw_l)
        else:
            variable = Variable(k, v)
        variables.append(variable)
    kwargs.update({"variables": variables})
    return [], kwargs


def experiment_length(config):
    variables = deserialize_experiment_config(config)
    print(f"Run contains {len(variables)} experiments.")
