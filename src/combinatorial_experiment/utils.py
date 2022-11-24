from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any, Dict, List, Union

import six


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    iteritems = dict.items

    @classmethod
    def from_dict(cls, obj):
        attrdict = cls()
        for k, v in obj.items():
            if isinstance(v, dict):
                attrdict[k] = cls.from_dict(v)
            else:
                if isinstance(v, list):
                    attrdict[k] = cls.check_list(v)
                else:
                    attrdict[k] = v
        else:
            return attrdict

    @classmethod
    def check_list(cls, obj):
        lst = []
        for v in obj:
            if isinstance(v, dict):
                lst.append(cls.from_dict(v))
            else:
                if isinstance(v, list):
                    lst.append(cls.check_list(v))
                else:
                    lst.append(v)
        else:
            return lst


class NestedDict(dict):
    def __getitem__(self, item):
        item = item.split(".")
        for i in range(len(item)):
            self = dict.__getitem__(self, item[i])
        return self

    def __setitem__(self, item, value):
        item = item.split(".")
        for i in range(len(item) - 1):
            try:
                self = dict.__getitem__(self, item[i])
            except KeyError:
                dict.__setitem__(self, item[i], {})
                self = dict.__getitem__(self, item[i])
        dict.__setitem__(self, item[-1], value)

    def flatten(self: Union[Dict[Any, Any], NestedDict]):
        new = {}
        for k, v in self.items():
            if isinstance(v, dict):
                lower = NestedDict.flatten(v)
                for k_l, v_l in lower.items():
                    new[f"{k}.{k_l}"] = v_l
            else:
                if isinstance(v, Iterable):
                    v = str(v)
                new[k] = v
        return new

    def update(self, new):
        for k, v in new.items():
            self[k] = v


def _get_matching_file(
    name: str,
    target: str,
    zero_ext: bool = False,
    ignore: Union[str, List[str]] = [],
    min_digits: int = 0,
    ignore_ext: bool = False,
) -> Union[str, Dict[str, str]]:
    assert target in ("last", "next", "all")
    name = os.path.abspath(name)
    if isinstance(ignore, six.string_types):
        ignore = [ignore]
    ignore = [os.path.abspath(el) for el in ignore]
    dir_, name = os.path.split(name)
    if ignore_ext:
        suffix = ""
        base = name
    else:
        suffix = re.findall("(?<=\\w)\\.\\w*$", name)
        if suffix:
            suffix = suffix[0]
            base = re.split(suffix, name)[0]
        else:
            suffix = ""
            base = name
    numeric = re.findall("_[0-9]+$", name)
    if numeric:
        numeric = numeric[0]
        base = re.split(numeric, base)[0]
    if zero_ext:
        zero_name = base + "_{{:0>{}d}}".format(min_digits).format(0) + suffix
    else:
        zero_name = base + suffix
    matching_files = {}
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    for file in os.listdir(dir_):
        if os.path.join(dir_, file) not in ignore:
            if file == zero_name:
                matching_files[0] = file
            else:
                number = re.findall("(?<={}_)[0-9]+(?={})".format(base, suffix), file)
                if number:
                    matching_files[int(number[0])] = os.path.join(dir_, file)
    if len(matching_files):
        maxnum = max(matching_files.keys())
        if target == "last":
            return os.path.join(dir_, matching_files[maxnum])
        if target == "next":
            return os.path.join(
                dir_,
                base + "_{{:0>{}d}}".format(min_digits).format(maxnum + 1) + suffix,
            )
        return {k: os.path.join(dir_, v) for k, v in matching_files.items()}
    elif target == "last":
        raise FileNotFoundError(
            "No file of format {} exists".format(base + "_xxx" + suffix)
        )
    else:
        if target == "next":
            return os.path.join(dir_, zero_name)
        return {k: os.path.join(dir_, v) for k, v in matching_files.items()}


def get_last(name, zero_ext=False, ignore=[], ignore_ext=False):
    """Get the last (highest-numbered) file of a particular name."""
    return _get_matching_file(
        name, "last", zero_ext=zero_ext, ignore=ignore, ignore_ext=ignore_ext
    )


def safe_save(name, zero_ext=False, verbose=False, min_digits=0, ignore_ext=False):
    name = _get_matching_file(
        name,
        "next",
        zero_ext=zero_ext,
        ignore=[],
        min_digits=min_digits,
        ignore_ext=ignore_ext,
    )
    if verbose:
        print("\n\nSaving in:\n\t{}\n\n".format(name))
    return name


def get_matching(name, zero_ext=False, ignore=[]):
    return _get_matching_file(name, "all", zero_ext=zero_ext, ignore=ignore)


class SizeUnknownError(Exception):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return repr(self.value)
