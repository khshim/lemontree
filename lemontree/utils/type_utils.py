"""
This code includes useful functions for base types(list, tuple, dict).
Most ideas came from google and stackoverflow.
"""

import numpy as np
from collections import OrderedDict


def uint_to_float(data):
    """
    This function returns data in uint8 type to float32 type.
    """
    return np.asarray(data, dtype='float32')


def uint_to_int(data):
    """
    This function returns data in uint8 type to int32 type.
    """
    return np.asarray(data, dtype='int32')


def merge_dicts(dicts):
    """
    This function merge multiple dictionaries into one dictionary.
    Assume each dictionary don't have any same key. (exclusive)
    A dictionary without any key is OK.

    Parameters
    ----------
    dicts: list
        a list of dictionaries.
        dict and Ordereddict are both accepted.

    Returns
    -------
    Ordereddict
        an Ordereddict which contains all key-value pairs in input dictionaries.
    """
    # check asserts
    assert isinstance(dicts, list), '"dicts" should be a list type.'
    assert all(isinstance(dic, dict) for dic in dicts), 'All input should be a dict or Ordereddict.'

    # do
    result = OrderedDict()
    for dic in dicts:
        if len(dic.keys()) != 0:
            for dk in dic.keys():
                assert dk not in result.keys(), 'Every input should not contain same keys each other.'
            result.update(dic)
    return result
