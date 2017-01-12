"""
This code includes useful functions for parameters.
Most function use parameter tags.
"""

import numpy as np


def filter_params_by_tags(params, include_tags, exclude_tags=None):
    """
    This function filters parameters and return which satisfies the condition.
    The condition is given by tags.
    If a parameter has multiple tags, "include_tags" and "exclude_tags" works like AND condition.
    In summary, to be in return, a parameter should have at least one tag in "include_tags" and no tag in "exclude_tags".

    Parameters
    ----------
    params: list
        a list of (shared variable) parameters.
    include_tags: list
        a list of (string) tags to include in return.
    exclude_tags: list, default: None
        a list of (string) tags to exclude in return.

    Returns
    -------
    list
        a list of filtered (shared variable) parameters.
    """
    # check asserts
    assert isinstance(params, list), '"params" should be a list type.'
    assert isinstance(include_tags,  list), '"include_tags" should be a list type.'
    if exclude_tags is not None:
        assert isinstance(exclude_tags, list), '"exclude_tags" should be a list type.'

    # check conditions
    return_list = []
    for pp in params:
        pt = pp.tags  # pt is a list
        flag = False  # not included in returns by default
        if any(x in pt for x in include_tags):
            flag = True  # included in returns
        if exclude_tags is not None:
            if any(x in pt for x in exclude_tags):
                flag = False  # not included in returns
        if flag:
            return_list.append(pp)
    return return_list


def print_tags_in_params(params, printing=False):
    """
    This function prints all tags in parameters.

    Parameters
    ----------
    params: list
        a list of (sahred variable) parameters.
    printing: bool
        a bool value whether to print output on command lines.

    Returns
    -------
    set
        a set of sorted (string) tags.
    """
    # check asserts
    assert isinstance(params, list), '"params" should be a list type.'
    assert isinstance(printing, bool), '"printing" should be a bool type.'

    # make tag set
    tag_set = set()
    for pp in params:
        pt = pp.tags  # pt is a list
        for tt in pt:
            tag_set.add(tt)
    if printing:
        print('Tags in parameters:', tag_set)
    return tag_set


def print_params_statistics(params):
    """
    This function prints statistics (mean, std, max, min) for parameters.

    Parameters
    ----------
    params: list
        a list of (shared variable) parameters.

    Returns
    -------
    None.
    """
    # check asserts
    assert isinstance(params, list), '"params" should be a list type.'

    # do
    print('Parameter statistics:')
    for pp in params:
        pvalue = pp.get_value()
        print(pp.name, 'mean:', np.mean(pvalue), 'std:', np.std(pvalue),
              'max:', np.max(pvalue), 'min:', np.min(pvalue), sep='\t')
