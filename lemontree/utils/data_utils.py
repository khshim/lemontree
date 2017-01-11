"""
This code includes useful functions for data managements.
Functions are not data dependent.
"""

import numpy as np


def split_data(input, rule=0.9):
    """
    This function split inputs to two groups.
    The order of input data (or label) is unchanged.
    Split starts from the start, i.e., 0.9 / 0.1, not 0.1 / 0.9.

    Parameters
    ----------
    input: n-dim nparray or list
        an array or a list of anything.
    rule: float, int
        rule to split.
        if 0 < rule < 1, split by ratio. if rule > 1, split by numbers.

    Returns
    -------
    tuple
        a tuple of two splitted parts.
    """
    # check asserts
    assert 0 < rule, 'Rule should be a positive value.'
    num_input = len(input)  # also works for ndarray.

    # divide
    if rule < 1:
        num_first = np.floor(num_input * rule)
        num_second = num_input - num_first
        print('Splitted to:', num_first, 'and', num_second)
        first_input = input[:num_first]
        second_input = input[num_first:]
    elif rule >= 1:
        assert rule < num_input, 'Rule cannot be bigger than the number of inputs.'
        assert isinstance(rule, int), 'If rule > 1, rule should be an integer.'
        print('Splitted to:', rule, 'and', num_input - rule)
        first_input = input[:rule]
        second_input = input[rule:]
    else:
        raise ValueError('Invalid input "rule".')
    return first_input, second_input
