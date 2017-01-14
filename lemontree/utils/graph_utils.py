"""
This code includes useful functions for graph management.
Convenient functions for Theano supported features.
"""

import theano
import theano.tensor as T
import theano.gof.graph as graph
from collections import OrderedDict


def get_inputs_of_variables(variables):
    """
    This function returns required inputs for the (tensor variable) variable.
    The order of the inputs are toposorted.

    Parameters
    ----------
    variable: list
        a list of (tensor variable) to see.
        usally this is a theano function output list. (loss, accuracy, etc.)

    Returns
    -------
    list
        a list of required inputs to compute the variable.
    """
    # assert
    assert isinstance(variables, list), 'Variables should be a list of tensor variable(s).'
    assert all(isinstance(var, T.TensorVariable) for var in variables), 'All input should be a tensor variable.'

    # do
    variable_inputs = [var for var in graph.inputs(variables) if isinstance(var, T.TensorVariable)]
    variable_inputs = list(OrderedDict.fromkeys(variable_inputs))  # preserve order and make to list
    print('Required inputs are:', variable_inputs)
    return variable_inputs