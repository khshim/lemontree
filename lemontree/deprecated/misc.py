# Kyuhong Shim 2016

"""
Misc functions
"""

import csv
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import theano.gof.graph as graph
from theano.tensor import TensorConstant, TensorVariable
from collections import OrderedDict

# moved
def split_data(data, label, rule=0.9):
    ndata = data.shape[0]
    if rule < 1:
        nfirst = np.floor(ndata * rule)
        nsecond = ndata - nfirst
        print('Splited to:', nfirst, 'and', nsecond)
        return data[:nfirst], label[:nfirst], data[nfirst:], label[nfirst:]
    elif rule >= 1:
        assert rule < ndata
        print('Splited to:', rule, 'and', ndata-rule)
        return data[:rule], label[:rule], data[rule:], label[rule:]
    else:
        raise ValueError('Not supported value')

# moved
def merge_dicts(x, y):
    result = OrderedDict()
    if len(x.keys()) != 0:
        result.update(x)
    if len(y.keys()) != 0:
        result.update(y)
    return result

# moved
def get_inputs(loss):
    loss_inputs = [var for var in graph.inputs([loss]) if isinstance(var, TensorVariable)]
    loss_inputs = list(OrderedDict.fromkeys(loss_inputs))  # preserve order
    print('Inputs are : ', loss_inputs)
    return loss_inputs


# moved
def filter_params(params, tag):
    return [pp for pp in params if pp.tag == tag]

# moved
def print_tags_params(params):
    tagset = set()
    for pp in params:
        tagset.add(pp.tag)
    print('Tags are : ', tagset)
