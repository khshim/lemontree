"""
File: lemontree.graphs.graph.py
Description:
    This file contains abstract base classes for graphs.
"""

import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
from lemontree.misc import merge_dicts


class BaseGraph(object):

    def __init__(self, name=None):
        self.name = name
        self.params = []
        self.layers = []
        self.outputs = []
        self.updates = OrderedDict()

    def set_input(self, input):
        assert len(self.outputs) == 0
        self.outputs.append(input)

    def get_output(self):
        return self.outputs[-1]

    def get_params(self):
        return self.params

    def get_updates(self):
        return self.updates

    def add_layer(self, layer):
        output, params, layer_updates = layer.get_output(self.get_output())
        self.layers.append(layer)
        self.outputs.append(output)
        self.params = self.params + params
        self.updates = merge_dicts(self.updates, layer_updates)
        print('Layer added', type(layer).__name__, layer.name)

    def add_layers(self, layers):
        assert isinstance(layers, list)
        for ly in layers:
            self.add_layer(ly)

    def change_flag(self, flag):
        for layer in self.layers:
            if hasattr(layer, 'flag'):
                layer.change_flag(flag)  # 1: train / -1: evaluation
