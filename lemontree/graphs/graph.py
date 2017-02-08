"""
This code contains simple stacking graphs.
In Keras, this type of graph is called 'Sequential'.
"""

import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
from lemontree.utils.type_utils import merge_dicts


class SimpleGraph(object):
    """
    This class defines base class for symbolic graph which merge layers.
    Although name includes 'simple', this is not abstract class and can be used generally.
    In this sequential grpah, input and output are only one each.
    """
    def __init__(self, name='graph', batch_size=-1):
        """
        This function initializes the class.

        Parameters
        ----------
        batch_size: int, default: -1
            an integer that is batch size (data per batch)
        name: string
            a string name of this graph.

        Returns
        -------
        None.
        """
        # set members
        self.index = 0
        self.batch_size = batch_size
        self.name = name
        self.layers = []  # stack layer class itself
        self.connections = {}  # connections of layers {index: ([ins], [outs])}
        self.params = []  # collection of layer parameters
        self.updates = OrderedDict()  # collection of layer internal updates

    def add_layer(self, layer, get_from=[-1]):
        """
        This function adds layer to the sequential graph.
        Each layer should append layer itself, output, parameters, and updates.

        Parameters
        ----------
        layer: BaseLayer
            a Layer class that has a single input and a single output.
        get_from: list, default: [-1]
            a list of connections that layer gets output from.
        """
        # check asserts
        assert isinstance(get_from, list), '"get_from" should be a list of required layers.'
        if len(get_from) != 0:
            for gf in get_from:
                assert gf < self.index, '"all layer in "get_from" should be previously added layers.'

        # collect from input layer
        self.layers.append(layer)
        layer.set_name(self.name + '_' + str(self.index) + '_' + layer.__class__.__name__)
        layer.set_shared()
        layer.set_batch_size(self.batch_size)
        print('... Layer added', self.name + '_' + str(self.index) + '_' + layer.__class__.__name__)

        # set connections
        get_from_new = []
        for gf in get_from:
            if gf < 0:
                get_from_new.append(self.index + gf)
            else:
                get_from_new.append(gf)
        self.connections[self.index] = (get_from_new, [])
        for gf in get_from_new:
            self.connections[gf] = (self.connections[gf][0], self.connections[gf][1] + [self.index])
        self.index += 1

    def get_output(self, inputs_, layer_out, layer_in=0):
        """
        This function create Tensorvariable using graph structure.

        Parameters
        ----------
        input_: TensorVariable
            a TensorVariable of required shape, for initial input.
        layer_out: integer
            an integer that which layer output will be returned.
        layer_in: integer, default: 0
            an integer that which layer will get the input.

        Returns
        -------
        TensorVariable
        """
        # check asserts
        assert isinstance(layer_out, int), '"layer_out" should be a non-negative integer.'
        assert isinstance(layer_in, int), '"layer_in" should be a non-negative integer.'
        if layer_in < 0:
            layer_in = len(self.layers) + layer_in
        if layer_out < 0:
            layer_out = len(self.layers) + layer_out
        assert layer_in <= layer_out, '"layer_out" should be bigger or equal to "layer_in".'

        # recursively find required layers
                
        def find_required_layers(index):
            required = []
            for cc in self.connections[index][0]:
                required += find_required_layers(cc)
            required = required + self.connections[index][0]
            # print(required)
            return required

        required = find_required_layers(layer_out) + [layer_out]
        required = list(sorted(set(required)))
        required = [item for item in required if item >= layer_in]
        print('... Required Layers', required)
        assert layer_in in required, 'Somewhere disconnected.'

        # check required
        for cc in required:
            if cc != layer_in:
                assert all(self.connections[cc][0][id] in required for id in range(len(self.connections[cc][0]))), 'Somewhere disconnected.'

        # compute outputs
        intermediate = {}
        for cc in required:
            if cc == layer_in:
                intermediate[cc] = self.layers[cc].get_output(*inputs_)
            else:
                cc_input = ()
                for dd in self.connections[cc][0]:
                    cc_input += (intermediate[dd],)
                intermediate[cc] = self.layers[cc].get_output(*cc_input)
            
        return intermediate[layer_out], required
    
    def get_params(self, layers=None):
        """
        This function returns parameters of whole layers.

        Parameters
        ----------
        layers: list, default: None
            a list of layers that parameters should be returned.

        Returns
        -------
        list
            a list of (shared variable) parameters from layers.
        """
        # check asserts
        if layers is not None:
            assert isinstance(layers, list), '"layers" should be None or list.'

        params = []
        if layers is None:
            for ll in self.layers:
                params += ll.get_params()
        else:
            for ind, ll in enumerate(self.layers):
                if ind in layers:
                    params += ll.get_params()

        return params

    def get_updates(self, layers=None):
        """
        This function returns internal updates from layers.

        Parameters
        ----------
        layers: list, default: None
            a list of layers that parameters should be returned.

        Returns
        -------
        OrderedDict
            a dictionary of updates, not from optimizers.
        """
        # check asserts
        if layers is not None:
            assert isinstance(layers, list), '"layers" should be None or list.'

        updates = OrderedDict()
        if layers is None:
            for ll in self.layers:
                updates = merge_dicts([updates, ll.get_updates()])
        else:
            for ind, ll in enumerate(self.layers):
                if ind in layers:
                    updates = merge_dicts([updates, ll.get_updates()])

        return updates

    def change_flag(self, flag):
        """
        This function change flag to alter the mode of layers.
        For some layers, i.e., dropout or batch normalization, there are two modes.
        One is for training, and the other is for inference (validation, test).
        Flag indicates which mode is used.

        Parameters
        ----------
        flag: int (or float)
            a single scalar value to be a new flag.
            most usage is using flag as condition, flag > 0 or not.
        """
        # iterate
        for ll in self.layers:
            if hasattr(ll, 'flag'):  # if a layer need mode change, it should have 'flag' as member.
                ll.change_flag(flag)  # 1: train / -1: evaluation
