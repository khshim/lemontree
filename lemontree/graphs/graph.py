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
    def __init__(self, name=None):
        """
        This function initializes the class.

        Parameters
        ----------
        name: string
            a string name of this graph.

        Returns
        -------
        None.
        """
        # set members
        self.name = name
        self.params = []  # collection of layer parameters
        self.layers = []  # stack layer class itself
        self.outputs = []  # stack outputs of each layer, the last one is the final output
        self.updates = OrderedDict()  # collection of layer internal updates

    def add_input(self, input_):
        """
        This function starts the symbolic graph by input.
        For most cases, input is defined outside, i.e., T.fmatrix('X').

        Parameters
        ----------
        input: TensorVariable
            a TensorVariable defined by programmer.

        Returns
        -------
        None.
        """
        # check asserts
        assert len(self.outputs) == 0  # should start with clean graph

        # do
        self.outputs.append(input_)

    def get_output(self):
        """
        This function gets current output of stacked layers.
        A convenient function to get output.
        Previous outputs are also stacked inside.

        Parameters
        ----------
        None.

        Returns
        -------
        TensorVariable
            a TensorVariable which is not an input to other layers.
        """
        return self.outputs[-1]

    def get_params(self):
        """
        This function returns parameters of whole layers.

        Parameters
        ----------
        None.

        Returns
        -------
        list
            a list of (shared variable) parameters from layers.
        """
        return self.params

    def get_updates(self):
        """
        This function returns internal updates from layers.

        Parameters
        ----------
        None.

        Returns
        -------
        OrderedDict
            a dictionary of updates, not from optimizers.
        """
        return self.updates

    def add_layer(self, layer):
        """
        This function adds layer to the sequential graph.
        Each layer should append layer itself, output, parameters, and updates.

        Parameters
        ----------
        layer: BaseLayer
            a Layer class that has a single input and a single output.

        Returns
        -------
        None.
        """
        # collect from input layer
        layer_params = layer.get_params()
        layer_updates = layer.get_updates()
        layer_output = layer.get_output(self.get_output())
        self.layers.append(layer)
        self.outputs.append(layer_output)
        self.params = self.params + layer_params
        self.updates = merge_dicts([self.updates, layer_updates])
        print('Layer added', type(layer).__name__, layer.name)

    def add_layers(self, layers):
        """
        This function adds multiple layers in one step.
        Convinient function for removing the need of "add_layer" multiple times.
        Simply iterates over list and add each layer one-by-one.

        Parameters
        ----------
        layers: list
            a list of (BaseLayer) layers to be stacked in order.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(layers, list), '"layers" should be a list of layer classes.'
        
        # iterate
        for ly in layers:
            self.add_layer(ly)

    def change_flag(self, flag):
        """
        This function change flag to alter the mode of layers.
        For some layers, i.e., dropout or batch normalization, there are two modes.
        One is for training, and the other is for inference (validation, test).
        Flag indicates which mode is used.

        Parameyers
        ----------
        flag: int (or float)
            a single scalar value to be a new flag.
            most usage is using flag as condition, flag > 0 or not.

        Returns
        -------
        None.
        """
        # iterate
        for ll in self.layers:
            if hasattr(ll, 'flag'):  # if a layer need mode change, it should have 'flag' as member.
                ll.change_flag(flag)  # 1: train / -1: evaluation
