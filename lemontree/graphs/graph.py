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
        self.index = 0
        self.name = name        
        self.layers = []  # stack layer class itself
        self.params = []  # collection of layer parameters
        self.updates = OrderedDict()  # collection of layer internal updates

    def get_outputs(self):
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
        pass

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
        pass

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
        pass

    def add_layer(self, layer, get_from=[-1]):
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
        self.layers.append(layer)

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

        Parameters
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

    def merge_graph(self, target, mode='add'):
        """
        This function merge other graph in this graph.
        Consider parameters, updates, and outputs.
        Two outputs are merged by mode.
        For ResNet, mode = 'add'.

        Parameters
        ----------
        target: graph
            a single graph that will be merged to the graph.
        mode: string, default: 'add'
            a string that how the graph will be merged.
            currently only 'add' is supported.

        Returns
        -------
        None.
        """
        # collect from target graph
        self.params = self.params + target.params
        self.updates = merge_dicts([self.updates, target.updates])
        if mode == 'add':
            self.outputs.append(self.get_output() + target.get_output())
            print(self.name, 'Graph', target.name, 'Merged')
        else:
            raise NotImplementedError('Currently not available mode')
