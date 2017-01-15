"""
This code incudes layer which only change shape of tensor.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class Flatten3DLayer(BaseLayer):
    """
    This class implements flatten layer for 3D representation.
    Mostly used for dense layer after convolution layer.
    """
    def __init__(self, input_shape, output_shape, name=None):
        """
        This function initializes the class.
        Input is 4D tensor, output is 2D tensor.

        Parameters
        ----------
        input_shape: tuple
            a tuple of three values, i.e., (input channel, input width, input height).
        output_shape: tuple
            a tuple of single value, i.e., (input channel,) or (input dim,).
        name: string
            a string name of this layer.

        Returns
        -------
        None.
        """
        super(Flatten3DLayer, self).__init__(name)
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 3, '"input_shape" should be a tuple with three values.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple with single value.'
        assert np.prod(input_shape) == output_shape[0], 'Flatten result is 2D tensor of (batch size, input channel * input width * input height).'

        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.reshape(input_, (input_.shape[0], T.prod(input_.shape[1:])))

