"""
This code incudes various type of merge layers.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class MergeAddLayer(BaseLayer):
    """
    This class implements layer add.
    """
    def __init__(self, input_shape,):
        """
        This function initializes the class.
        The shape of two tensor should be equal.

        Parameters
        ----------
        input_shape: tuple
            a tuple of three values, i.e., (input channel, input width, input height).
        output_shape: tuple
            a tuple of single value, i.e., (input channel,) or (input dim,).
        """
        super(MergeAddLayer, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple), '"input_shape" should be a tuple.'
        # set members
        self.input_shape = input_shape

    def get_output(self, input1_, input2_):
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
        return input1_ + input2_

