"""
This code includes simple dense layer.
Dense layer is also well known as fully-connected alyer.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class DenseLayer(BaseLayer):
    """
    This class implements dense layer connection.
    """
    def __init__(self, input_shape, output_shape, use_bias=True, name=None):
        """
        This function initializes the class.
        Input is 2D tensor, output is 2D tensor.
        For efficient followed batch normalization, use_bias = False.

        Parameters
        ----------
        input_shape: tuple
            a tuple of single value, i.e., (input dim,)
        output_shape: tupe
            a tuple of single value, i.e., (output dim,)
        use_bias: bool
            a bool value whether we use bias or not.
        name: string
            a string name of this layer.

        Returns
        -------
        None.
        """
        super(DenseLayer, self).__init__(name)
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple with single value.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple with single value.'
        assert isinstance(use_bias, bool), '"use_bias" should be a bool value.'

        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.use_bias = use_bias

        # create shared variables
        """
        Shared Variables
        ----------------
        W: 2D matrix
            shape is (input dim, output dim).
        b: 1D vector
            shape is (output dim,).
        """
        W = np.zeros((input_shape[0], output_shape[0])).astype(theano.config.floatX)  # weight matrix
        self.W = theano.shared(W, self.name + '_weight')
        self.W.tags = ['weight', self.name]
        b = np.zeros(output_shape).astype(theano.config.floatX)  # bias vector, initialize with 0.
        self.b = theano.shared(b, self.name + '_bias')
        self.b.tags = ['bias', self.name]

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.
        
        Math Expression
        -------------------
        Y = dot(X, W) + b
        Y = dot(X, W)
            bias is automatically broadcasted. (supported theano feature)

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        if self.use_bias:
            return T.dot(input_, self.W) + self.b
        else:
            return T.dot(input_, self.W)

    def get_params(self):
        """
        This function overrides the parents' one.
        Returns interal layer parameters.

        Parameters
        ----------
        None.

        Returns
        -------
        list
            an empty list for consistency.
        """
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]

    def get_updates(self):
        """
        This function overrides the parents' one.
        Returns internal updates.

        Parameters
        ----------
        None.

        Returns
        -------
        OrderedDict
            an empty dictionary for consistency.
        """
        return OrderedDict()
