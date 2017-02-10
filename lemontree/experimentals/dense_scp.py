"""
This code includes simple dense layer.
Dense layer is also well known as fully-connected alyer.
This code implements one-step dense-softmax-categorical crossentropy / Perplexity.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer
from lemontree.layers.activation import Softmax
from lemontree.objectives import CategoricalCrossentropy, WordPerplexity


class DenseLayerSCP(BaseLayer):
    """
    This class implements dense layer connection.
    """
    def __init__(self, input_shape, output_shape, use_bias=True, target_cpu=False):
        """
        This function initializes the class.
        Input is 2D tensor, output is 2D tensor.
        For efficient following batch normalization, use_bias = False.

        Parameters
        ----------
        input_shape: tuple
            a tuple of single value, i.e., (input dim,)
        output_shape: tupe
            a tuple of single value, i.e., (output dim,)
        use_bias: bool, default: True
            a bool value whether we use bias or not.
        target_cpu: bool, default: False
            a bool value whether shared variable will be on cpu or gpu.
        """
        super(DenseLayerSCP, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple with single value.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple with single value.'
        assert isinstance(use_bias, bool), '"use_bias" should be a bool value.'
        assert isinstance(target_cpu, bool), '"target_cpu" should be a bool value.'

        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.use_bias = use_bias
        self.target_cpu = target_cpu

    def set_shared(self):
        """
        This function overrides the parents' one.
        Set shared variables.

        Shared Variables
        ----------------
        W: 2D matrix
            shape is (input dim, output dim).
        b: 1D vector
            shape is (output dim,).
        """
        W = np.zeros((self.input_shape[0], self.output_shape[0])).astype(theano.config.floatX)  # weight matrix
        if self.target_cpu:
            self.W = theano.shared(W, self.name + '_weight', target='cpu')
        else:
            self.W = theano.shared(W, self.name + '_weight')
        self.W.tags = ['weight', self.name]
        b = np.zeros(self.output_shape,).astype(theano.config.floatX)  # bias vector, initialize with 0.
        if self.target_cpu:
            self.b = theano.shared(b, self.name + '_bias', target='cpu')
        else:
            self.b = theano.shared(b, self.name + '_bias')
        self.b.tags = ['bias', self.name]

    def get_output(self, input_, label):
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
            result = T.dot(input_, self.W) + self.b
        else:
            result = T.dot(input_, self.W)

        result = T.nnet.softmax(result)
        cross_entropy = CategoricalCrossentropy(True).get_output(result, label)
        perplexity = WordPerplexity().get_output(result, label)

        return cross_entropy, perplexity

    def get_params(self):
        """
        This function overrides the parents' one.
        Returns interal layer parameters.

        Returns
        -------
        list
            a list of shared variables used.
        """
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]


class TimeDistributedDenseLayerSCP(BaseLayer):
    """
    This class implements time distributed dense layer connection.
    """
    def __init__(self, input_shape, output_shape, use_bias=True, target_cpu=False):
        """
        This function initializes the class.
        Input is 3D tensor, output is 3D tensor.
        For efficient following batch normalization, use_bias = False.

        Parameters
        ----------
        input_shape: tuple
            a tuple of single value, i.e., (input dim,)
        output_shape: tupe
            a tuple of single value, i.e., (output dim,)
        use_bias: bool, default: True
            a bool value whether we use bias or not.
        target_cpu: bool, default: False
            a bool value whether shared variable will be on cpu or gpu.
        """
        super(TimeDistributedDenseLayerSCP, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple with single value.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple with single value.'
        assert isinstance(use_bias, bool), '"use_bias" should be a bool value.'
        assert isinstance(target_cpu, bool), '"target_cpu" should be a bool value.'

        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.use_bias = use_bias
        self.target_cpu = target_cpu

    def set_shared(self):
        """
        This function overrides the parents' one.
        Set shared variables.

        Shared Variables
        ----------------
        W: 2D matrix
            shape is (input dim, output dim).
        b: 1D vector
            shape is (output dim,).
        """
        W = np.zeros((self.input_shape[0], self.output_shape[0])).astype(theano.config.floatX)  # weight matrix
        if self.target_cpu:
            self.W = theano.shared(W, self.name + '_weight', target='cpu')
        else:
            self.W = theano.shared(W, self.name + '_weight')
        self.W.tags = ['weight', self.name]
        b = np.zeros(self.output_shape,).astype(theano.config.floatX)  # bias vector, initialize with 0.
        if self.target_cpu:
            self.b = theano.shared(b, self.name + '_bias', target='cpu')
        else:
            self.b = theano.shared(b, self.name + '_bias')
        self.b.tags = ['bias', self.name]

    def get_output(self, input_, label, mask=None):
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
        input__ = input_.dimshuffle(1,0,2)  # (sequence_length, batch_size, input_dim)
        label_ = label.dimshuffle(1,0)  # (sequence_length, batch_size)
        if mask is not None:
            m_ = mask.dimshuffle(1,0)  # (sequence_length, batch_size)

        def step(input_, label):
            if self.use_bias:
                result = T.dot(input_, self.W) + self.b
            else:
                result = T.dot(input_, self.W)
            result = T.nnet.softmax(result)
            cross_entropy = T.nnet.categorical_crossentropy(T.clip(result, 1e-7, 1.0 - 1e-7), label)  # (batch_size,)
            perplexity = -T.log2(result[T.arange(self.batch_size), label])  # (batch_size,)

            return cross_entropy, perplexity

        output_ = theano.scan(step,
                              sequences=[input__, label_],
                              outputs_info=[None, None])[0]

        if mask is not None:
            cross_entropy = T.sum(output_[0] * m_) / T.sum(m_)  # ()
            perplexity = T.pow(2, T.sum(output_[1] * m_) / T.sum(m_))  # ()
        else:
            cross_entropy = T.mean(output_[0])
            perplexity = T.pow(2, T.sum(output_[1]))
        return cross_entropy, perplexity

    def get_params(self):
        """
        This function overrides the parents' one.
        Returns interal layer parameters.

        Returns
        -------
        list
            a list of shared variables used.
        """
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]
