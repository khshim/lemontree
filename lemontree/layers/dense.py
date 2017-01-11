# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from .layer import BaseLayer


class DenseLayer(BaseLayer):
    def __init__(self, input_shape, output_shape, use_bias=True, name=None):
        super(DenseLayer, self).__init__(name)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.use_bias = use_bias

        W = np.zeros((input_shape, output_shape)).astype(theano.config.floatX)
        self.W = theano.shared(W, self.name + '_weight')
        self.W.tag = 'weight'
        b = np.zeros((output_shape,)).astype(theano.config.floatX)
        self.b = theano.shared(b, self.name + '_bias')
        self.b.tag = 'bias'

    def _compute_output(self, inputs):
        if self.use_bias:
            return T.dot(inputs, self.W) + self.b
        else:
            return T.dot(inputs, self.W)

    def _collect_params(self):
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]

    def _collect_updates(self):
        return OrderedDict()
