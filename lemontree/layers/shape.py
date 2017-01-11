# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from collections import OrderedDict
from .layer import BaseLayer


class Flatten3DLayer(BaseLayer):

    def __init__(self, input_shape, output_shape, name=None):
        super(Flatten3DLayer, self).__init__(name)
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert len(self.input_shape) == 3

    def _compute_output(self, inputs):
        # assert len(inputs.shape) == 4
        return T.reshape(inputs, (inputs.shape[0], T.prod(inputs.shape[1:])))

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()
