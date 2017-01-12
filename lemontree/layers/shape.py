# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class Flatten3DLayer(BaseLayer):

    def __init__(self, input_shape, output_shape, name=None):
        super(Flatten3DLayer, self).__init__(name)
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert len(self.input_shape) == 3

    def get_output(self, inputs):
        # assert len(inputs.shape) == 4
        return T.reshape(inputs, (inputs.shape[0], T.prod(inputs.shape[1:])))

    def get_params(self):
        return []

    def get_updates(self):
        return OrderedDict()
