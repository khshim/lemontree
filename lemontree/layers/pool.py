# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from collections import OrderedDict
from .layer import BaseLayer


class Pooling2DLayer(BaseLayer):

    def __init__(self, input_shape, output_shape,
                 kernel_shape=(2, 2), pool_mode='max', stride=(2, 2), name=None):
        super(Pooling2DLayer, self).__init__(name)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_shape = kernel_shape
        assert len(self.input_shape) == 3
        assert len(self.output_shape) == 3
        assert len(self.kernel_shape) == 2
        self.pool_mode = pool_mode
        self.stride = stride
        assert len(stride) == 2

    def _compute_output(self, inputs):
        # assert len(inputs.shape) == 4
        return pool_2d(inputs,
                       ds=self.kernel_shape,
                       ignore_border=True,
                       st=self.stride,
                       mode=self.pool_mode
                       )

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()


class GlobalAveragePooling2DLayer(BaseLayer):

    def __init__(self, input_shape, output_shape,
                 pool_mode='average_exc_pad', name=None):
        super(GlobalAveragePooling2DLayer, self).__init__(name)
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert len(self.input_shape) == 3
        assert self.input_shape[0] == self.output_shape
        self.pool_mode = pool_mode

    def _compute_output(self, inputs):
        result = pool_2d(inputs,
                         ds=self.input_shape[1:],
                         ignore_border=True,
                         st=self.input_shape[1:],
                         mode=self.pool_mode
                         )
        return T.reshape(result, (inputs.shape[0], inputs.shape[1]))

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()
