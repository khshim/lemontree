# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from .layer import BaseLayer


class Convolution2DLayer(BaseLayer):

    def __init__(self, input_shape, output_shape, kernel_shape,
                 border_mode='valid', stride=(1, 1), use_bias=True, name=None):
        super(Convolution2DLayer, self).__init__(name)
        self.input_shape = input_shape  # (number of maps, width of map, height of map)
        self.output_shape = output_shape  # (number of maps, width of map, height of map)
        self.kernel_shape = kernel_shape  # (width of kernel, height of kernel)
        assert len(self.input_shape) == 3
        assert len(self.output_shape) == 3
        assert len(self.kernel_shape) == 2
        self.border_mode = border_mode
        self.stride = stride
        assert len(self.stride) == 2
        self.use_bias = use_bias

        W = np.zeros((self.output_shape[0], self.input_shape[0], self.kernel_shape[0], self.kernel_shape[1])).astype(theano.config.floatX)
        self.W = theano.shared(W, self.name + '_weight')
        self.W.tag = 'weight'
        b = np.zeros((self.output_shape[0])).astype(theano.config.floatX)
        self.b = theano.shared(b, self.name + '_bias')
        self.b.tag = 'bias'

    def _compute_output(self, inputs):
        # assert len(inputs.shape) == 4
        result = T.nnet.conv2d(inputs,
                               filters=self.W,
                               input_shape=(None,) + self.input_shape,
                               filter_shape=(self.output_shape[0], self.input_shape[0], self.kernel_shape[0], self.kernel_shape[1]),
                               border_mode=self.border_mode,
                               subsample=self.stride
                               )
        if self.use_bias:
            return result + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            return result

    def _collect_params(self):
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]

    def _collect_updates(self):
        return OrderedDict()


class Padding2DLayer(BaseLayer):

    def __init__(self, input_shape, output_shape, padding=(1, 1, 1, 1), name=None):
        super(Padding2DLayer, self).__init__(name)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.padding = padding  # (up, down, left, right)
        assert len(self.input_shape) == 3
        assert len(self.output_shape) == 3
        assert len(self.padding) == 4

    def _compute_output(self, inputs):
        # assert len(inputs.shape) == 4
        shape = (inputs.shape[0],) + (self.input_shape[0], self.input_shape[1] + self.padding[0] + self.padding[1], self.input_shape[2] + self.padding[2] + self.padding[3])
        result = T.zeros(shape, dtype=theano.config.floatX)
        indices = (slice(None),
                   slice(None),
                   slice(self.padding[0], self.input_shape[1] + self.padding[0]),
                   slice(self.padding[2], self.input_shape[2] + self.padding[2])
                   )
        return T.set_subtensor(result[indices], inputs)

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()
