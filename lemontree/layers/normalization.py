# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from .layer import BaseLayer


class BatchNormalization1DLayer(BaseLayer):

    def __init__(self, input_shape, momentum=0.99, name=None):
        super(BatchNormalization1DLayer, self).__init__(name)
        self.input_shape = input_shape
        self.momentum = momentum
        self.updates = OrderedDict()
        self.flag = theano.shared(1, 'flag')  # 1: train / -1: evaluation
        self.flag.tag = 'flag'
        bn_mean = np.zeros((input_shape,)).astype(theano.config.floatX)
        self.bn_mean = theano.shared(bn_mean, self.name + '_bn_mean')
        self.bn_mean.tag = 'bn_mean'
        bn_std = np.ones((input_shape,)).astype(theano.config.floatX)
        self.bn_std = theano.shared(bn_std, self.name + '_bn_std')
        self.bn_std.tag = 'bn_std'
        gamma = np.ones((input_shape,)).astype(theano.config.floatX)
        self.gamma = theano.shared(gamma, self.name + '_gamma')
        self.gamma.tag = 'gamma'
        beta = np.zeros((input_shape,)).astype(theano.config.floatX)
        self.beta = theano.shared(beta, self.name + '_beta')
        self.beta.tag = 'beta'

    def change_flag(self, new_flag):
        self.flag.set_value(float(new_flag))

    def _compute_output(self, inputs):
        batch_mean = T.mean(inputs, axis=0)
        batch_std = T.std(inputs, axis=0)
        self.updates[self.bn_mean] = self.bn_mean * self.momentum + batch_mean * (1 - self.momentum)
        self.updates[self.bn_std] = self.bn_std * self.momentum + batch_std * (1 - self.momentum)
        return T.switch(T.gt(self.flag, 0),
                        T.nnet.batch_normalization(inputs, self.gamma, self.beta, batch_mean, batch_std),
                        T.nnet.batch_normalization(inputs, self.gamma, self.beta, self.bn_mean, self.bn_std))

    def _collect_params(self):
        return [self.gamma, self.beta, self.bn_mean, self.bn_std]

    def _collect_updates(self):
        return self.updates


class BatchNormalization2DLayer(BaseLayer):

    def __init__(self, input_shape, momentum=0.99, name=None):
        super(BatchNormalization2DLayer, self).__init__(name)
        self.input_shape = input_shape  # (number of maps, width of map, height of map)
        assert len(self.input_shape) == 3
        self.momentum = momentum
        self.updates = OrderedDict()
        self.flag = theano.shared(1, 'flag')  # 1: train / -1: evaluation
        self.flag.tag = 'flag'
        bn_mean = np.zeros((input_shape[0],)).astype(theano.config.floatX)
        self.bn_mean = theano.shared(bn_mean, self.name + '_bn_mean')
        self.bn_mean.tag = 'bn_mean'
        bn_std = np.ones((input_shape[0],)).astype(theano.config.floatX)
        self.bn_std = theano.shared(bn_std, self.name + '_bn_std')
        self.bn_std.tag = 'bn_std'
        gamma = np.ones((input_shape[0],)).astype(theano.config.floatX)
        self.gamma = theano.shared(gamma, self.name + '_gamma')
        self.gamma.tag = 'gamma'
        beta = np.zeros((input_shape[0],)).astype(theano.config.floatX)
        self.beta = theano.shared(beta, self.name + '_beta')
        self.beta.tag = 'beta'

    def change_flag(self, new_flag):
        self.flag.set_value(float(new_flag))

    def _compute_output(self, inputs):
        batch_mean = T.mean(inputs, axis=[0, 2, 3])
        batch_std = T.std(inputs, axis=[0, 2, 3])
        self.updates[self.bn_mean] = self.bn_mean * self.momentum + batch_mean * (1 - self.momentum)
        self.updates[self.bn_std] = self.bn_std * self.momentum + batch_std * (1 - self.momentum)
        return T.switch(T.gt(self.flag, 0),
                        T.nnet.batch_normalization(inputs, self.gamma.dimshuffle('x', 0, 'x', 'x'), self.beta.dimshuffle('x', 0, 'x', 'x'),
                        batch_mean.dimshuffle('x', 0, 'x', 'x'), batch_std.dimshuffle('x', 0, 'x', 'x')),
                        T.nnet.batch_normalization(inputs, self.gamma.dimshuffle('x', 0, 'x', 'x'), self.beta.dimshuffle('x', 0, 'x', 'x'),
                        self.bn_mean.dimshuffle('x', 0, 'x', 'x'), self.bn_std.dimshuffle('x', 0, 'x', 'x')))

    def _collect_params(self):
        return [self.gamma, self.beta, self.bn_mean, self.bn_std]

    def _collect_updates(self):
        return self.updates


class LayerNormalization1DLayer(BaseLayer):

    def __init__(self, input_shape, momentum=0.99, name=None):
        super(LayerNormalization1DLayer, self).__init__(name)
        self.input_shape = input_shape
        self.momentum = momentum
        gamma = np.ones((input_shape,)).astype(theano.config.floatX)
        self.gamma = theano.shared(gamma, self.name + '_gamma')
        self.gamma.tag = 'gamma'
        beta = np.zeros((input_shape,)).astype(theano.config.floatX)
        self.beta = theano.shared(beta, self.name + '_beta')
        self.beta.tag = 'beta'

    def _compute_output(self, inputs):
        dim_mean = T.mean(inputs, axis=1)
        dim_std = T.std(inputs, axis=1)
        return self.gamma * (inputs - dim_mean.dimshuffle(0, 'x')) / (dim_std.dimshuffle(0, 'x') + 1e-8) + self.beta

    def _collect_params(self):
        return [self.gamma, self.beta]

    def _collect_updates(self):
        return OrderedDict()


class LayerNormalization2DLayer(BaseLayer):

    def __init__(self, input_shape, momentum=0.99, name=None):
        super(LayerNormalization2DLayer, self).__init__(name)
        self.input_shape = input_shape  # (number of maps, width of map, height of map)
        assert len(self.input_shape) == 3
        self.momentum = momentum
        gamma = np.ones((input_shape[0],)).astype(theano.config.floatX)
        self.gamma = theano.shared(gamma, self.name + '_gamma')
        self.gamma.tag = 'gamma'
        beta = np.zeros((input_shape[0],)).astype(theano.config.floatX)
        self.beta = theano.shared(beta, self.name + '_beta')
        self.beta.tag = 'beta'

    def _compute_output(self, inputs):
        dim_mean = T.mean(inputs, axis=[1, 2, 3])
        dim_std = T.std(inputs, axis=[1, 2, 3])
        return self.gamma.dimshuffle('x', 0, 'x', 'x') * (inputs - dim_mean.dimshuffle(0, 'x', 'x', 'x')) / (dim_std.dimshuffle(0, 'x', 'x', 'x') + 1e-8) + self.beta.dimshuffle('x', 0, 'x', 'x')

    def _collect_params(self):
        return [self.gamma, self.beta]

    def _collect_updates(self):
        return OrderedDict()
