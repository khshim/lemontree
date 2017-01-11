# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from .layer import BaseLayer


class ReLU(BaseLayer):

    def __init__(self, alpha=0, name=None):
        self.alpha = alpha
        super(ReLU, self).__init__(name)

    def _compute_output(self, inputs):
        return T.nnet.relu(inputs, self.alpha)

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()


class Linear(BaseLayer):

    def __init__(self, name=None):
        super(ReLU, self).__init__(name)

    def _compute_output(self, inputs):
        return inputs

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()


class Tanh(BaseLayer):

    def __init__(self, name = None):
        super(BaseLayer, self).__init__(name)

    def _compute_output(self, inputs):
        return T.tanh(inputs)

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()


class ELU(BaseLayer):

    def __init__(self, alpha=1.0, name=None):
        self.alpha = alpha
        super(ELU, self).__init__(name)

    def _compute_output(self, inputs):
        return T.nnet.elu(inputs, self.alpha)

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()


class Sigmoid(BaseLayer):

    def __init__(self, name=None):
        super(Sigmoid, self).__init__(name)

    def _compute_output(self, inputs):
        return T.nnet.sigmoid(inputs)

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()


class Softmax(BaseLayer):

    def __init__(self, name=None):
        super(Softmax, self).__init__(name)

    def _compute_output(self, inputs):
        return T.nnet.softmax(inputs)

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()


class DistilledSoftmax(BaseLayer):

    def __init__(self, temperature_init=1.0, name=None):
        temperature = np.array(temperature_init).astype('float32')
        self.temperature = theano.shared(temperature, 'temperature')
        self.temperature.tag = ' temperature'
        super(DistilledSoftmax, self).__init__(name)

    def change_temperature(self, new_temperature):
        self.temperature.set_value(float(new_temperature))

    def _compute_output(self, inputs):
        return T.nnet.softmax(inputs / self.temperature)

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()
