# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict


class BaseLayer(object):

    def __init__(self, name=None):
        self.name = name

    def _compute_output(self, inputs):
        raise NotImplementedError('Abstract class method')

    def _collect_params(self):  # Trained by optimizers
        raise NotImplementedError('Abstarct class method')

    def _collect_updates(self):  # Additional updates
        raise NotImplementedError('Abstract class method')

    def get_output(self, inputs):
        output = self._compute_output(inputs)
        params = self._collect_params()
        updates = self._collect_updates()
        return output, params, updates


class BaseRecurrentLayer(BaseLayer):

    def __init__(self, gradient_steps=-1, output_return_index=[-1],
                 precompute=False, unroll=False, backward=False, name=None):
        self.gradient_steps = gradient_steps
        self.output_return_index = output_return_index
        assert isinstance(output_return_index, list)
        self.precompute = precompute
        self.unroll = unroll
        self.backward = backward
        self.name = name
        if unroll and gradient_steps <= 0:
            raise ValueError('Network Unroll requires exact gradient step')

    def _compute_output(self, inputs, masks, hidden_init):
        raise NotImplementedError('Abstract class method')

    def _collect_params(self):  # Trained by optimizers
        raise NotImplementedError('Abstarct class method')

    def _collect_updates(self):  # Additional updates
        raise NotImplementedError('Abstract class method')

    def get_output(self, inputs, masks, hidden_init):
        output = self._compute_output(inputs, masks, hidden_init)
        params = self._collect_params()
        updates = self._collect_updates()
        return output, params, updates
