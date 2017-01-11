# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG
from .layer import BaseLayer


class DropoutLayer(BaseLayer):

    def __init__(self, drop_probability=0.3, rescale=True, name=None):
        super(DropoutLayer, self).__init__(name)
        self.drop_probability = drop_probability
        self.flag = theano.shared(1, 'flag')  # 1: train / -1: evaluation
        self.flag.tag = 'flag'
        self.rng = MRG(np.random.randint(1, 2147462569))
        self.rescale = rescale

    def change_flag(self, new_flag):
        self.flag.set_value(float(new_flag))

    def _compute_output(self, inputs):
        if self.rescale is True:
            coeff = 1 / (1 - self.drop_probability)
        else:
            coeff = 1
        mask = self.rng.binomial(inputs.shape, p=1 - self.drop_probability, dtype=inputs.dtype)
        return T.switch(T.gt(self.flag, 0),
                        inputs * mask * coeff,
                        inputs)

    def _collect_params(self):
        return []

    def _collect_updates(self):
        return OrderedDict()
