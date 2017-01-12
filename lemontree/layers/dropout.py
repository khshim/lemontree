# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG
from lemontree.layers.layer import BaseLayer


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

    def get_output(self, input):
        if self.rescale is True:
            coeff = 1 / (1 - self.drop_probability)
        else:
            coeff = 1
        mask = self.rng.binomial(input.shape, p=1 - self.drop_probability, dtype=input.dtype)
        return T.switch(T.gt(self.flag, 0),
                        input * mask * coeff,
                        input)

    def get_params(self):
        return []

    def get_updates(self):
        return OrderedDict()
