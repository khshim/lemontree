# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T


class BaseLearninigRateScheduler(object):

    def __init__(self, lr_shared):
        self.lr = lr_shared

    def change_learningrate(self, epoch):
        if epoch == 0:
            print('...Learning rate started at', self.lr.get_value())
        else:
            new_learningrate = self._compute_learningrate(epoch)
            self.lr.set_value(float(new_learningrate))
            print('...Learning rate changed to', new_learningrate)

    def _compute_learningrate(self, epoch):
        raise NotImplementedError('Abstract class method')


class LearningRateDecayScheduler(BaseLearninigRateScheduler):

    def __init__(self, lr_shared, decay_rate):
        super(LearningRateDecayScheduler, self).__init__(lr_shared)
        self.decay_rate = decay_rate

    def _compute_learningrate(self, epoch):
        return self.lr.get_value() / (1.0 + self.decay_rate * epoch)


class LearningRateMultiplyScheduler(BaseLearninigRateScheduler):

    def __init__(self, lr_shared, mult_rate):
        super(LearningRateMultiplyScheduler, self).__init__(lr_shared)
        assert mult_rate < 1
        self.mult_rate = mult_rate

    def _compute_learningrate(self, epoch):
        return self.lr.get_value() * self.mult_rate
