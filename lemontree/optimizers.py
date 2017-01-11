# Kyuhong Shim 2016
"""
Optimizers compute gradients for shared variables.
Return updates.
"""

import numpy as np
import theano
import theano.tensor as T
import theano.gof.graph as graph
from theano.tensor import TensorConstant, TensorVariable
from collections import OrderedDict


class BaseOptimizer(object):

    def __init__(self, learningrate_init=0.001,
                 clipnorm=None, clipvalue=None, block_tags=None):
        self.internals = []
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        learningrate = np.array(learningrate_init).astype('float32')
        self.lr = theano.shared(learningrate, 'learningrate')
        self.lr.tag = 'learningrate'
        self.block_tags = ['bn_mean', 'bn_std', 'flag']  # Manually
        if block_tags is not None:
            self.block_tags = self.block_tags + block_tags
        # inputs = graph.inputs([loss])
        # self.inputs = [var for var in inputs if isinstance(var, TensorVariable)]

    def _compute_gradients(self, loss, params):
        grads = T.grad(loss, params)
        if self.clipnorm is not None and self.clipnorm > 0:
            norm = T.sqrt(sum([T.sum(T.square(g)) for g in grads]))
            grads = [T.switch(g > self.clipnorm, g * self.clipnorm / norm. g) for g in grads]
        if self.clipvalue is not None and self.clipvalue > 0:
            grads = [T.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def _gradients_to_updates(self, params, grads):
        return NotImplementedError('Abstract class method')

    def change_learningrate(self, new_learningrate):
        self.lr.set_value(float(new_learningrate))

    def get_update(self, loss, params):
        if not isinstance(params, list):
            params = [params]
        # self.loss = loss
        # self.params = params
        unblocked_params = []
        for pp in params:
            if pp.tag not in self.block_tags:
                unblocked_params.append(pp)
        grads = self._compute_gradients(loss, unblocked_params)
        updates = self._gradients_to_updates(unblocked_params, grads)
        return updates

    def get_internals(self):
        return self.internals


class SGD(BaseOptimizer):

    def __init__(self, learningrate_init=0.001,
                 clipnorm=None, clipvalue=None):
        super(SGD, self).__init__(learningrate_init, clipnorm, clipvalue)

    def _gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        for pp, gg in zip(params, grads):
            updates[pp] = pp - self.lr * gg
        return updates


class Momentum(BaseOptimizer):

    def __init__(self, learningrate_init=0.001, momentum_init=0.9,
                 clipnorm=None, clipvalue=None):
        momentum = np.array(momentum_init).astype('float32')
        self.momentum = theano.shared(momentum, 'momentum')
        self.momentum.tag = 'momentum'
        super(Momentum, self).__init__(learningrate_init, clipnorm, clipvalue)

    def change_momentum(self, new_momentum):
        self.momentum.set_value(float(new_momentum))

    def _gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        for pp, gg in zip(params, grads):
            value = pp.get_value(borrow=True)
            self.velocity = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'momentum_velocity_'+pp.name)
            self.internals.append(self.velocity)
            self.velocity.tag = 'optimizer_internal'
            x = self.momentum * self.velocity - self.lr * gg
            updates[self.velocity] = x
            updates[pp] = pp + x
        return updates


class Nesterov(BaseOptimizer):

    def __init__(self, learningrate_init=0.001, momentum_init=0.9,
                 clipnorm=None, clipvalue=None):
        momentum = np.array(momentum_init).astype('float32')
        self.momentum = theano.shared(momentum, 'momentum')
        self.momentum.tag = 'momentum'
        super(Nesterov, self).__init__(learningrate_init, clipnorm, clipvalue)

    def change_momentum(self, new_momentum):
        self.momentum.set_value(float(new_momentum))

    def _gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        for pp, gg in zip(params, grads):
            value = pp.get_value(borrow=True)
            self.velocity = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'nesterov_velocity_'+pp.name)
            self.internals.append(self.velocity)
            self.velocity.tag = 'optimizer_internal'
            x = self.momentum * self.velocity - self.lr * gg
            updates[self.velocity] = x
            updates[pp] = pp + self.momentum * x + self.lr * gg
        return updates


class RMSprop(BaseOptimizer):

    def __init__(self, learningrate_init=0.001, rho_init=0.9,
                 clipnorm=None, clipvalue=None):
        rho = np.array(rho_init).astype('float32')
        self.rho = theano.shared(rho, 'rho')
        self.rho.tag = 'rho'
        super(RMSprop, self).__init__(learningrate_init, clipnorm, clipvalue)

    def _gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        for pp, gg in zip(params, grads):
            value = pp.get_value(borrow=True)
            self.accu = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'rmsprop_accu_'+pp.name)
            self.internals.append(self.accu)
            self.accu.tag = 'optimizer_internal'
            accu_new = self.rho * self.accu + (1 - self.rho) * T.sqr(gg)
            updates[self.accu] = accu_new
            updates[pp] = pp - (self.lr * gg / (T.sqrt(accu_new) + 1e-8))
        return updates


class AdaDelta(BaseOptimizer):

    def __init__(self, learningrate_init=1.0, rho_init=0.9,
                 clipnorm=None, clipvalue=None):
        rho = np.array(rho_init).astype('float32')
        self.rho = theano.shared(rho, 'rho')
        self.rho.tag = 'rho'
        super(AdaDelta, self).__init__(learningrate_init, clipnorm, clipvalue)

    def _gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        for pp, gg in zip(params, grads):
            value = pp.get_value(borrow=True)
            self.accu = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'adadelta_accu_'+pp.name)
            self.delta_accu = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'adadelta_delta_accu_'+pp.name)
            self.internals.append(self.accu)
            self.internals.append(self.delta_accu)
            self.accu.tag = 'optimizer_internal'
            self.delta_accu.tag = 'optimizer_internal'
            accu_new = self.rho * self.accu + (1 - self.rho) * T.sqr(gg)
            updates[self.accu] = accu_new
            ud = gg * (T.sqrt(self.delta_accu) + 1e-8) / (T.sqrt(accu_new) + 1e-8)
            updates[pp] = pp - self.lr * ud
            delta_accu_new = self.rho * self.delta_accu + (1 - self.rho) * T.sqr(ud)
            updates[self.delta_accu] = delta_accu_new
        return updates


class Adam(BaseOptimizer):

    def __init__(self, learningrate_init=0.002, beta1_init=0.9, beta2_init=0.999,
                 clipnorm=None, clipvalue=None):
        beta1 = np.array(beta1_init).astype('float32')
        beta2 = np.array(beta2_init).astype('float32')
        self.beta1 = theano.shared(beta1, 'beta1')
        self.beta2 = theano.shared(beta2, 'beta2')
        self.beta1.tag = 'beta1'
        self.beta2.tag = 'beta2'
        super(Adam, self).__init__(learningrate_init, clipnorm, clipvalue)

    def _gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        self.i = theano.shared(np.array(0).astype('float32'), 'adam_i')
        self.internals.append(self.i)
        self.i.tag = 'optimizer_internal'
        i_t = self.i + 1
        fix1 = 1.0 - T.pow(self.beta1, i_t)
        fix2 = 1.0 - T.pow(self.beta2, i_t)
        lr_t = self.lr * (T.sqrt(fix2) / fix1)
        for pp, gg in zip(params, grads):
            value = pp.get_value(borrow=True)
            self.m = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'adam_m_'+pp.name)
            self.v = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'adam_v_'+pp.name)
            self.internals.append(self.m)
            self.internals.append(self.v)
            self.m.tag = 'optimizer_internal'
            self.v.tag = 'optimizer_internal'
            m_t = (self.beta1 * self.m) + ((1.0 - self.beta1) * gg)
            v_t = (self.beta2 * self.v) + ((1.0 - self.beta2) * T.sqr(gg))
            g_t = m_t / (T.sqrt(v_t) + 1e-8)
            p_t = pp - (lr_t * g_t)
            updates[self.m] = m_t
            updates[self.v] = v_t
            updates[pp] = p_t
        updates[self.i] = i_t
        return updates
