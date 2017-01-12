"""
This code includes various optimizer algorithms.
Optimizers compute gradients for shared variables, based on gradient descent algorithm.
Planning not to use T.grad anymore, instead we are using theano.gradient.grad.
If you don't have the function, please update your Theano version.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.utils.param_utils import filter_params_by_tags, print_tags_in_params


class BaseOptimizer(object):
    """
    This class defines abstract base class for optimizers.
    Most of the function is well-defined and not overrided.
    Each optimizer can only get one scalar loss.
    """
    def __init__(self, lr_init=0.001,
                 clipnorm=None, clipvalue=None, exclude_tags=[]):
        """
        This function initializes the class.

        Parameters
        ----------
        lr_init: float, default: 0.001
            a float value for initial learning rate.
        clipnorm: float, default: None
            a float value which is upper bound of rms norm of gradients.
            norm means root-mean-square of all gradient elements.
            if None, we don't clip gradients.
        clipvalue: float, default: None
            a float value which is upper bound of each gradient elements.
            if None, we don't clip gradients.
        exclude_tags: list, default: []
            a list of (shared variable) parameter tags.
            if a parameter has the tag, the optimizer ignore the parameter.

        Returns
        -------
        None.
        """
        # check asserts
        assert lr_init > 0, '"lr_init" should be positive float value.'
        assert isinstance(exclude_tags, list), '"exclude_tags" should be a list of string tags.'
        if clipnorm is not None:
            assert clipnorm > 0, '"clipnorm" should be a positive float value.'
        if clipvalue is not None:
            assert clipvalue > 0, '"clipvalue" should be a positive float value.'

        # set members
        self.params = []
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.exclude_tags = ['bn_mean', 'bn_std', 'flag'] + exclude_tags

        # learning rate to shared variable
        lr = np.array(lr_init).astype(theano.config.floatX)
        self.lr = theano.shared(lr, 'lr')
        self.lr.tags = ['lr']

    def change_lr(self, new_lr):
        """
        This function changes learning rate to new value.
        Since learning rate is shared variable, we should use set_value().

        Parameters
        ----------
        new_lr: float
            a float value to be new learning rate.

        Returns
        -------
        None.
        """
        # do
        self.lr.set_value(float(new_lr))

    def compute_gradients(self, loss, params):
        """
        This function computes gradients for each parameters based on the loss.
        Output is exactly same size as parameters.

        Parameters
        ----------
        loss: TensorVariable
            a TensorVaraible which is scalar.
        params: list
            a list of (shared variable) parameters.

        Returns
        -------
        list
            a list of (tensor variable) parameter gradients.
            the order of gradients are same as parameters.
        """
        # check asserts
        assert isinstance(params, list), '"params" should be a list type.'
        
        # compute gradients
        grads = T.grad(loss, params)
        if self.clipnorm is not None:
            grad_l2norm = T.sqrt(sum([T.sum(T.square(gg)) for gg in grads]))
            grads = [T.switch(grad_l2norm > self.clipnorm, g * self.clipnorm / (grad_l2norm + 1e-8), g) for g in grads]
        if self.clipvalue is not None:
            grads = [T.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def gradients_to_updates(self, params, grads):
        """
        This function makes update rule from parameters and gradients.
        Each update is exactly same size as corresponding parameter.

        Parameters
        ----------
        params: list
            a list of (shared variable) parameters.
        grads: list
            a list of TensorVariables.
            grads are computed from "compute_gradients", with or without clippling.

        Returns
        -------
        OrderedDict
            an OrderedDict of updates for each parameter.
        """
        return NotImplementedError('Abstract class method')

    def get_updates(self, loss, params):
        """
        This function returns computed update rule.
        Actual computation is done is "compute_gradients" and "gradients_to_updates".

        Parameters
        ----------
        loss: TensorVariable
            a TensorVaraible which is scalar.
        params: list
            a list of (shared varaible) parameters.

        Returns
        -------
        OrderedDict
            an OrderedDict of updates for each parameter.
        """
        # check asserts 
        assert isinstance(params, list), '"params" should be a list type.'

        # do
        include_tags = print_tags_in_params(params)  # include all tags to include all parameters
        unblocked_params = filter_params_by_tags(params, include_tags, self.exclude_tags)
        grads = self.compute_gradients(loss, unblocked_params)
        updates = self.gradients_to_updates(unblocked_params, grads)
        return updates

    def get_params(self):
        return self.params


class SGD(BaseOptimizer):

    def __init__(self, lr_init=0.001,
                 clipnorm=None, clipvalue=None):
        super(SGD, self).__init__(lr_init, clipnorm, clipvalue)

    def gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        for pp, gg in zip(params, grads):
            updates[pp] = pp - self.lr * gg
        return updates


class Momentum(BaseOptimizer):

    def __init__(self, lr_init=0.001, momentum_init=0.9,
                 clipnorm=None, clipvalue=None):
        momentum = np.array(momentum_init).astype('float32')
        self.momentum = theano.shared(momentum, 'momentum')
        self.momentum.tags = ['momentum']
        super(Momentum, self).__init__(lr_init, clipnorm, clipvalue)

    def change_momentum(self, new_momentum):
        self.momentum.set_value(float(new_momentum))

    def gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        for pp, gg in zip(params, grads):
            value = pp.get_value(borrow=True)
            self.velocity = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'momentum_velocity_'+pp.name)
            self.params.append(self.velocity)
            self.velocity.tags = ['optimizer_param']
            x = self.momentum * self.velocity - self.lr * gg
            updates[self.velocity] = x
            updates[pp] = pp + x
        return updates


class Nesterov(BaseOptimizer):

    def __init__(self, lr_init=0.001, momentum_init=0.9,
                 clipnorm=None, clipvalue=None):
        momentum = np.array(momentum_init).astype('float32')
        self.momentum = theano.shared(momentum, 'momentum')
        self.momentum.tags = ['momentum']
        super(Nesterov, self).__init__(lr_init, clipnorm, clipvalue)

    def change_momentum(self, new_momentum):
        self.momentum.set_value(float(new_momentum))

    def gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        for pp, gg in zip(params, grads):
            value = pp.get_value(borrow=True)
            self.velocity = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'nesterov_velocity_'+pp.name)
            self.params.append(self.velocity)
            self.velocity.tags = ['optimizer_param']
            x = self.momentum * self.velocity - self.lr * gg
            updates[self.velocity] = x
            updates[pp] = pp + self.momentum * x + self.lr * gg
        return updates


class RMSprop(BaseOptimizer):

    def __init__(self, lr_init=0.001, rho_init=0.9,
                 clipnorm=None, clipvalue=None):
        rho = np.array(rho_init).astype('float32')
        self.rho = theano.shared(rho, 'rho')
        self.rho.tags = ['rho']
        super(RMSprop, self).__init__(lr_init, clipnorm, clipvalue)

    def gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        for pp, gg in zip(params, grads):
            value = pp.get_value(borrow=True)
            self.accu = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'rmsprop_accu_'+pp.name)
            self.params.append(self.accu)
            self.accu.tags = ['optimizer_param']
            accu_new = self.rho * self.accu + (1 - self.rho) * T.sqr(gg)
            updates[self.accu] = accu_new
            updates[pp] = pp - (self.lr * gg / (T.sqrt(accu_new) + 1e-8))
        return updates


class AdaDelta(BaseOptimizer):

    def __init__(self, lr_init=1.0, rho_init=0.9,
                 clipnorm=None, clipvalue=None):
        rho = np.array(rho_init).astype('float32')
        self.rho = theano.shared(rho, 'rho')
        self.rho.tags = ['rho']
        super(AdaDelta, self).__init__(lr_init, clipnorm, clipvalue)

    def gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        for pp, gg in zip(params, grads):
            value = pp.get_value(borrow=True)
            self.accu = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'adadelta_accu_'+pp.name)
            self.delta_accu = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'adadelta_delta_accu_'+pp.name)
            self.params.append(self.accu)
            self.params.append(self.delta_accu)
            self.accu.tags = ['optimizer_param']
            self.delta_accu.tags = ['optimizer_param']
            accu_new = self.rho * self.accu + (1 - self.rho) * T.sqr(gg)
            updates[self.accu] = accu_new
            ud = gg * (T.sqrt(self.delta_accu) + 1e-8) / (T.sqrt(accu_new) + 1e-8)
            updates[pp] = pp - self.lr * ud
            delta_accu_new = self.rho * self.delta_accu + (1 - self.rho) * T.sqr(ud)
            updates[self.delta_accu] = delta_accu_new
        return updates


class Adam(BaseOptimizer):

    def __init__(self, lr_init=0.002, beta1_init=0.9, beta2_init=0.999,
                 clipnorm=None, clipvalue=None):
        beta1 = np.array(beta1_init).astype('float32')
        beta2 = np.array(beta2_init).astype('float32')
        self.beta1 = theano.shared(beta1, 'beta1')
        self.beta2 = theano.shared(beta2, 'beta2')
        self.beta1.tags = ['beta1']
        self.beta2.tags = ['beta2']
        super(Adam, self).__init__(lr_init, clipnorm, clipvalue)

    def gradients_to_updates(self, params, grads):
        updates = OrderedDict()
        self.i = theano.shared(np.array(0).astype('float32'), 'adam_i')
        self.params.append(self.i)
        self.i.tags = ['optimizer_param']
        i_t = self.i + 1
        fix1 = 1.0 - T.pow(self.beta1, i_t)
        fix2 = 1.0 - T.pow(self.beta2, i_t)
        lr_t = self.lr * (T.sqrt(fix2) / fix1)
        for pp, gg in zip(params, grads):
            value = pp.get_value(borrow=True)
            self.m = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'adam_m_'+pp.name)
            self.v = theano.shared(np.zeros(value.shape, dtype=theano.config.floatX), 'adam_v_'+pp.name)
            self.params.append(self.m)
            self.params.append(self.v)
            self.m.tags = ['optimizer_param']
            self.v.tags = ['optimizer_param']
            m_t = (self.beta1 * self.m) + ((1.0 - self.beta1) * gg)
            v_t = (self.beta2 * self.v) + ((1.0 - self.beta2) * T.sqr(gg))
            g_t = m_t / (T.sqrt(v_t) + 1e-8)
            p_t = pp - (lr_t * g_t)
            updates[self.m] = m_t
            updates[self.v] = v_t
            updates[pp] = p_t
        updates[self.i] = i_t
        return updates
