"""
This code includes objectives for deep learning loss function.
Every computations are tensor operations.
"""

import numpy as np
import theano
import theano.tensor as T


class BaseObjective(object):
    """
    This class defines abstract base class for objectives.
    """
    def get_loss(self, predict, label):
        """
        This function computes the loss by model prediction and real label.

        Parameters
        ----------
        predict: ndarray
            an array of (batch size, prediction).
        label: ndarray
            an array of (batch size, answer).

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        raise NotImplementedError('Abstract class method')


class CategoricalCrossentropy(BaseObjective):

    def __init__(self, stabilize=False, mode='mean'):
        """
        This function initializes the class.

        Parameters
        ----------
        stabilize: bool, default: False
            a bool value to use stabilization or not.
            if yes, predictions are clipped to small, nonnegative values to prevent NaNs.
            the prediction slightly ignores the probability distribution assumtion of sum = 1.
            for most cases, it is OK to use 'False'.
            however, if you are using many-class such as imagenet, this option may matter.
        mode: string {'mean', 'sum'}, default: 'mean'
            a string to choose how to compute loss as a scalar.
            'mean' computes loss as an average loss through (mini) batch.
            'sum' computes loss as a sum loss through (mini) batch.

        Returns
        -------
        None.
        """
        # check assert
        assert isinstance(stabilize, bool), '"stabilize" should be a bool value.'
        assert mode in ['mean', 'sum'], '"mode" should be either "mean" or "sum".'

        # set members
        self.tags = ['loss', 'categorical_crossentropy']
        self.stabilize = stabilize
        self.mode = mode

    def get_loss(self, predict, label, mask=None):
        """
        This function overrides the parents' one.
        Computes the loss by model prediction and real label.
        use theano implemented categorical_crossentropy directly.

        Parameters
        ----------
        predict: ndarray
            an array of (batch size, prediction).
            for cross entropy task, "predict" is 2D matrix.
        label: ndarray
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for classification, highly recommend second one.
        mask: ndarray
            an array of (batchsize,) only contains 0 and 1.
            loss are summed or averaged only through 1.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        if mask is None:
            if self.mode == 'mean':
                if self.stabilize:
                    return T.mean(T.nnet.categorical_crossentropy(T.clip(predict, 1e-7, 1.0 - 1e-7), label))
                else:
                    return T.mean(T.nnet.categorical_crossentropy(predict, label))
            elif self.mode == 'sum':
                if self.stabilize:
                    return T.sum(T.nnet.categorical_crossentropy(T.clip(predict, 1e-7, 1.0 - 1e-7), label))
                else:
                    return T.sum(T.nnet.categorical_crossentropy(predict, label))
            else:
                raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')
        else:
            if self.mode == 'mean':
                if self.stabilize:
                    return T.sum(T.nnet.categorical_crossentropy(T.clip(predict, 1e-7, 1.0 - 1e-7), label) * mask) / T.sum(mask)
                else:
                    return T.sum(T.nnet.categorical_crossentropy(predict, label) * mask) / T.sum(mask)
            elif self.mode == 'sum':
                if self.stabilize:
                    return T.sum(T.nnet.categorical_crossentropy(T.clip(predict, 1e-7, 1.0 - 1e-7), label) * mask) / T.sum(mask)
                else:
                    return T.sum(T.nnet.categorical_crossentropy(predict, label) * mask)
            else:
                raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')


class CategoricalAccuracy(BaseObjective):

    def __init__(self, top_k=1):
        """
        This function initializes the class.

        Parameters
        ----------
        top_k: int, default: 1
            an integer that determines what will be correct.
            for k > 1, if an answer is in top-k probable labels, assigned as correct one.

        Returns
        -------
        None.
        """
        # check assert
        assert isinstance(top_k, int) and top_k > 0, '"top_k" should be a positive integer.'

        # set members
        self.tags = ['accuracy', 'categorical_accuracy']
        self.top_k = top_k

    def get_loss(self, predict, label, mask=None):
        """
        This function overrides the parents' one.
        Computes the loss by model prediction and real label.

        Parameters
        ----------
        predict: ndarray
            an array of (batch size, prediction).
            for accuracy task, "predict" is 2D matrix.
        label: ndarray
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for classification, highly recommend second one.
            should make label as integer.
        mask: ndarray
            an array of (batchsize,) only contains 0 and 1.
            loss are summed or averaged only through 1.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        if mask is None:
            if self.top_k == 1:
                if label.ndim == 1:
                    return T.mean(T.eq(T.argmax(predict, axis=-1), label))
                elif label.ndim == 2:
                    return T.mean(T.eq(T.argmax(predict, axis=-1), T.argmax(label, axis=-1)))
                else:
                    raise ValueError()
            else:
                # TODO: not yet tested
                top_k_predict = T.argsort(predict)[:, -self.top_k:]  # sort by values and keep top k indices
                if label.ndim == 1:
                    return T.mean(T.any(T.eq(top_k_predict, label), axis=-1))
                elif label.ndim == 2:
                    return T.mean(T.any(T.eq(top_k_predict, T.argmax(label,axis=-1)), axis=-1))
                raise ValueError()
        else:
            if self.top_k == 1:
                if label.ndim == 1:
                    return T.sum(T.eq(T.argmax(predict, axis=-1), label) * mask) / T.sum(mask)
                elif label.ndim == 2:
                    return T.sum(T.eq(T.argmax(predict, axis=-1), T.argmax(label, axis=-1)) * mask) / T.sum(mask)
                else:
                    raise ValueError()
            else:
                # TODO: not yet tested
                top_k_predict = T.argsort(predict)[:, -self.top_k:]  # sort by values and keep top k indices
                if label.ndim == 1:
                    return T.sum(T.any(T.eq(top_k_predict, label), axis=-1) * mask) / T.sum(mask)
                elif label.ndim == 2:
                    return T.sum(T.any(T.eq(top_k_predict, T.argmax(label,axis=-1)), axis=-1) * mask) / T.sum(mask)
                raise ValueError()


class BinaryCrossentropy(BaseObjective):

    def __init__(self, stabilize=False, mode='mean'):
        """
        This function initializes the class.

        Parameters
        ----------
        stabilize: bool, default: False
            a bool value to use stabilization or not.
            if yes, predictions are clipped to small, nonnegative values to prevent NaNs.
            the prediction slightly ignores the probability distribution assumtion of sum = 1.
            for most cases, it is OK to use 'False'.
            however, if you are using many-class such as imagenet, this option may matter.
        mode: string {'mean', 'sum'}, default: 'mean'
            a string to choose how to compute loss as a scalar.
            'mean' computes loss as an average loss through (mini) batch.
            'sum' computes loss as a sum loss through (mini) batch.

        Returns
        -------
        None.
        """
        # check assert
        assert isinstance(stabilize, bool), '"stabilize" should be a bool value.'
        assert mode in ['mean', 'sum'], '"mode" should be either "mean" or "sum".'

        # set members
        self.tags = ['loss', 'binary_crossentropy']
        self.stabilize = stabilize
        self.mode = mode

    def get_loss(self, predict, label):
        """
        This function overrides the parents' one.
        Computes the loss by model prediction and real label.
        use theano implemented binary_crossentropy directly.

        Parameters
        ----------
        predict: ndarray
            an array of (batch size, prediction).
            for cross entropy task, "predict" is 2D matrix.
        label: ndarray
            an array of or (batchsize,) whose value is 0 or 1.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        if self.mode == 'mean':
            if self.stabilize:
                return T.mean(T.nnet.binary_crossentropy(T.clip(predict, 1e-7, 1.0 - 1e-7), label))
            else:
                return T.mean(T.nnet.binary_crossentropy(predict, label))
        elif self.mode == 'sum':
            if self.stabilize:
                return T.sum(T.nnet.binary_crossentropy(T.clip(predict, 1e-7, 1.0 - 1e-7), label))
            else:
                return T.sum(T.nnet.binary_crossentropy(predict, label))
        else:
            raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')


class BinaryAccuracy(BaseObjective):

    def __init__(self):
        """
        This function initializes the class.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # set members
        self.tags = ['accuracy', 'binary_accuracy']

    def get_loss(self, predict, label):
        """
        This function overrides the parents' one.
        Computes the loss by model prediction and real label.

        Parameters
        ----------
        predict: ndarray
            an array of (batch size, prediction).
            for accuracy task, "predict" is 2D matrix.
        label: ndarray
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for classification, highly recommend second one.
            should make label as integer.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        # TODO: Not tested
        return T.mean(T.eq(T.gt(predict, 0.5), label)) 


class SquareLoss(BaseObjective):

    def __init__(self, mode='mean'):
        """
        This function initializes the class.

        Parameters
        ----------
        mode: string {'mean', 'sum'}, default: 'mean'
            a string to choose how to compute loss as a scalar.
            'mean' computes loss as an average loss through (mini) batch.
            'sum' computes loss as a sum loss through (mini) batch.

        Returns
        -------
        None.
        """
        # check assert
        assert mode in ['mean', 'sum'], '"mode" should be either "mean" or "sum".'

        # set members
        self.tags = ['loss', 'square_loss']
        self.mode = mode

    def get_loss(self, predict, label):
        """
        This function overrides the parents' one.
        Computes the loss by model prediction and real label.

        Parameters
        ----------
        predict: ndarray
            an array of (batch size, prediction).
            for accuracy task, "predict" is 2D matrix.
        label: ndarray
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for classification, highly recommend first one.
            should make label as one-hot encoding.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        if self.mode == 'mean':
            return T.mean(T.square(predict - label))
        elif self.mode == 'sum':
            return T.sum(T.square(predict - label))
        else:
            raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')


class WordPerplexity(BaseObjective):

    def __init__(self):
        """
        This function initializes the class.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.tags = ['loss', 'word_perplexity']

    def get_loss(self, predict, label, mask):
        """
        This function overrides the parents' one.
        Computes the loss by mode prediction and real label.
        
        Parameters
        ----------
        predict: ndarray
            an array of (batch size, prediction).
            for accuracy task, "predict" is 2D matrix.
        label: ndarray
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for word perplexity case, currently only second one is supported.
            should make label as integer.
        mask: ndarray
            an array of (batchsize,) only contains 0 and 1.
            loss are summed or averaged only through 1.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        if mask is None:
            return T.pow(2, -T.mean(T.log2(predict[T.arange(label.shape[0]), label])))
        else:
            return T.pow(2, -T.sum(T.log2(predict[T.arange(label.shape[0]), label]) * mask) / T.sum(mask))


class L1norm(BaseObjective):

    def __init__(self):
        """
        This function initializes the class.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # set members
        self.tags = ['loss', 'l1_norm']

    def get_loss(self, params):
        """
        This function overrides the parents' one.
        Computes the loss by summing absolute parameter values.

        Parameters
        ----------
        params: list
            a list of (shared variable) parameters.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # check asserts
        assert isinstance(params, list), '"params" should be a list type.'
        # do
        loss_sum = 0
        for pp in params:
            loss_sum += T.sum(T.abs_(pp))
        return loss_sum


class L2norm(BaseObjective):

    def __init__(self):
        """
        This function initializes the class.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # set members
        self.tags = ['loss', 'l2_norm']

    def get_loss(self, params):
        """
        This function overrides the parents' one.
        Computes the loss by summing squared parameter values.

        Parameters
        ----------
        params: list
            a list of (shared variable) parameters.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # check asserts
        assert isinstance(params, list), '"params" should be a list type.'
        # do
        loss_sum = 0
        for pp in params:
            loss_sum += T.sum(T.square(pp))
        return loss_sum
