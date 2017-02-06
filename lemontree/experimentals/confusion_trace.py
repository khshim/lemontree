"""
This code includes Confusion matrix computation.
Return trace of confusion matrix as loss.
"""

import numpy as np
import theano
import theano.tensor as T
from lemontree.objectives import BaseObjective


class ConfusionTrace(BaseObjective):
    """
    This class implements confusion matrix trace.
    """
    def __init__(self, num_class):
        """
        This function initializes the class.

        Parameters
        ----------
        num_class: int
            an integer how many classes will be exist.

        Returns
        -------
        None.
        """
        self.num_class = num_class

    def get_loss(self, predict, label):
        """
        This function overrides the parents' one.
        Computes the loss by model prediction and real label.
        use theano implemented categorical_crossentropy directly.

        Parameters
        ----------
        predict: TensorVariable
            an array of (batch size, prediction).
            for accuracy task, "predict" is 2D matrix.
        label: TensorVariable
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for classification, highly recommend second one.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """

        if label.ndim == 1:
            one_hot_label = T.eq(label.dimshuffle(0,'x'), T.arange(self.num_class).dimshuffle('x',0))
        elif label.ndim == 2:
            one_hot_label = label
        predict_max = T.argmax(predict, axis=-1)
        one_hot_predict = T.eq(predict_max.dimshuffle(0,'x'), T.arange(self.num_class).dimshuffle('x',0))
        
        confusion = T.dot(T.transpose(one_hot_label), one_hot_predict)
        confusion_sum = T.sum(confusion, axis = 1)
        confusion_norm = confusion / (confusion_sum.dimshuffle(0,'x') + 1e-7)
        return T.nlinalg.trace(confusion_norm)