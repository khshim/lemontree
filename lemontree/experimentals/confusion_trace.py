import numpy as np
import theano
import theano.tensor as T
from lemontree.objectives import BaseObjective

class ConfusionTrace(BaseObjective):
    def __init__(self, num_class):
        self.num_class = num_class

    def get_loss(self, predict, label):
        one_hot_label = T.eq(label.dimshuffle(0,'x'), T.arange(self.num_class).dimshuffle('x',0))
        predict_max = T.argmax(predict, axis=-1)
        one_hot_predict = T.eq(predict_max.dimshuffle(0,'x'), T.arange(self.num_class).dimshuffle('x',0))
        
        confusion = T.dot(T.transpose(one_hot_label), one_hot_predict)
        confusion_sum = T.sum(confusion, axis = 1)
        confusion_norm = confusion / (confusion_sum.dimshuffle(0,'x') + 1e-8)
        return T.nlinalg.trace(confusion_norm)