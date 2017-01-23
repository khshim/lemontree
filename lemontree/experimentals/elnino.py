"""
This code includes el-nino softmax.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class ElninoSoftmax(BaseLayer):
    """
    This class implements softmax activation function.
    """
    def __init__(self, name=None):
        """
        This function initializes the class.

        Parameters
        ----------
        name: string
            a string name of this class.

        Returns
        -------
        None.
        """
        super(ElninoSoftmax, self).__init__(name)

    def get_output(self, input_, label, elnino_matrix):
        """
        This function overrides the parents' one.
        Softmax converts output energy to probability distributuion.

        Math Expression
        -------------------

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.nnet.softmax(input_)