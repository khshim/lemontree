"""
This code incudes various type of merge layers.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class MergeAddLayer(BaseLayer):
    """
    This class implements layer add.
    """
    def __init__(self):
        """
        This function initializes the class.
        The shape of two tensor should be equal.
        """
        super(MergeAddLayer, self).__init__()

    def get_output(self, input1_, input2_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return input1_ + input2_


class MergeConcatenateLayer(BaseLayer):
    """
    This class implements layer concatenation.
    """
    def __init__(self, axis=0):
        """
        This function initializes the class.
        The shape of two tensor should be equal except given axis.
        """
        super(MergeConcatenateLayer, self).__init__()
        self.axis = axis

    def get_output(self, input1_, input2_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.concatenate([input1_, input2_], axis=self.axis)

