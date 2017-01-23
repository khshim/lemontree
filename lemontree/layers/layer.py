"""
This code includes base layer class for every layers.
Base types are feed-foward, recurrent, and merge.
"""

from collections import OrderedDict


class BaseLayer(object):
    """
    This class defines abstract base class for all layers.
    Every layer should have its own name, which is given from initialization.
    """
    def __init__(self, name=None):
        """
        This function initializes the class.

        Parameters
        ----------
        name: string
            a string name for this layer.

        Returns
        -------
        None.
        """
        # set members
        self.name = name

    def get_output(self, input_):
        """
        This function creates symbolic function to compute output from an input.
        Basically, each layer has only one input.

        Parameters
        ----------
        input_: TensorVariable
            a TensorVariable to compute the output.
            name 'input' is pre-defined, so we use 'input_' instead.

        Returns
        -------
        TensorVariable
            a TensorVariable computed by the layer.
        """
        raise NotImplementedError('Abstract class method')

    def get_params(self):
        """
        This function returns interal layer parameters.
        Parameters may or may not be trained by optimizers.
        If you don't want optimizer to train this params, there are two ways.
        First, you can give optimizer "exclude_tags" option for not generating updates.
        Second, you can add parameters after computing optimizer updates.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        return []  # for default
        # raise NotImplementedError('Abstarct class method')

    def get_updates(self):
        """
        This function returns internal updates.
        Updates are applied for each mini-batch training (or not).
        Layer updates can be merged with optimizer updates by "merge_dicts" function.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        return OrderedDict()  # for default
        # raise NotImplementedError('Abstract class method')
