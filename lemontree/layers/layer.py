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
    def __init__(self):
        """
        This function initializes the class.
        """
        # set members
        self.name = None
        self.batch_size = -1

    def set_name(self, name=None):
        """
        This function set name of the layer.
        Every layer should have name!

        Parameters
        ----------
        name: string
            a (new) name for this layer.
        """
        self.name = name

    def set_batch_size(self, batch_size=-1):
        """
        This function set batch size of the layer.
        Some layer require batch size!

        Parameters
        ----------
        batch_size: integer, default: -1
            an integer used for the graph.
        """
        self.batch_size = batch_size

    def set_shared(self):
        """
        This function creates shared variables used in this layer.

        Parameters
        ----------
        input_: TensorVariable
            a TensorVariable to compute the output.
            name 'input' is pre-defined, so we use 'input_' instead.
        """
        pass  # for default, no shared

    def set_shared_by(self, params):
        """
        This function loads shared variable params from other layer, same class.
        Assume the order is get_params() order.

        Parameters
        ----------
        params: list
            a list of shared variables.        
        """
        pass  # for default, no shared

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
        """
        return []  # for default

    def get_updates(self):
        """
        This function returns internal updates.
        Updates are applied for each mini-batch training (or not).
        Layer updates can be merged with optimizer updates by "merge_dicts" function.
        """
        return OrderedDict()  # for default
