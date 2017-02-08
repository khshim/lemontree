"""
This code includes base layer class for every layers.
Base types are feed-foward, recurrent, and merge.
"""

from lemontree.layers.layer import BaseLayer


class BaseRecurrentLayer(BaseLayer):
    """
    This class implements abstract base class for recurrent connection layers.
    """
    def __init__(self, gradient_steps=-1, output_return_index=[-1],
                 precompute=False, unroll=False, backward=False):
        """
        This function initializes the class.
        Arguments are the minimal requirements for theano scan operation.
        
        Parameters
        ----------
        gradient_steps: int, default: -1
            an integer which indicates unroll of rnn for backward path.
            if -1, pass gradient through whole sequence length.
            if a positive integer, pass gradient back only for that much.
        output_return_index: list, default: [-1]
            a list of integers which output step should be saved.
            if [-1], only final output is returned.
            if none, return all steps through sequence.
        precompute: bool, default: False
            a bool value determine input precomputation.    
            for speedup, we can precompute input in return of increased memory usage.
        unroll: bool, default: False
            a bool value determine recurrent loop unrolling.
            for speedup, we can unroll and compile theano function,
            in return of increased memory usage and much increased compile time.
        backward: bool, default: False
            a bool value determine the direction of sequence.
            although using backward True, output will be original order.
            
        Returns
        -------
        None.      
        """
        super(BaseRecurrentLayer, self).__init__()
        # check asserts
        assert isinstance(gradient_steps, int), '"gradient_steps" should be either positive integer or -1.'
        if output_return_index is not None:
            assert isinstance(output_return_index, list), '"output_return_index" should be a list of integer indices to return.'
        assert isinstance(precompute, bool), '"precompute" should be a bool value.'
        assert isinstance(backward, bool), '"backward" should be a bool value.'
        assert isinstance(unroll, bool), '"unroll" should be a bool value.'

        # set members
        self.gradient_steps = gradient_steps
        self.output_return_index = output_return_index
        self.precompute = precompute
        self.unroll = unroll
        self.backward = backward
        if unroll and gradient_steps <= 0:
            raise ValueError('Network Unroll requires exact gradient step')
