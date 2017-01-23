"""
This code includes generators to make minibatch and shuffle.
Generator make data to feed into training function for every mini-batch.
Each generator can hold multiple data, such as data and label.
"""

import numpy as np


class SimpleGenerator(object):
    """
    This class get tensor type data and load to RAM.
    No special pre-processing included.
    """
    def __init__(self, data_list, batch_size=128, name=None, seed=333):
        """
        This function initializes the class.

        Parameters
        ----------
        data_list: list
            a list of data, each include equal number of data.
        name: string
            a string name of the class.
        seed: int
            an integer value of numpy random generator.
        batch_size: int
            an integer value, which is the number of data in mini-batch.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(data_list, list), '"data_list" should be a list of data.'
        assert isinstance(batch_size, int), '"batch_size" should be an integer.'
        assert isinstance(seed, int), '"seed" should be an integer value.'
        
        # set members
        self.data_list = data_list
        self.max_data = len(data_list)  # how many data included in input
        self.name = name
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        # all data should have same data
        self.ndata = len(data_list[0])  # first data, the number of data
        for dd in self.data_list:
            assert self.ndata == len(dd), 'All data in "data_list" should have same number of data.'
        self.order = self.rng.permutation(self.ndata)
        self.max_index = self.ndata // self.batch_size

    def shuffle(self):
        """
        This function shuffles the order of data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.order = self.rng.permutation(self.ndata)

    def change_batchsize(self, new_batch_size):
        """
        This function changes the initial batch size of the class.

        Parameters
        ----------
        new_batch_size: int
            an integer value which will replace the previous batch size.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(new_batch_size, int), '"new_batch_size" should be an integer.'

        self.batch_size = new_batch_size
        self.max_index = self.ndata // self.batch_size

    def get_minibatch(self, index):
        """
        This function generates the mini batch data.
        Only crop the data and return.

        Parameters
        ----------
        index: int
            an integer value that indicates which mini batch will be returned.

        Returns
        -------
        tuple
            a tuple of partial data in data list.
        """
        # check asserts
        assert index < self.max_index, '"index" should be below maximum index.'

        # make returns
        data = ()
        for dd in self.data_list:
            data = data + (dd[self.order[self.batch_size * index: self.batch_size * (index+1)]],)
        return data

    def get_fullbatch(self):
        """
        This function just return all data inside, as tuple type.

        Parameters
        ----------
        None.

        Returns
        -------
        tuple
            a tuple of full data in data list.
        """
        return tuple(self.data_list)
