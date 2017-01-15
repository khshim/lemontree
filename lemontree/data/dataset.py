"""
This code includes abstract class for datasets.
Although not recommended, we use different class for each dataset.
"""

import numpy as np
from lemontree.utils.data_utils import split_data


class BaseDataset(object):
    """
    This class is an abstract base class for every dataset.
    """
    def __init__(self, base_datapath, seed):
        """
        This function initializes the class.
        
        Parameters
        ----------
        base_datapath: string
            a string path where mnist is saved.
        seed: integer, default: 999
            a random seed to shuffle the data.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(base_datapath, str), '"base_datapath" should be a string path.'
        assert isinstance(seed, int), '"seed" should be an integer value.'
        
        # set members
        self.base_datapath = base_datapath
        self.rng = np.random.RandomState(seed)

    def split_train_valid(self, rule=0.9):
        self.valid_exist = True  # flag on
        self.train_data, self.valid_data = split_data(self.train_data, rule)
        if hasattr(self, 'train_label'):
            self.train_label, self.valid_label = split_data(self.train_label, rule)

    def get_fullbatch_train(self):
        print('Train data shape: ', self.train_data.shape, self.train_data.dtype)
        if hasattr(self, 'train_label'):
            print('Train label shape: ', self.train_label.shape, self.train_label.dtype)
            return self.train_data, self.train_label
        else:
            return self.train_data
    
    def get_fullbatch_test(self):
        print('Test data shape: ', self.test_data.shape, self.test_data.dtype)
        if hasattr(self, 'test_label'):
            print('Test label shape: ', self.test_label.shape, self.test_label.dtype)
            return self.test_data, self.test_label
        else:
            return self.test_data

    def get_fullbatch_valid(self):
        assert self.valid_exist, 'First divide train and valid by "divide_train_valid".'
        print('Valid data shape: ', self.valid_data.shape, self.valid_data.dtype)
        if hasattr(self, 'valid_label'):
            print('Valid label shape: ', self.valid_label.shape, self.valid_label.dtype)
            return self.valid_data, self.valid_label
        else:
            return self.valid_data