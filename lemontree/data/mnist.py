"""
This code includes MNIST data usage.

Download
--------
MNIST data as matlab file.
http://www.cs.nyu.edu/~roweis/data.html
Download 'mnist_all.mat'.

Base datapath: '~~~/data/'
Additional foder structure: '~~~/data/mnist/mnist_all.mat'
"""

import numpy as np
import scipy.io
from lemontree.utils.data_utils import split_data, int_to_onehot
from lemontree.utils.type_utils import uint_to_float


class MNIST(object):
    """
    This class load mnist dataset to a tensor form.
    Flat mode returns 2D matrix, and tensor mode returns 4D matrix.
    """
    def __init__(self, base_datapath, mode='flat', seed=999, one_hot=False):
        """
        This function initializes the class.
        MNISTt is not a big dataset, so we can hold all floating values in memory.

        Parameters
        ----------
        base_datapath: string
            a string path where mnist is saved.
        mode: string, default: 'flat'
            a string either {'flat', 'tensor'}.
        seed: integer, default: 999
            a random seed to shuffle the data.
        one_hot: bool, default: False
            a bool value that choose wheter the label will be one-hot encoded or just integer.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(base_datapath, str), '"base_datapath" should be a string path.'
        assert mode in ['flat', 'tensor'], '"mode" should be a string either "flat" or "tensor".'

        # valid not yet divided
        self.valid_exist = False  # flag off

        # load
        mnist_file = base_datapath + 'mnist/mnist_all.mat'
        print('MNIST load file:', mnist_file)
        mnist = scipy.io.loadmat(mnist_file)

        # as default pre processing, we make [0, 255] to [0, 1]
        
        test0 = uint_to_float(mnist['test0']) / 255.0
        test1 = uint_to_float(mnist['test1']) / 255.0
        test2 = uint_to_float(mnist['test2']) / 255.0
        test3 = uint_to_float(mnist['test3']) / 255.0
        test4 = uint_to_float(mnist['test4']) / 255.0
        test5 = uint_to_float(mnist['test5']) / 255.0
        test6 = uint_to_float(mnist['test6']) / 255.0
        test7 = uint_to_float(mnist['test7']) / 255.0
        test8 = uint_to_float(mnist['test8']) / 255.0
        test9 = uint_to_float(mnist['test9']) / 255.0
        train0 = uint_to_float(mnist['train0']) / 255.0
        train1 = uint_to_float(mnist['train1']) / 255.0
        train2 = uint_to_float(mnist['train2']) / 255.0
        train3 = uint_to_float(mnist['train3']) / 255.0
        train4 = uint_to_float(mnist['train4']) / 255.0
        train5 = uint_to_float(mnist['train5']) / 255.0
        train6 = uint_to_float(mnist['train6']) / 255.0
        train7 = uint_to_float(mnist['train7']) / 255.0
        train8 = uint_to_float(mnist['train8']) / 255.0
        train9 = uint_to_float(mnist['train9']) / 255.0

        train_data = np.vstack((train0, train1, train2, train3, train4,
                                train5, train6, train7, train8, train9))
        test_data = np.vstack((test0, test1, test2, test3, test4,
                               test5, test6, test7, test8, test9))

        train_label = np.hstack((np.tile(np.array([0]), (train0.shape[0],)),
                                 np.tile(np.array([1]), (train1.shape[0],)),
                                 np.tile(np.array([2]), (train2.shape[0],)),
                                 np.tile(np.array([3]), (train3.shape[0],)),
                                 np.tile(np.array([4]), (train4.shape[0],)),
                                 np.tile(np.array([5]), (train5.shape[0],)),
                                 np.tile(np.array([6]), (train6.shape[0],)),
                                 np.tile(np.array([7]), (train7.shape[0],)),
                                 np.tile(np.array([8]), (train8.shape[0],)),
                                 np.tile(np.array([9]), (train9.shape[0],))))
        test_label = np.hstack((np.tile(np.array([0]), (test0.shape[0],)),
                                np.tile(np.array([1]), (test1.shape[0],)),
                                np.tile(np.array([2]), (test2.shape[0],)),
                                np.tile(np.array([3]), (test3.shape[0],)),
                                np.tile(np.array([4]), (test4.shape[0],)),
                                np.tile(np.array([5]), (test5.shape[0],)),
                                np.tile(np.array([6]), (test6.shape[0],)),
                                np.tile(np.array([7]), (test7.shape[0],)),
                                np.tile(np.array([8]), (test8.shape[0],)),
                                np.tile(np.array([9]), (test9.shape[0],))))

        self.rng = np.random.RandomState(seed)
        train_order = self.rng.permutation(train_label.shape[0])
        test_order = self.rng.permutation(test_label.shape[0])

        self.train_data = train_data[train_order]
        self.test_data = test_data[test_order]
        self.train_label = train_label[train_order]
        self.test_label = test_label[test_order]

        if mode == 'flat':
            pass
        elif mode == 'tensor':
            self.train_data = np.reshape(self.train_data, (self.train_data.shape[0], 1, 28, 28))
            self.test_data = np.reshape(self.test_data, (self.test_data.shape[0], 1, 28, 28))
        else:
            raise NotImplementedError('No such mode exist.')

        if one_hot:
            self.train_label = int_to_onehot(self.train_label, 10)
            self.test_label = int_to_onehot(self.test_label, 10)

    def split_train_valid(self, rule=0.9):
        self.valid_exist = True  # flag on
        self.train_data, self.valid_data = split_data(self.train_data, rule)
        self.train_label, self.valid_label = split_data(self.train_label, rule)

    def get_fullbatch_train(self):
        print('Train data shape: ', self.train_data.shape, self.train_data.dtype)
        print('Train label shape: ', self.train_label.shape, self.train_label.dtype)
        return self.train_data, self.train_label
    
    def get_fullbatch_test(self):
        print('Test data shape: ', self.test_data.shape, self.test_data.dtype)
        print('Test label shape: ', self.test_label.shape, self.test_label.dtype)
        return self.test_data, self.test_label

    def get_fullbatch_valid(self):
        assert self.valid_exist, 'First divide train and valid by "divide_train_valid".'
        print('Valid data shape: ', self.valid_data.shape, self.valid_data.dtype)
        print('Valid label shape: ', self.valid_label.shape, self.valid_label.dtype)
        return self.valid_data, self.valid_label