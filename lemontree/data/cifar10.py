"""
This code includes CIFAR-10 data usage.

Download
--------
CIFAR-10 data as matlab file.
https://www.cs.toronto.edu/~kriz/cifar.html
# Download data_batch_1.mat ~ data_batch_5.mat, test_batch.mat

Base datapath: '~~~/data/'
Additional foder structure: '~~~/data/cifar10/...mat' (total 6 files)
"""

import numpy as np
import scipy.io
from lemontree.utils.data_utils import split_data, int_to_onehot
from lemontree.utils.type_utils import uint_to_float, uint_to_int
from lemontree.data.dataset import BaseDataset


class CIFAR10(BaseDataset):
    """
    This class load cifar-10 dataset to a tensor form.
    Flat mode returns 2D matrix, and tensor mode returns 4D matrix.
    """
    def __init__(self, base_datapath, mode='tensor', seed=999, one_hot=False):
        """
        This function initializes the class.
        CIFAR-10 is not a big dataset, so we can hold all floating values in memory.

        Parameters
        ----------
        base_datapath: string
            a string path where mnist is saved.
        mode: string, default: 'tensor'
            a string either {'flat', 'tensor'}.
        seed: integer, default: 999
            a random seed to shuffle the data.
        one_hot: bool, default: False
            a bool value that choose wheter the label will be one-hot encoded or just integer.

        Returns
        -------
        None.
        """
        super(CIFAR10, self).__init__(base_datapath, seed)
        # check asserts
        assert mode in ['flat', 'tensor'], '"mode" should be a string either "flat" or "tensor".'

        # valid not yet divided
        self.valid_exist = False  # flag off

        # load
        cifar_train1 = scipy.io.loadmat(self.base_datapath + 'cifar10/data_batch_1.mat')
        cifar_train2 = scipy.io.loadmat(self.base_datapath + 'cifar10/data_batch_2.mat')
        cifar_train3 = scipy.io.loadmat(self.base_datapath + 'cifar10/data_batch_3.mat')
        cifar_train4 = scipy.io.loadmat(self.base_datapath + 'cifar10/data_batch_4.mat')
        cifar_train5 = scipy.io.loadmat(self.base_datapath + 'cifar10/data_batch_5.mat')
        cifar_test = scipy.io.loadmat(self.base_datapath + 'cifar10/test_batch.mat')
        train1_data = uint_to_float(cifar_train1['data']) / 255.0
        train2_data = uint_to_float(cifar_train2['data']) / 255.0
        train3_data = uint_to_float(cifar_train3['data']) / 255.0
        train4_data = uint_to_float(cifar_train4['data']) / 255.0
        train5_data = uint_to_float(cifar_train5['data']) / 255.0
        test_data = uint_to_float(cifar_test['data']) / 255.0
        train1_label = uint_to_int(cifar_train1['labels'])
        train2_label = uint_to_int(cifar_train2['labels'])
        train3_label = uint_to_int(cifar_train3['labels'])
        train4_label = uint_to_int(cifar_train4['labels'])
        train5_label = uint_to_int(cifar_train5['labels'])
        test_label = uint_to_int(cifar_test['labels'])

        train_data = np.vstack((train1_data, train2_data, train3_data,
                                train4_data, train5_data))
        train_label = np.vstack((train1_label, train2_label, train3_label,
                                 train4_label, train5_label))

        train_label = np.squeeze(train_label)
        test_label = np.squeeze(test_label)

        train_order = self.rng.permutation(train_label.shape[0])
        test_order = self.rng.permutation(test_label.shape[0])

        self.train_data = train_data[train_order]
        self.test_data = test_data[test_order]
        self.train_label = train_label[train_order]
        self.test_label = test_label[test_order]

        if mode == 'tensor':
            train_data = np.reshape(train_data, (train_data.shape[0], 3, 32, 32))
            test_data = np.reshape(test_data, (test_data.shape[0], 3, 32, 32))
        elif mode == 'flat':
            pass
        else:
            raise NotImplementedError('No such mode exist')

        if one_hot:
            self.train_label = int_to_onehot(self.train_label, 10)
            self.test_label = int_to_onehot(self.test_label, 10)


def label_to_real(label):
    if label == 0:
        return 'airplane'
    elif label == 1:
        return 'automobile'
    elif label == 2:
        return 'bird'
    elif label == 3:
        return 'cat'
    elif label == 4:
        return 'deer'
    elif label == 5:
        return 'dog'
    elif label == 6:
        return 'frog'
    elif label == 7:
        return 'horse'
    elif label == 8:
        return 'ship'
    elif label == 9:
        return 'truck'
    else:
        raise NotImplementedError('No such label exist')
