# Kyuhong Shim 2016

import numpy as np
import scipy.io

# SVHN data as matlab file.
# http://ufldl.stanford.edu/housenumbers/
# Download train_32x32.mat, test_32x32.mat
# LCN


def uint_to_float(data):
    return np.asarray(data, dtype='float32')


def double_to_int(data):
    return np.asarray(data, dtype='int32')


def load_svhn(mode='tensor'):
    svhn_train = scipy.io.loadmat(base_datapath + 'svhn/train_32x32.mat')
    svhn_test = scipy.io.loadmat(base_datapath + 'svhn/test_32x32.mat')
    train_data = uint_to_float(svhn_train['X']) / 255.0
    test_data = uint_to_float(svhn_test['X']) / 255.0
    train_label = double_to_int(svhn_train['y'])
    test_label = double_to_int(svhn_test['y'])

    train_data = np.swapaxes(train_data, 0, 3)
    train_data = np.swapaxes(train_data, 1, 2)
    train_data = np.swapaxes(train_data, 2, 3)
    test_data = np.swapaxes(test_data, 0, 3)
    test_data = np.swapaxes(test_data, 1, 2)
    test_data = np.swapaxes(test_data, 2, 3)
    train_label = np.squeeze(train_label)
    test_label = np.squeeze(test_label)

    print('Train data shape: ', train_data.shape, train_data.dtype)
    print('Test data shape: ', test_data.shape, test_data.dtype)

    print('Train label shape: ', train_label.shape, train_label.dtype)
    print('Test label shape: ', test_label.shape, test_label.dtype)

    train_order = np.random.permutation(train_label.shape[0])
    test_order = np.random.permutation(test_label.shape[0])

    train_data = train_data[train_order]
    test_data = test_data[test_order]
    train_label = train_label[train_order]
    test_label = test_label[test_order]

    if mode == 'tensor':
        pass
    elif mode == 'flat':
        train_data = np.reshape(train_data, (train_data.shape[0], 3072))
        test_data = np.reshape(test_data, (test_data.shape[0], 3072))
    else:
        raise NotImplementedError('No such mode exist')

    return train_data, train_label, test_data, test_label
