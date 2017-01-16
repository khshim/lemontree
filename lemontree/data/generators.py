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
        None.
        """
        # check asserts
        assert index <= self.max_index, '"index" should be below maximum index.'

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
        None.
        """
        return tuple(self.data_list)


class ImageGenerator(SimpleGenerator):
    """
    This class get image data 4D tensor and generate mini-batch.
    Only one of the input data can be an image.
    YUV, GCN, LCN, ZCA - preprocessing. done through all data.
    flip, padding, crop - done during making mini-batch.
    """
    def __init__(self, data_list, batch_size=128,
                 flip_lr=False, flip_ud=False, random_crop=False, padding=None, crop_size=None,
                 image_index=0, name=None, seed=333):
        """
        This function initializes the class.
        Preprocessing is different to random generation.
        Preprocessing is done before training, while random generation is done for each mini-batch during training.

        Parameters
        ----------
        data_list: list
            a list of data, each include equal number of data.
        batch_size: int
            an integer value, which is the number of data in mini-batch.
        flip_lr: bool, default: False
            an bool value, whether randomly flip image left-right with probability 0.5 or not.
        flip_ud: bool, default: False
            an bool vaue, whether randomly flip image up-down with probability 0.5 or not.
        random_crop: bool, default: False
            an bool value, whether randomly crop image from the original data.
            first padding the image with nearest pixels.
            second crop image with given configuration.
        padding: tuple, default: None
            a tuple of (padding up, padding down, padding left, padding right).
        crop_size: tuple, default: None
            a tuple of (crop width, crop height).
        image_index: int, default: 0
            an integer which indicates which of the given data is an image.
            usually first one is image.
        name: string
            a string name of the class.
        seed: int
            an integer value of numpy random generator.
        

        Returns
        -------
        None.
        """
        super(ImageGenerator, self).__init__(data_list, batch_size, name, seed) 

        # check asserts
        assert isinstance(flip_lr, bool), '"flip_lr" should be a bool value.'
        assert isinstance(flip_ud, bool), '"flip_ud" should be a bool value.'
        if padding is not None:
            assert isinstance(padding, tuple) and len(padding) == 4, '"padding" should be a tuple of length 4.'
        assert isinstance(random_crop, bool), '"random_crop" should be a bool value.'
        if random_crop:
            assert isinstance(crop_size, tuple) and len(crop_size) == 2, '"random_crop_config" should be a tuple of length 2.'
        assert isinstance(image_index, int) and image_index < self.max_data, '"image_index" should be an integer under total data in "data_list".'
        assert len(self.data_list[image_index].shape) == 4, 'A data of "image_index" should be an image of 4D tensor.'

        # set members
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self.random_crop =  random_crop
        self.padding = padding
        self.crop_size = crop_size
        self.image_index = image_index

    def rgb_to_yuv(self):
        # assume r, g, b order in channel
        image = self.data_list[self.image_index]
        new_data = np.zeros_like(image, image.dtype)
        new_data[:, 0, :, :] = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        new_data[:, 1, :, :] = 0.5 - 0.168736 * image[:, 0, :, :] - 0.331364 * image[:, 1, :, :] + 0.5 * image[:, 2, :, :]
        new_data[:, 2, :, :] = 0.5 + 0.5 * image[:, 0, :, :] - 0.418688 * image[:, 1, :, :] - 0.081312 * image[:, 2, :, :]
        self.data_list[self.image_index] = new_data

    def yuv_to_rgb(self):
        # assume y, u, v order (or y, cb, cr) in channel
        image = self.data_list[self.image_index]
        new_data = np.zeros_like(image, image.dtype)
        new_data[:, 0, :, :] = image[:, 0, :, :] + 1.402 * (image[:, 2, :, :] - 0.5)
        new_data[:, 1, :, :] = image[:, 0, :, :] - 0.34414 * (image[:, 1, :, :] - 0.5) - 0.71414 * (image[:, 2, :, :] - 0.5)
        new_data[:, 2, :, :] = image[:, 0, :, :] + 1.772 * (image[:, 1, :, :] - 0.5)
        self.data_list[self.image_index] = new_data

    def gcn(self, mean=None, std=None):
        image = self.data_list[self.image_index]
        if mean is None and std is None:
            mean = np.mean(image, axis=0)
            std = np.std(image, axis=0)
        result = (image - mean) / (std + 1e-8)
        self.data_list[self.image_index] = result
        return mean, std

    def lcn(self):
        image = self.data_list[self.image_index]
        flat = np.reshape(image, (image.shape[0], np.prod(image.shape[1:])))
        mean = np.mean(flat, axis=1)
        flat = flat - mean[:, np.newaxis]
        # std = np.sqrt(np.sum(flat ** 2, axis = 1)) + 1e-8
        std = np.std(flat, axis=1)
        result = flat / (std[:, np.newaxis] + 1e-8)
        result = np.reshape(result, image.shape)
        self.data_list[self.image_index] = result

    def zca(self, pc_matrix=None):
        flat = np.reshape(self.data_list[self.image_index], (self.data_list[self.image_index].shape[0], np.prod(self.data_list[self.image_index].shape[1:])))
        # print('Flatten shape: ', flat.shape)

        if pc_matrix is None:
            sigma = np.dot(flat.T, flat) / flat.shape[0]
            U, S, V = np.linalg.svd(sigma)
            newS = np.diag(1.0 / (np.sqrt(S) + 1e-8))
            self.pc_matrix = np.dot(np.dot(U, newS), np.transpose(U))
        else:
            self.pc_matrix = pc_matrix

        white = np.dot(flat, self.pc_matrix)
        result = np.reshape(white, self.data.shape)
        self.data = result
        return self.pc_matrix

    def get_minibatch(self, index):
        assert index <= self.max_index
        batch_data = self.data[self.order[self.batchsize * index: self.batchsize * (index+1)]]
        batch_label = self.label[self.order[self.batchsize * index: self.batchsize * (index+1)]]
        if self.flip_lr:
            random_choice = np.random.permutation(self.batchsize)[:self.batchsize//2]
            batch_data[random_choice] = batch_data[random_choice, :, :, ::-1]
        if self.flip_ud:
            random_choice = np.random.permutation(self.batchsize)[:self.batchsize//2]
            batch_data[random_choice] = batch_data[random_choice, :, ::-1]
        return (batch_data, batch_label)


class SequenceGenerator(SimpleGenerator):

    def __init__(self, name=None, batchsize=128, sequence_length=50):
        super(SequenceGenerator, self).__init__(name, batchsize)
        self.sequence_length = sequence_length

    def initialize(self, data, label):
        self.data = []
        for i in range(len(data) // (self.batchsize * self.sequence_length)):
            self.data.append
        
    #def initialize(self, data, sort=False):
    #    if sort:
    #        self.data = sorted(data, key=len)  # list of sentences (strings)
    #    else:
    #        self.data = data
    #    self.ndata = len(data)
        
    #    if self.bucket > 0:
    #        assert sort  # To use bucketing, we should sort sentences
    #        self.bucket_key = []
    #        for ind in range(self.ndata // self.bucket, self.ndata, self.ndata // self.bucket):
    #            self.bucket_key.append(len(self.data[ind]))
    #        self.bucket_key = self.bucket_key[:self.bucket-1]



