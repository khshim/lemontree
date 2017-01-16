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
                 flip_lr=False, flip_ud=False, padding=None, crop_size=None,
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
        padding: tuple, default: None
            a tuple of (padding up, padding down, padding left, padding right).
        crop_size: tuple, default: None
            a tuple of (crop width, crop height).
            randomly crop the image of given size.
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
        if crop_size is not None:
            assert isinstance(crop_size, tuple) and len(crop_size) == 2, '"crop_size" should be a tuple of length 2.'
        assert isinstance(image_index, int) and image_index < self.max_data, '"image_index" should be an integer under total data in "data_list".'
        assert len(self.data_list[image_index].shape) == 4, 'A data of "image_index" should be an image of 4D tensor.'

        # set members
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self.padding = padding
        self.crop_size = crop_size
        self.image_index = image_index

    def rgb_to_yuv(self):
        """
        This function converts RGB channel to YUV channel.
        Assume r, g, b order in channel
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        image = self.data_list[self.image_index]
        new_data = np.zeros_like(image, image.dtype)
        new_data[:, 0, :, :] = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        new_data[:, 1, :, :] = 0.5 - 0.168736 * image[:, 0, :, :] - 0.331364 * image[:, 1, :, :] + 0.5 * image[:, 2, :, :]
        new_data[:, 2, :, :] = 0.5 + 0.5 * image[:, 0, :, :] - 0.418688 * image[:, 1, :, :] - 0.081312 * image[:, 2, :, :]
        self.data_list[self.image_index] = new_data

    def yuv_to_rgb(self):
        """
        This function converts RGB channel to YUV channel.
        Assume y, u, v order (or y, cb, cr) in channel
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        image = self.data_list[self.image_index]
        new_data = np.zeros_like(image, image.dtype)
        new_data[:, 0, :, :] = image[:, 0, :, :] + 1.402 * (image[:, 2, :, :] - 0.5)
        new_data[:, 1, :, :] = image[:, 0, :, :] - 0.34414 * (image[:, 1, :, :] - 0.5) - 0.71414 * (image[:, 2, :, :] - 0.5)
        new_data[:, 2, :, :] = image[:, 0, :, :] + 1.772 * (image[:, 1, :, :] - 0.5)
        self.data_list[self.image_index] = new_data

    def global_mean_sub(self, mean=None):
        """
        This function substract mean through all batch per pixel.

        Parameters
        ----------
        mean: ndarray, default: None
            an 3D numpy array which contains global mean of pixels.
            if none, compute mean and return.

        Returns
        -------
        ndarray
            an 3D numpy array which contains global mean of pixels.
        """
        image = self.data_list[self.image_index]
        if mean is None:
            mean = np.mean(image, axis = 0)  # compute mean through batch dimension
        result = image - mean
        self.data_list[self.image_index] = result
        return mean

    def global_std_div(self, std=None):
        """
        This function divide std through all batch per pixel.

        Parameters
        ----------
        std: ndarray, default: None
            an 3D numpy array which contains global std of pixels.
            if none, compute mean and return.

        Returns
        -------
        ndarray
            an 3D numpy array which contains global std of pixels.
        """
        image = self.data_list[self.image_index]
        if std is None:
            std = np.std(image, axis = 0)  # compute std through batch dimension
        result = image / (std + 1e-8)
        self.data_list[self.image_index] = result
        return std

    def gcn(self, mean=None, std=None):
        """
        This function is combination of "global_mean_sub" and "global_std_div".

        Parameters
        ----------
        mean: ndarray, default: None
            an 3D numpy array which contains global mean of pixels.
            if none, compute mean and return.
        std: ndarray, default: None
            an 3D numpy array which contains global std of pixels.
            if none, compute mean and return.

        Returns
        -------
        tuple
            two 3D numpy array which contains global mean and std of pixels.
        """
        mean = self.global_mean_sub(mean)
        std = self.global_std_div(std)
        return mean, std

    def local_mean_sub(self):
        """
        This function substract mean through all pixel per data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        image = self.data_list[self.image_index]
        flat = np.reshape(image, (image.shape[0], np.prod(image.shape[1:])))
        mean = np.mean(flat, axis=-1)
        flat = flat - mean[:, np.newaxis]
        result = np.reshape(flat, image.shape)
        self.data_list[self.image_index] = result

    def local_std_div(self):
        """
        This function divide std through all pixel per data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        image = self.data_list[self.image_index]
        flat = np.reshape(image, (image.shape[0], np.prod(image.shape[1:])))
        std = np.std(flat, axis=-1)
        flat = flat / (std[:, np.newaxis] + 1e-8)
        result = np.reshape(flat, image.shape)
        self.data_list[self.image_index] = result

    def lcn(self):
        """
        This function is combination of "local_mean_sub" and "local_std_div".

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.local_mean_sub()
        self.local_std_div()

    def zca(self, pc_matrix=None):
        """
        This function performs ZCA transform of input data.
        All channels are flattened and used for convariance matrix computation.

        Parameters
        ----------
        pc_matrix: ndarray, default: None
            an 2D numpy array which contains ZCA whitening matrix.
            if none, compute pc_matrix.

        Returns
        -------
        ndarray
            an 2D numpy array of ZCA whitening matrix.
        """
        image = self.data_list[self.image_index]
        flat = np.reshape(image, (image.shape[0], np.prod(image.shape[1:])))

        if pc_matrix is None:
            sigma = np.dot(flat.T, flat) / flat.shape[0]
            U, S, V = np.linalg.svd(sigma)
            newS = np.diag(1.0 / (np.sqrt(S) + 1e-8))
            pc_matrix = np.dot(np.dot(U, newS), np.transpose(U))

        white = np.dot(flat, pc_matrix)
        result = np.reshape(white, image.shape)
        self.data_list[self.image_index] = result
        return pc_matrix

    def get_minibatch(self, index):
        """
        This function overrides parents' ones.
        Generates the mini batch data.
        If preprocessing exist, do preprocessing for cropped mini-batch first.

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
        index = 0
        for dd in self.data_list:
            if index == self.image_index:
                image = dd[self.order[self.batch_size * index: self.batch_size * (index+1)]]
                if self.flip_lr:
                    random_choice = self.rng.permutation(self.batchsize)[:self.batchsize//2]
                    image[random_choice] = image[random_choice, :, :, ::-1]
                if self.flip_ud:
                    random_choice = self.rng.permutation(self.batchsize)[:self.batchsize//2]
                    image[random_choice] = image[random_choice, :, ::-1]
                if self.padding is not None:
                    image = np.pad(image, ((0,),(0,),(self.padding[0], self.padding[1]),(self.padding[2], self.padding[3])), mode='edge')
                if self.crop_size is not None:
                    random_row = self.rng.randint(0, image.shape[2] - self.crop_size[0] + 1)
                    random_col = self.rng.randint(0, image.shape[3] - self.crop_size[1] + 1)
                    image = image[:,:,random_row:random_row + self.crop_size[0], random_col:random_col + self.crop_size[1]]
                data = data + (image,)
            else:
                data = data + (dd[self.order[self.batch_size * index: self.batch_size * (index+1)]],)
        return data


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



