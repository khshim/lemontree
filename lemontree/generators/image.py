"""
This code includes image generators to make minibatch and shuffle.
Generator make data to feed into training function for every mini-batch.
Each generator can hold multiple data, such as data and label.
Image Generator preprocess image before training and during training.
"""

import numpy as np
from lemontree.generators.generator import SimpleGenerator


class ImageGenerator(SimpleGenerator):
    """
    This class get image data 4D tensor and generate mini-batch.
    Only one of the input data can be an image.
    YUV, GCN, LCN, ZCA - preprocessing. done through all data.
    flip, padding, crop - done during making mini-batch.
    """
    def __init__(self, data_list, batch_size=128,
                 flip_lr=False, flip_ud=False, padding=None, crop_size=None,
                 image_index=0, name=None, seed=334):
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
        result = image / (std + 1e-7)
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
        flat = flat / (std[:, np.newaxis] + 1e-7)
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
            newS = np.diag(1.0 / (np.sqrt(S) + 1e-7))
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
        tuple
            a tuple of partial data in data list.
        """
        # check asserts
        assert index <= self.max_index, '"index" should be below maximum index.'

        # make returns
        data = ()
        data_index = 0
        for dd in self.data_list:
            if data_index == self.image_index:
                image = dd[self.order[self.batch_size * index: self.batch_size * (index+1)]]
                if self.flip_lr:
                    random_choice = self.rng.permutation(self.batch_size)[:self.batch_size//2]
                    image[random_choice] = image[random_choice, :, :, ::-1]
                if self.flip_ud:
                    random_choice = self.rng.permutation(self.batch_size)[:self.batch_size//2]
                    image[random_choice] = image[random_choice, :, ::-1]
                if self.padding is not None:
                    image = np.pad(image, ((0,0),(0,0),(self.padding[0], self.padding[1]),(self.padding[2], self.padding[3])), mode='edge')
                if self.crop_size is not None:
                    random_row = self.rng.randint(0, image.shape[2] - self.crop_size[0] + 1)
                    random_col = self.rng.randint(0, image.shape[3] - self.crop_size[1] + 1)
                    image = image[:,:,random_row:random_row + self.crop_size[0], random_col:random_col + self.crop_size[1]]
                data = data + (image,)
            else:
                data = data + (dd[self.order[self.batch_size * index: self.batch_size * (index+1)]],)
            data_index += 1
        assert data_index == self.max_data
        return data
