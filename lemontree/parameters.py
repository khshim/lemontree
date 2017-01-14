"""
This code includes parameter management classes.
Each object has its own parameter saving directory.
"""

import os
import numpy as np
from lemontree.utils.param_utils import filter_params_by_tags, print_tags_in_params


class SimpleParameter(object):
    """
    This class defines base class for saving and loading parameters.
    Although name includes 'simple', this is not abstract class and can be used generally.
    One class has only one parameter directory.
    """
    def __init__(self, params, paramdir):
        """
        This function initializes the class.

        Parameters
        ----------
        params: list
            a list of (shared variable) parameters.
        paramdir: string
            a string of path where we should make parameter directory.
            in other words, default load/save directory.
        """
        # check asserts
        assert isinstance(params, list), '"params" should be a list type.'
        assert isinstance(paramdir, str), '"paramdir" should be a string path.'

        # set members
        self.params = params
        self.paramdir = paramdir
        if not os.path.exists(self.paramdir):
            os.makedirs(self.paramdir)  # make directory if not exists

    def save_params(self, include_tags=None, exclude_tags=None, save_to_other_dir=None):
        """
        This function filters parameters which class itself holds, and save the parameters.
        If "include_tags" is None, we save all parameters.
        Else, we filter parameters and save only those.
        Save to the default directory, but may save to other directory.
        If the other target directory does not exist, we make directory first.

        Parameters
        ----------
        include_tags: list, default: None
            a list of (string) tags to include in return.
        exclude_tags: list, default: None
            a list of (string) tags to exclude in return.
        save_to_other_dir: string, default: None
            a string of path where we save parameter.

        Returns
        -------
        None.
        """
        # check asserts
        if not include_tags:  # if no specific guideline is given, which is the most common case
            include_tags = print_tags_in_params(self.params, False)  # include all tags to include all parameters
        assert isinstance(include_tags,  list), '"include_tags" should be a list type.'
        if exclude_tags is not None:
            assert isinstance(exclude_tags, list), '"exclude_tags" should be a list type.'
        if save_to_other_dir is not None:
            assert isinstance(save_to_other_dir, str), '"save_to_other_dir" should be a string path.'
            if not os.path.exists(save_to_other_dir):
                os.makedirs(save_to_other_dir)  # make directory if not exists

        # filter and save parameters
        filtered_params = filter_params_by_tags(self.params, include_tags, exclude_tags)
        for pp in filtered_params:
            if save_to_other_dir is None:  # save to default directory
                np.save(self.paramdir + pp.name + '.npy', pp.get_value())
            else:  # save to other directory
                np.save(save_to_other_dir + pp.name + '.npy', pp.get_value())
        print('...weight save done')

    def load_params(self, include_tags=None, exclude_tags=None, load_from_other_dir=None):
        """
        This function filters parameters which class itself holds, and load the parameters.
        If "incude_tags" is None, we load all parameters.
        Else, we filter parameters and load only those.
        Load from the default directory, but may load from other directory.

        Parameters
        ----------
        include_tags: list, default: None
            a list of (string) tags to include in return.
        exclude_tags: list, default: None
            a list of (string) tags to exclude in return.
        load_from_other_dir: string, default: None
            a string of path where we load parameter.

        Returns
        -------
        None.
        """
        # check asserts
        if not include_tags:  # if no specific guideline is given, which is the most common case
            include_tags = print_tags_in_params(self.params, False)  # include all tags to include all parameters
        assert isinstance(include_tags,  list), '"include_tags" should be a list type.'
        if exclude_tags is not None:
            assert isinstance(exclude_tags, list), '"exclude_tags" should be a list type.'
        if load_from_other_dir is not None:
            assert isinstance(load_from_other_dir, str), '"load_from_other_dir" should be a string path.'

        # filter and load parameters
        filtered_params = filter_params_by_tags(self.params, include_tags, exclude_tags)
        for pp in filtered_params:
            if load_from_other_dir is None:  # load from default directory
                pp.set_value(np.load(self.paramdir + pp.name + '.npy'))
            else:
                pp.set_value(np.load(load_from_other_dir + pp.name + '.npy'))
        print('...weight load done')
