"""
This code includes history recording class.
Each object writes the result in a single dictionary.
The result is saved as csv file.
"""

import os
import numpy as np
from collections import OrderedDict


class SimpleHistory(object):
    """
    This class implements base history class.
    Only keep records, do not control training.
    """
    def __init__(self, historydir):
        """
        This function initializes the class.
        History is kept in a dictionary, whose key is what we are tracking.
        History dictionary is {key: list} form, and list keep the history.
        Three keys are given by default.

        Parameters
        ----------
        historydir: string
            a string of path where we should make history directory.
            training history will saved in the directory.
            csv, image, etc can be saved.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(historydir, str), '"historydir" should be a string path.'

        # set members
        self.history = OrderedDict()
        self.historydir = historydir
        if not os.path.exists(self.historydir):
            os.makedirs(self.historydir)  # make directory if not exists

        # set default keys
        self.history['train_loss'] = []
        self.history['valid_loss'] = []
        self.history['test_loss'] = []       
        
    def add_keys(self, keys):
        """
        This function add keys to history.
        Usually, adding keys happen before training starts.

        Parameters
        ----------
        keys: list
            a list of strings which are key candidates.
            if given input key is already exist in the history, ignore the input.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(keys, list), '"keys" should be a list of string keys.'
        for kk in keys:
            if kk not in self.history.keys():
                self.history[kk] = []  # add new key

    def add_value_to_key(self, value, key):
        """
        This function add value to history of given key.
        Usually we keep epoch-level history, but it is optional.

        Parameters
        ----------
        value: float
            a float value to append to key.
        key: string
            a string which is already in key list.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(key, str), '"key" should be a string which indicates history key.'
        assert key in self.history.keys(), '"key" should be given as already existing history key.'

        # add
        self.history[key].append(value)

    def print_keys(self, keys):
        """
        This function print all current keys in history.

        Parameters
        ----------
        keys: list
            a list of strings which are key candidates.
            if given input key is already exist in the history, ignore the input.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(keys, list), '"keys" should be a list of string keys.'
        print('Keys in history:', self.history.keys())

    def best_loss_and_epoch_of_key(self, key='valid_loss'):
        """
        This function returns best (minimal) valid loss and which epoch it happened.
        Convenient function to get best result through history.

        Parameters
        ----------
        key: string, default: 'valid_loss'
            a string which is already in key list.
            usually 'valid_loss' is of interest because of early stopping.

        Returns
        -------
        None
        """
        # check asserts
        assert isinstance(key, str), '"key" should be a string which indicates history key.'
        assert key in self.history.keys(), '"key" should be given as already existing history key.'

        # find best
        best_loss = np.asarray(self.history[key])
        return np.min(best_loss), np.argmin(best_loss)  # loss, epoch order

    def remove_history_after_epoch(self, epoch, erase_keys=None):
        """
        This function removes all history after specific epoch.
        To erase history for only few keys, use 'erase_keys' argument.

        Parameters
        ----------
        epoch: int
            a positive integer which should be smaller than total history length.
            epoch'th history is kept, i.e., if 'epoch' is 3, history (0, 1, 2, 3) survives.
        erase_keys: list, default: None
            a list of string keys in history.
            if none, erase history of every keys.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(epoch, int), '"epoch" should be a positive integer no larger than maximum history.'
        if erase_keys is not None:
            assert isinstance(erase_keys, list), '"erase_keys" should be a list of string keys.'
            for kk in erase_keys:
                assert kk in self.history.keys(), 'Some key in "erase_keys" is not in history.'

        # erase
        if keys is None:
            for key in self.history.keys():
                self.history[key] = self.history[key][:epoch + 1]
        else:
            for key in erase_keys:
                self.history[key] = self.history[key][:epoch + 1]

    def print_history_of_epoch(self, epoch=-1, print_keys=None):
        """
        This function prints history at certain epoch.
        Usually we see current history right after each training epoch is done.

        Parameters
        ----------
        epoch: int
            an integer (not restricted to positive) which should be smaller than total hitory length.
        print_keys: list, default: None
            a list of string keys in history.
            if none, print through all keys.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(epoch, int), '"epoch" should be a positive integer no larger than maximum history.'
        if print_keys is not None:
            assert isinstance(print_keys, list), '"print_keys" should be a list of string keys.'
            for kk in print_keys:
                assert kk in self.history.keys(), 'Some key in "print_keys" is not in history.'

        # print
        if print_keys is None:
            for key in self.history.keys():
                if len(self.history[key]) != 0:
                    print('......', key, self.history[key][epoch])
        else:
            for key in keylist:
                if len(self.history[key]) != 0:
                    print('......', key, self.history[key][epoch])
    
    def save_history_to_csv(self, save_keys=None):
        """
        This function saves history to csv file in history directory.

        Parameters
        ----------
        save_keys: list, default: None
            a list of string keys in history.
            if none, save history of every keys.

        Returns
        -------
        None.
        """
        # check asserts
        if save_keys is not None:
            assert isinstance(save_keys, list), '"save_keys" should be a list of string keys.'
            for kk in save_keys:
                assert kk in self.history.keys(), 'Some key in "save_keys" is not in history.'

        # select and sort keys
        if save_keys is None:
            csv_keys = sorted(self.history.keys())  # sort alphabetically
            csv_filename = 'history_all.csv'
        else:
            csv_keys = sorted(save_keys)
            csv_filename = 'history'
            for key in csv_keys:
                csv_filename += '_' + key
            csv_filename += '.csv'

        # save to csv
        import csv
        from itertools import zip_longest
        csv_rows = zip_longest(*self.history.values())
        with open(self.historydir + csv_filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter = '\t')
            writer.writerow(csv_keys)  # writerow 
            writer.writerows(csv_rows)  # writerow 's'
        print('History save done')        


class HistoryWithEarlyStopping(SimpleHistory):
    """
    This class implements History class with early stopping signals.
    Control (or try to control) training by returning behavior flags.
    Early stopping is applied.
    """
    def __init__(self, historydir, max_patience=10, max_change=3):
        """
        This function initializes the class.
        Initialize with how long we wait and how many times we change learning rate.
        Learning rate change is done by a scheduler, not by this class.

        Parameters
        ---------
        historydir: string
            a string of path where we should make history directory.
            training history will saved in the directory.
            csv, image, etc can be saved.
        max_patience: int
            a positive integer which means how long we will wait.
        max_change: int
            a positive integer which means how many times we drop the learning rate.

        Returns
        -------
        None.
        """
        super(HistoryWithEarlyStopping, self).__init__(historydir)

        # set members
        self.patience = 0  # current patience
        self.change = 0  # current change
        self.max_patience = max_patience
        self.max_change = max_change

    def check_earlystopping(self, key='valid_loss'):
        """
        This function check whether early stopping should happen or not.
        Also determines when to stop training.
        Usually done after each epoch, and monitors 'valid_loss'.

        Parameters
        ----------
        key: string
            a string key in history to monitor and control training.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(key, str), '"key" should be a string which indicates history key.'
        assert key in self.history.keys(), '"key" should be given as already existing history key.'

        # check conditions
        current_value = self.history[key][-1]  # added just before
        current_best_value, current_best_epoch = self.best_loss_and_epoch_of_key(key)

        if current_best_value < current_value:  # something wrong
            self.patience += 1  # increase patience
            print('...... current patience', self.patience)
            print('...... current best value of', key, current_best_value)
            print('...... current best epoch at', current_best_epoch)
            if self.patience >= self.max_patience:
                self.change += 1  # increase learning rate changes
                self.patience = 0  # rest patience, new start with new learning rate
                print('...... current lr change', self.change)
                if self.change >= self.max_change:  # enough learning rate changes
                    # load param, stop training, (remove history)
                    return 'end_train'
                else:
                    # load param, change learning rate, reset internals, (remove history)
                    return 'change_lr'
            else:
                # keep training
                return 'keep_train'
        else:  # doing good
            self.patience = 0  # reset patience
            print('...... current patience', self.patience)
            print('...... current best value of', key, current_best_value)
            print('...... current best epoch at', current_best_epoch)
            # save param, keep training
            return 'save_param'
