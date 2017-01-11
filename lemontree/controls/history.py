# Kyuhong Shim 2016

import numpy as np
import theano
import theano.tensor as T


class History(object):

    def __init__(self):
        self.history = {}
        self.history['train_loss'] = []
        self.history['train_accuracy'] = []
        self.history['valid_loss'] = []
        self.history['valid_accuracy'] = []
        self.history['test_loss'] = []
        self.history['test_accuracy'] = []

    def add_key(self, key):
        self.history[key] = []

    def best_valid_loss(self):
        valid_loss_np = np.asarray(self.history['valid_loss'])
        return np.min(valid_loss_np), np.argmin(valid_loss_np)

    def remove_history_after(self, index, keylist=None):
        if keylist is None:
            self.history['train_loss'] = self.history['train_loss'][:index + 1]
            self.history['train_accuracy'] = self.history['train_loss'][:index + 1]
            self.history['valid_loss'] = self.history['valid_loss'][:index + 1]
            self.history['valid_accuracy'] = self.history['valid_accuracy'][:index + 1]
        else:
            for key in keylist:
                self.history[key] = self.history[key][:index + 1]

    def print_history_recent(self, keylist=None):
        if keylist is None:
            print('......Train loss', self.history['train_loss'][-1])
            print('......Train accuracy', self.history['train_accuracy'][-1])
            print('......Valid loss', self.history['valid_loss'][-1])
            print('......Valid accuracy', self.history['valid_accuracy'][-1])
        else:
            for key in keylist:
                print('......' + key, self.history[key][-1])
    # TODO: export history to csv using pandas


class HistoryWithEarlyStopping(History):

    def __init__(self, max_patience=10, max_change=3):
        super(HistoryWithEarlyStopping, self).__init__()
        self.patience = 0
        self.change = 0
        self.max_patience = max_patience
        self.max_change = max_change

    def check_earlystopping(self):
        val_loss = self.history['valid_loss'][-1]
        current_best_loss, current_best_epoch = self.best_valid_loss()
        if current_best_loss < val_loss:
            self.patience += 1
            print('......current patience', self.patience)
            print('......current best valid loss', current_best_loss)
            if self.patience >= self.max_patience:
                self.change += 1
                self.patience = 0
                print('......current change', self.change)
                if self.change >= self.max_change:
                    return 2  # we should load param, stop training, remove history
                else:
                    return 1  # we should load param, change learning rate, remove history
            else:
                return 3  # we should keep training
        else:
            self.patience = 0
            print('......current patience', self.patience)
            print('......current best valid loss', current_best_loss)
            return 0  # we should save param
