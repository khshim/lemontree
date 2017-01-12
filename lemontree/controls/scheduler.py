"""
This code includes learning rate scheduler classes.
Each object manage only one learning rate.
"""

import numpy as np
import theano
import theano.tensor as T


class BaseLearninigRateScheduler(object):
    """
    This class defines abstract base class of learning rate scheduler.
    """
    def __init__(self, lr_shared):
        """
        This function initializes the class.
        Get a pointer(?) of (shared variable) learning rate and manage it.

        Parameters
        ----------
        lr_shared: shared variable
            a shared variable used for learning rate.

        Returns
        -------
        None.
        """
        # check asserts
        assert 'lr' in lr_shared.tags, '"lr_shared" should be a learning rate shared variable.'

        # set members
        self.lr = lr_shared

    def change_learningrate(self, epoch):
        """
        This function change learning rate by function 'compute_learningrate'.
        Epoch is often used as an argument for the function.

        Parameters
        ----------
        epoch: int
            a positive integer value.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(epoch, int), '"epoch" should be a positive integer.'

        # print and change
        if epoch == 0:
            print('...Learning rate started at', self.lr.get_value())
        else:
            new_learningrate = self.compute_learningrate(epoch)
            self.lr.set_value(float(new_learningrate))
            print('...Learning rate changed to', new_learningrate)

    def compute_learningrate(self, epoch):
        """
        This function computes new learning rate based on epoch and current learning rate.

        Parameters
        ----------
        epoch: int
            a positive integer value.

        Returns
        -------
        None.
        """
        raise NotImplementedError('Abstract class method')


class LearningRateDecayScheduler(BaseLearninigRateScheduler):
    """
    This class implements learning rate decay scheduler.
    """
    def __init__(self, lr_shared, decay_rate=0.05):
        """
        This function initializes the class.

        Parameters
        ----------
        lr_shared: shared variable
            a shared variable used for learning rate.
        decay_rate: float, default: 0.05
            a positive float value which is in range (0,1).

        Returns
        -------
        None.
        """
        super(LearningRateDecayScheduler, self).__init__(lr_shared)        
        # check asserts
        assert decay_rate > 0 and decay_rate < 1, '"decay_rate" should be in range (0,1).'

        # set members
        self.decay_rate = decay_rate

    def compute_learningrate(self, epoch):
        """
        This function overrides parents' ones.
        Compute learning rate using epoch.

        Math Expression
        ---------------
        lr = lr / (1 + decay_rate * epoch)

        Parameters
        ----------
        epoch: int
            a positive integer value.

        Returns
        -------
        float
            a float value computed.
        """
        return self.lr.get_value() / (1.0 + self.decay_rate * epoch)


class LearningRateMultiplyScheduler(BaseLearninigRateScheduler):
    """
    This class implements learning rate multiply scheduler.
    """
    def __init__(self, lr_shared, mult_rate=0.2):
        """
        This function initializes the class.

        Parameters
        ----------
        lr_shared: shared variable
            a shared variable used for learning rate.
        mult_rate: float, default: 0.2
            a positive float value which is in range (0,1).
            for default 0.2, each time its value becames one fifth.

        Returns
        -------
        None.
        """
        super(LearningRateMultiplyScheduler, self).__init__(lr_shared)
        # check asserts
        assert mult_rate > 0 and mult_rate < 1, '"mult_rate" should be in range (0,1).'

        # set members
        self.mult_rate = mult_rate

    def compute_learningrate(self, epoch):
        """
        This function overrides parents' ones.
        Compute learning rate using epoch.

        Math Expression
        ---------------
        lr = lr * mult_rate

        Parameters
        ----------
        epoch: int
            a positive integer value.

        Returns
        -------
        float
            a float value computed.
        """
        return self.lr.get_value() * self.mult_rate
