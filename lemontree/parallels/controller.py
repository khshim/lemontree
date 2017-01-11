# Kyuhong Shim 2017

import time
import numpy as np
import zmq
import logging
from multiprocessing import Process, Queue

class BaseController(object):
    """
    Multi-process Controller for data parallelism.
    """
    def __init__(self, port=5556, number_of_workers=1,
                 name='', logger_name='__main__'):
        self.port = port
        self.number_of_workers = number_of_workers
        self.name = name
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.queue = Queue()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind('tcp://*:{}'.format(self.port))

     