"""
This code includes sequence generators to make minibatch and shuffle.
Generator make data to feed into training function for every mini-batch.
Each generator can hold multiple data, such as data and mask.
Sequence Generator preprocess sequence before training and during training.
"""

import numpy as np
from lemontree.generators.generator import SimpleGenerator
from lemontree.data.glove import GloveData


class WordSequenceGenerator(SimpleGenerator):
    """
    This class get sequence data 3D tensor and generate mini-batch of words.
    Multiple sequences are allowed.
    Each sequence is converted to (batch_size, sequence_length).
    Each sequence is also with mask of (batch_size, sequence_length).
    Words are saved as integer.
    Words are converted to word vector during training.
    Assume <EOS> and <SOS> are added while reading data.
    """
    def __init__(self, data_list, embedding, batch_size=128,
                 sequence_length=16, overlap_length=8,
                 use_label_as_shift=1, name=None, seed=335):
        pass