"""
This code includes sequence generators to make minibatch and shuffle.
Generator make data to feed into training function for every mini-batch.
Each generator can hold multiple data, such as data and mask.
Sequence Generator preprocess sequence before training and during training.
"""

import numpy as np
from lemontree.generators.generator import SimpleGenerator
from lemontree.data.glove import GloveData


class GloVeWordLMGenerator(SimpleGenerator):
    """
    This class get sequence data 3D tensor and generate mini-batch of words.
    Using GloVe word vector to make word embedding.
    Words are saved as integer.
    Words are converted to word vector while generating each mini-batch.
    Assume <EOS> and/or <SOS> are added while reading data.
    """
    def __init__(self, data_list, glove, sequence_length=20, overlap_length=None, batch_size=128, name=None, seed=333):
        """
        This function initializes the class.

        Parameters
        ----------
        data_list: list
            a list of data, each include equal number of data.
        glove: GloveData
            a GloveData which includes the embeddings and dictionary.
        sequence_length: integer, default: 100
            an integer which indicates each length of the sentence.
            in otherwords, the number of timestep RNN gets each time.
        overlap_length: integer, default: None
            an integer which indicates the index of next mini-batch starts.
            if None, sequence_length == overlap length.
            i.e., next batch starts from next word.
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
        super(GloVeWordLMGenerator, self).__init__(data_list, batch_size, name, seed)
        # check asserts
        assert isinstance(glove, GloveData), '"glove" should be GloveData instance.'
        assert isinstance(sequence_length, int), '"sequence_length" should be a positive integer.'
        if overlap_length is None:
            overlap_length = sequence_length
        else:
            assert isinstance(overlap_length, int), '"overlap_length" should be a positive integer.'
            assert overlap_length <= sequence_length, '"overlap_length" should be smaller than "sequence_length".'

        # set members
        self.glove = glove
        self.sequence_length = sequence_length
        self.overlap_length = overlap_length
        self.preserved_data = {}  # (batch index, data)

    def get_minibatch(self, index):
        """
        This function overrides parents' ones.
        Generates the mini batch data.
        Convert word indices to word embedding vector.

        Parameters
        ----------
        index: int
            an integer value that indicates which mini batch will be returned.

        Returns
        -------
        tuple
            a tuple of partial data in data list.
            (3D data, 2D mask, 2D label, 1D reset)
            if reset = 0, next batch cell/hidden initialization will be 0.
            if reset = 1 , next batch cell/hidden initialization
                next specific batch index will be continued data.
        """
        # check asserts
        assert index < self.max_index

        # prepare minibatch
        minibatch = [self.data_list[0][i] for i in self.order[self.batch_size * index: self.batch_size * (index+1)]]
        for k in self.preserved_data.keys():
            minibatch[k] = self.preserved_data[k]  # replace if data should be processed continuously
        next_preserved_data = {}

        mask = np.ones((self.batch_size, self.sequence_length), dtype = 'int32')
        label = np.zeros((self.batch_size, self.sequence_length), dtype = 'int32')
        reset = np.zeros((self.batch_size,), dtype = 'int32')
        
        """
        Case explain
        ------------
        Assume sequence_length = 4, overlap_length = 2

        if len(data) <= sequence_length + 1:
            data: [1,2,3]
                1) make label, [2,3]
                2) padding data to fit sequence length, remove last one. [1,2,0,0]
                3) padding label to fit sequence length. [2,3,0,0]
                4) mask [1,1,0,0]
                5) reset = 0 (there will be new sequence next time)
                    hope cell_init = 0, hidden_init = 0 next time
        elif len(data) > sequence_length + 1:
            data: [1,2,3,4,5,6,7]
                1) make label, [2,3,4,5]
                2) save data [3,4,5,6,7]
                3) cropping data to fit sequence length, [1,2,3,4]
                4) mask [1,1,1,1]
                6) reset = 1 (there will be continues sequence next time)
                    hope cell_init, hidden_init is properly given next time.
        """
        for i in range(self.batch_size):
            data_length = len(minibatch[i])
            if data_length <= self.sequence_length + 1:
                # make label
                label[i, :data_length-1] = np.asarray(minibatch[i][1:], dtype='int32')
                # pad data
                minibatch[i] = minibatch[i][:-1] + [0] * (self.sequence_length - data_length + 1)
                # get mask
                mask[i, data_length-1:] = 0                
            else:
                # make label
                label[i] = np.asarray(minibatch[i][1:1+self.sequence_length])
                # save data
                next_preserved_data[i] = minibatch[i][self.overlap_length:]
                # crop data
                minibatch[i] = minibatch[i][:self.sequence_length]
                # reset
                reset[i] = 1
            minibatch[i] = self.glove.indices_to_vec(minibatch[i])
        np_minibatch = np.asarray(minibatch, dtype = 'float32')

        self.preserved_data = next_preserved_data

        return (np_minibatch, mask, label, reset)