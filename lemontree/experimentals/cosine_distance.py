"""
This code includes cosine distance measurement.
"""

import theano
import theano.tensor as T
from lemontree.objectives import BaseObjective


class CosineDistance(BaseObjective):
    """
    This class implements cosine distance between embedding matrix and target vector.
    """
    def __init__(self, embedding):
        """
        This function initializes the class.

        Parameters
        ----------
        embedding: ndarray
            a numpy array of shape (vocabulary, dimesion)

        Returns
        -------
        None.
        """
        # check asserts
        assert len(embedding.shape) == 2, '"embedding" should be a 2D matrix.'

        # make shared variable
        self.embedding = theano.shared(embedding, 'emb')
        self.embedding.tags = ['embedding']

    def get_loss(self, vector):
        """
        This function overrides the parents' one.
        Computes the loss by model prediction and real label.
        use theano implemented categorical_crossentropy directly.

        Parameters
        ----------
        vector: TensorVariable
            an array of (batch size, dimension).
            dimension should be same as embedding dimension.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable of size (batch_size, vocabulary).
        """
        # do
        vector_norm = T.sqrt(T.sum(T.square(vector), axis=-1) + 1e-7)  # (batch_size,)
        embedding_norm = T.sqrt(T.sum(T.square(self.embedding), axis=-1) + 1e-7)  # (vocabulary,)

        vector_dimshuffle = vector.dimshuffle(0, 'x', 1)  # (batch_size, 1, dimension)
        embedding_dimshuffle = self.embedding.dimshuffle('x', 0, 1)  # (1, vocabulary, dimension)

        inner_product = T.sum(vector_dimshuffle * embedding_dimshuffle, axis=-1)  # (batch_size, vocabulary)
        inner_product = inner_product / vector_norm.dimshuffle(0, 'x')   # (batch_size, vocabulary)
        inner_product = inner_product / embedding_norm.dimshuffle('x', 0)   # (batch_size, vocabulary)

        return inner_product