"""
This code includes functions to preprocess text in bookcorpus dataset.
11,038 books are included.
See "Skip-Thought Vectors" for bookcorpus usage.
(Ryan Kiros, Yukun Zhu et al., 2015.)
Contact authors to get bookcorpus dataset.

Using two text files.
Base datapath: '~~~/data/'
Additional foder structure: '~~~/data/bookcorpus/books_large_p1.txt' (p2.txt)

First, preprocess bookcorpus to make pickle.
Second, load it to class.
"""

import pickle
import numpy as np
from lemontree.data.dataset import BaseDataset
from lemontree.data.glove import GloveData
from lemontree.utils.data_utils import split_data


def read_bookcorpus(base_datapath, glove, mode='books_large_p1',
                    num_sentence=100000, min_length=2, max_length=100, seed=9685):
    """
    This function read bookcorpus data, tokenize to word-level, select and save sentences.
    For default, only put '<EOS>' at the end of the sentence.
    Save pickle of selected sentences for reuse.

    Parameters
    ----------
    base_datapath: string
        a string path where bookcorpus is saved.
    glove: GloveData
        a GloveData which includes the embeddings and dictionary.
    mode: string, default: 'books_large_p1'
        a string which file will be read. {'books_large_p1', 'books_large_p2'}
    num_sentences: integer, default: 100000
        an integer that how much sentences will be selected.
        the number is not yet filtered by length.
    min_length: integer, default: 1
        an integer which indicates minimum length of the sentence.
    max_length: integer, default: 100
        an integer which indicates maximum length of the sentence.
    seed: integer, default: 9685
        a random seed to select random sentences
    
    Returns
    -------
    None.
    """
    # check asserts
    assert isinstance(base_datapath, str), '"base_datapath" should be a string path.'
    assert isinstance(glove, GloveData), '"glove" should be GloveData instance.'
    assert mode in ['books_large_p1', 'books_large_p2'], '"mode" should be one name of the two files.'
    assert isinstance(num_sentence, int), '"num_sentence" should be a positive integer.'
    assert isinstance(min_length, int), '"min_length" should be a positive integer.'
    assert isinstance(max_length, int), '"max_length" should be a positive integer.'
    assert min_length < max_length, '"max_length" should be larger than "min_length".'
    assert isinstance(seed, int), '"seed" should be a positive integer.'

    file_name = base_datapath + 'bookcorpus/' + mode + '.txt'
    print('BookCorpus load corpus:', file_name)
    # glove = GloveData(base_datapath, load_pickle = True)
    
    import nltk
    # nltk.download()
    # download punkt module if not exist

    with open(file_name, 'r') as f:
        start_time = time.clock()
        corpus = f.read().lower().replace('\n', ' ')  # remove linebreak
        end_time = time.clock()
        print('BookCorpus load time:', end_time - start_time)

        # sentence tokenize all
        start_time = time.clock()
        sentences = nltk.tokenize.sent_tokenize(corpus)  # list of string
        end_time = time.clock()
        print('NLTK tokenize time:', end_time - start_time)
        print('Total sentences:', len(sentences))

        # word tokenize selected sentences, convert to indices
        start_time = time.clock()        
        selected_indices = np.random.permutation(len(sentences))[:num_sentence]
        selected_sentences = []
        count = 0

        for i in selected_indices:
            if count % 100 == 0:
                print('... count', count)
            if len(sentences[i]) >= min_length and len(sentences[i]) <= max_length:
                count += 1
                words_from_string = nltk.tokenize.word_tokenize(sentences[i])  # list of words
                words_from_string = words_from_string + ['<EOS>']
                words_indices = glove.words_to_indices(words_from_string)  # list of word indices
                selected_sentences.append(words_indices)

        end_time = time.clock()
        print('Total count', count)
        print('Word indice convert time:', end_time - start_time)
    
    # save pickle
    with open(base_datapath + 'bookcorpus/' + mode + '_' + str(num_sentence) + '.pickle', 'wb') as f:
        pickle.dump(selected_sentences, f, protocol=pickle.HIGHEST_PROTOCOL)


class BookCorpusWordCorpus(BaseDataset):
    """
    This class load bookcorpus in word indices.
    Assume input data is preprocessed by "read_bookcorpus" function.
    """
    def __init__(self, base_datapath, mode='books_large_p1_100000', seed=9686):
        """
        This function initializes the class.

        Parameters
        ----------
        base_datapath: string
            a string path where bookcorpus is saved.
        mode: string, default: 'books_large_p1_100000'
            a string which file will be read.
            should be in pickle format.
        seed: integer, default: 9685
            a random seed to select random sentences
    
        Returns
        -------
        None.
        """
        super(BookCorpusWordCorpus, self).__init__(base_datapath, seed)

        # check asserts
        assert isinstance(mode, str), '"mode" should be a string filename.'

        # load pickle
        corpus_file = self.base_datapath + 'bookcorpus/' + mode + '.pickle'
        with open(corpus_file, 'rb') as f:
            corpus = pickle.load(f)
        print('Corpus length:', len(corpus))

        # split data to train(90%), test(5%), valid(5%)
        train_data, test_data = split_data(corpus, int(len(corpus) * 0.95))
        train_data, valid_data = split_data(train_data, int(len(corpus) * 0.9))

        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data

        self.valid_exist = True