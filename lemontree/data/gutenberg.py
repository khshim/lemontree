"""
This code includes functions to preprocess text in gutenberg dataset.
Especially, for certain book.
Download
--------
Project Gutenberg.
https://www.gutenberg.org/ebooks/

Base datapath: '~~~/data/'
Additional folder structure: '~~~/data/gutenberg/alice_in_wonderland.txt'
SAVE THE TEXT TO ANSI FORMAT.
"""

import time
import numpy as np


class GutenbergWordData(object):
    """
    This class use gutenberg book text to make better tensor form.
    Word level sequence pre-processing.
    """
    def __init__(self, base_datapath, mode='alice_in_wonderland', eos=True, sos=False, lower=True):
        """
        This function initializes the class.
        Currently, only one book per class is implemented.
        Reading multiple books at one time is for future implementation.
        For initialization, split text into sentences.

        Parameters
        ----------
        base_datapath: string
            a string path where textbook is saved.
        mode: string, default: 'alice_in_wonderland'
            a string which will be target book.
            i.e., 'alice_in_wonderland.txt'
        eos: bool, default: True.
            a bool value to determine whether to put <eos> at the end of sentences.
        sos: bool, default: False.
            a bool value to determine wheter to put <sos> at the front of sentences.
        lower: bool, default: True.
            a bool value, whether we should lower all cases or not (for english).

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(base_datapath, str), '"base_datapath" should be a string path.'
        assert isinstance(mode, str), '"mode" should be a string name for textbook.'
        assert isinstance(eos, bool), '"eos" should be a bool value to determine <eos> insert or not.'
        assert isinstance(sos, bool), '"eos" should be a bool value to determine <sos> insert or not.'

        # load
        book_file =  base_datapath + 'gutenberg/' + mode + '.txt'
        print('Gutenberg load book:', book_file)
        start_time = time.clock()
        import nltk

        with open(book_file, 'r') as f:
            if lower:
                corpus = f.read().lower().replace('\n', ' ')
            else:
                corpus = f.read().replace('\n', ' ')
            # nltk.download()  # download model -> punkt if you get an error
            self.sentences = nltk.tokenize.sent_tokenize(corpus)  # a list of sentences, each sentence is string
            for i in range(len(self.sentences)):
                words_from_string = nltk.tokenize.word_tokenize(self.sentences[i])
                if eos:
                    words_from_string = words_from_string + ['<EOS>']
                if sos:
                    words_from_string = ['<SOS>'] + words_from_string
                self.sentences[i] = words_from_string  # string to word, now sentence is list of list of words
        print('Gutenberg number of sentences:', len(self.sentences))
        end_time = time.clock()
        print('Gutenberg load time:', end_time - start_time)
            

if __name__ == '__main__':
    base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
    guten = GutenbergWordData(base_datapath, 'alice_in_wonderland',
                              eos=True, sos=False)

    for i in range(10):
        seq = guten.sentences[i]
        print(len(seq))
