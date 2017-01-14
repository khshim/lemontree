"""
This code implements sentence distortion to make wrong sentences.
Four distortion is introduced.
Shuffle (only one pair adjoint)
Delete (one)
Insert (one)
Shuffle (random)
"""

import numpy as np
from lemontree.data.glove import GloveData


def sentence_distortion_shuffle_one(words):
    """
    This function shuffles two adjacent words.

    Parameters
    ----------
    words: list
        a list of words (or word indices).

    Returns
    -------
    list
        a list of new words, same length as input.
    """
    # check asserts
    assert isinstance(words, list), '"words" should be a list of words or word indices.'

    # shuffle
    num = len(words)
    where = np.random.randint(0, num - 1)  # if num = 100, randomly choose between [0, 98]
    word_1 = words[where]
    word_2 = words[where + 1]
    words[where + 1] = word_1
    words[where] = word_2

    return words

def sentence_distortion_shuffle_random(words):
    """
    This function shuffles words in a random order.

    Parameters
    ----------
    words: list
        a list of words (or word indices).

    Returns
    -------
    list
        a list of new words, same length as input.
    """
    # check asserts
    assert isinstance(words, list), '"words" should be a list of words or word indices.'

    # shuffle
    num = len(words)
    how = np.random.permutation(num)  # if num = 100, randomly choose between [0, 98]
    new_words = []
    for i in range(num):
        new_words.append(words[how[i]])

    return new_words


def sentence_distortion_delete_one(words):
    """
    This function deletes a random word.

    Parameters
    ----------
    words: list
        a list of words (or word indices).

    Returns
    -------
    list
        a list of new words, one word is missing.
    """
    # check asserts
    assert isinstance(words, list), '"words" should be a list of words or word indices.'

    # shuffle
    num = len(words)
    where = np.random.randint(0, num)  # if num = 100, randomly choose between [0, 99]
    words.pop(where)

    return words


def sentence_distortion_insert_one(words, candidates):
    """
    This function deletes a random word.

    Parameters
    ----------
    words: list
        a list of words (or word indices).
    candidates: list
        a list of words which will enter.

    Returns
    -------
    list
        a list of new words, one word is missing.
    """
    # check asserts
    assert isinstance(words, list), '"words" should be a list of words or word indices.'

    # shuffle
    num = len(words)
    where = np.random.randint(0, num)  # if num = 100, randomly choose between [0, 99]
    what = np.random.randint(0, len(candidates))
    words.insert(where, candidates[what])

    return words


if __name__ == '__main__':
    words = ['a', 'cat', 'is', 'now', 'sleeping']
    shuffle_one = sentence_distortion_shuffle_one(words)
    print(shuffle_one)
    delete_one = sentence_distortion_delete_one(words)
    print(delete_one)
    base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
    glove = GloveData(base_datapath)
    insert_one = sentence_distortion_insert_one(words, glove.dict.keys())
    print(insert_one)
    random_one = sentence_distortion_shuffle_random(words)
    print(random_one)