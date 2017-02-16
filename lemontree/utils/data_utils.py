"""
This code includes useful functions for data managements.
Functions are not data dependent.
"""

import numpy as np


def int_to_onehot(input, total_class):
    """
    This function converts integer label data to one-hot encoded matrix.

    Parameters
    ----------
    input: vector
        a numpy vector that only contains the label of class.
    total_class: integer
        a integer value which indicates total classes.
        output dimension will be this.

    Returns
    -------
    ndarray
        a numpy matrix of (batch size, total class).
    """
    # check asserts
    assert isinstance(total_class, int), '"total_class" should be positive integer, number of classes.'
    n_input = len(input)
    result = np.zeros((n_input, total_class)).astype('int32')  # create output
    result[np.arange(n_input), input] = 1  # set 1 for each label, each data
    return result


def split_data(input, rule=0.9):
    """
    This function split inputs to two groups.
    The order of input data (or label) is unchanged.
    Split starts from the start, i.e., 0.9 / 0.1, not 0.1 / 0.9.

    Parameters
    ----------
    input: n-dim nparray or list
        an array or a list of anything.
    rule: float, int
        rule to split.
        if 0 < rule < 1, split by ratio. if rule > 1, split by numbers.

    Returns
    -------
    tuple
        a tuple of two splitted parts.
    """
    # check asserts
    assert 0 < rule, 'Rule should be a positive value.'
    num_input = len(input)  # also works for ndarray.

    # divide
    if rule < 1:
        num_first = np.floor(num_input * rule).astype('int32')
        num_second = num_input - num_first
        print('Splitted to:', num_first, 'and', num_second)
        first_input = input[:num_first]
        second_input = input[num_first:]
    elif rule >= 1:
        assert rule < num_input, 'Rule cannot be bigger than the number of inputs.'
        assert isinstance(rule, int), 'If rule > 1, rule should be an integer.'
        print('Splitted to:', rule, 'and', num_input - rule)
        first_input = input[:rule]
        second_input = input[rule:]
    else:
        raise ValueError('Invalid input "rule".')
    return first_input, second_input


def sentence_padding(sentences, pad_length):
    """
    This function add padding to all sentences.
    Assert all sentences length is equal or smaller than pad_length.
    Returns mask, too.

    Parameters
    ----------
    sentences: list
        a list of list of sentence words or word indices.
        i.e,. [[1,3,4],[0,5,7,9,6]]
        each sentences should be no longer than pad_length.
    pad_length: int
        an integer of max length.
        sentence of length smaller are padded by zero.
        sentence of length larger is cut-off.
    Returns
    -------
    tuple
        a tuple of two equal-shape list. (n_data, pad_length)
    """
    # check asserts
    assert isinstance(sentences, list), '"sentences" should be a list.'
    assert isinstance(pad_length, int), '"pad_length" should be an integer.'

    # do
    new = []
    mask = []
    for sen in sentences:
        if len(sen) > pad_length:
            new.append(sen[:pad_length])
            mask.append([1] * pad_length)
        else:
            new.append(sen + [0] * (pad_length - len(sen)))
            mask.append([1] * len(sen) + [0] * (pad_length - len(sen)))

    print(len(sentences), 'sentences are padded to length', pad_length)

    return new, mask


def sentence_bucketing(sentences, buckets=[20,40,60,80,100]):
    """
    This function classifies sentences to buckets.
    All sentence length is padded to length in buckets.

    Parameters
    ----------
    sentences: list
        a list of list of sentence words or word indices.
        i.e,. [[1,3,4], [0,5,7,9,6]]
    buckets: list
        a list of pad_length for each buckets.
        sentences are sorted to proper buckets.
        if sentence is too long than the longest bucket, it is cut-off.

    Returns
    -------
    tuple
        a tuple of two equal-shape list. (n_bucket, n_data, pad_length)
    """
    # check asserts
    assert isinstance(sentences, list), '"sentences" should be a list.'
    assert isinstance(buckets, list), '"buckets" should be a list of integers.'

    # do
    buckets = sorted(buckets)
    largest_bucket = buckets[-1]
    num_bucket = len(buckets)
    new = [[] for i in range(num_bucket)]
    mask = [[] for i in range(num_bucket)]
    
    for sen in sentences:
        for i in range(num_bucket):
            if len(sen) > largest_bucket:
                new[-1].append(sen)
                break
            elif len(sen) <= buckets[i]:
                new[i].append(sen)
                break

    for i in range(num_bucket):
        bucket_new, bucket_mask = sentence_padding(new[i], buckets[i])
        new[i] = bucket_new
        mask[i] = bucket_mask

    return new, mask

if __name__ == '__main__':
    sentence = [[1,2,3,4], [5,6,7,8,9,0], [3,5,7], [8,9]]
    #pad_sentence, pad_mask = sentence_padding(sentence, 4)
    pad_sentence, pad_mask = sentence_bucketing(sentence, [2,4,8])
    print(pad_sentence)
    print(pad_mask)