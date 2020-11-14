"""Creating word2vec type of algorithm from scratch."""

__author__ = 'simone carolini'

import numpy as np
import pandas as pd
from typing import Type, List, Union

from utils.util import text_processing

# todo! this is just a test corpus. Replace with a bunch of well written text from db.
text_corpus = ['simone carolini is a man, simona carolini is a woman, he is a man, he is a boy, girl is a woman',
               'transferwise is a company. transferwise is not a man. transferwise is not a woman']


def create_context_values(input_corpus: Type[List] = None, window_length: int = 2) -> Type[Union[List, List]]:
    corpus_processed, words_list = [], []
    for text in input_corpus:
        text = text_processing(text)  # Clean the data.
        print(f'Cleaned text: {text}')
        corpus_processed.append(text)  # Append words.
        # Context words.
        for i, word in enumerate(text):
            for w in range(window_length):
                # Ahead of focus word.
                if i + 1 + w < len(text):
                    words_list.append([word] + [text[(i + 1 + w)]])
                if i - 1 - w >= 0:
                    words_list.append([word] + [text[(i - 1 - w)]])
    return corpus_processed, words_list


def create_word_index(corpus_processed: Type[List]):
    """
    Create a dictionary with word and its index.

    :param
    ------
    corpus_processed : list
        List with all the words.

    :return:

    """

    print('Corupus processed: ', corpus_processed)
    tmp_words = []
    for sentence in corpus_processed:
        tmp_words += sentence  # It is the same as using .extend function in pandas.
    print('Words : ', tmp_words)
    words = list(set(tmp_words))
    words.sort(reverse=False)
    print(words)
    word_index = {w: i for i, w in enumerate(words)}
    return word_index


corpus, word_list = create_context_values(text_corpus)
word_index = create_word_index(corpus)
print(word_index)
