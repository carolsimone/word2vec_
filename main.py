"""Creating word2vec type of algorithm from scratch."""

__author__ = 'simone carolini'

import numpy as np
import os
from scipy import sparse
from typing import Type, List, Union, Dict
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from utils.util import text_processing
from utils.util import unit_test_true


# todo! this is just a test corpus.
text_corpus = ['simone carolini is a man, simona is a woman, he is a man, he is a boy, girl is a woman, she is a girl.',
               'Transferwise is a company. Transferwise is not a man. Transferwise is not a woman']


def create_context_values(input_corpus: Type[List] = None, window_length: int = 2) -> Type[Union[List, List]]:
    corpus_processed, words_list = [], []
    for text in input_corpus:
        text = text_processing(text)  # Clean the data.
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

    tmp_words = []
    for sentence in corpus_processed:
        tmp_words += sentence  # It is the same as using .extend function in pandas.
    words = list(set(tmp_words))
    words.sort(reverse=False)
    word_index = {w: i for i, w in enumerate(words)}
    return word_index


def plot_word2vec_representation(embeddings_dict: Type[Dict], w: Type[List]):
    x_1 = [i[0] for i in embeddings_dict.values()]
    x_2 = [i[1] for i in embeddings_dict.values()]

    ax = sns.scatterplot(x=x_1, y=x_2)

    for w, x, y in zip(w, x_1, x_2):
        ax.text(x+.02, y, w)


if __name__ == '__main__':
    corpus, word_lists = create_context_values(text_corpus)
    word_index = create_word_index(corpus)

    print(word_lists)
    n_words = len(word_index)  # features/columns in your matrix.
    print(f'number of unique words: ', n_words)

    X, Y = [], []

    for i, word_list in enumerate(word_lists):
        main_word_index = word_index.get(word_list[0])  # Get the index of your focus word.
        context_word_index = word_index.get(word_list[1])  # Get the index of the context word
        print(main_word_index, context_word_index)
        print(word_index)
        # Define the 1 x n array. n = number of features.
        X_row = np.zeros(n_words)
        Y_row = np.zeros(n_words)
        # Replace with 1 the words that are present
        X_row[main_word_index] = 1
        Y_row[context_word_index] = 1
        # Append for all consecutives word_list items.
        X.append(X_row)
        Y.append(Y_row)
    # Store as spare matrices:
    # https://towardsdatascience.com/why-we-use-sparse-matrices-for-recommender-systems-2ccc9ab698a4
    # In jupyter notebook you may have to use np.asarray(X) etc.
    X = sparse.csr_matrix(X)
    Y = sparse.csr_matrix(Y)

    unit_test_true(X, Y)  # Check that the two matrices are the same length.

    embeddings_size = 2
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Odd issue with my env: https://github.com/openai/spinningup/issues/16
    # Deep Learning phase.
    focus_m = Input(shape=(X.shape[1],))
    context_m = Dense(units=embeddings_size, activation='linear')(focus_m)
    context_m = Dense(units=Y.shape[1], activation='softmax')(context_m)
    model = Model(inputs=focus_m, outputs=context_m)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(x=X, y=Y, batch_size=256, epochs=1000)
    weights = model.get_weights()[0]  # Weights from neural network.
    print(weights)

    words = list(word_index.keys())  # transform dict to list.
    # print(words)
    embeddings_dict = dict()
    for word in words:
        embeddings_dict.update({word: weights[word_index.get(word)]})
    # word - [[embedding, embedding]]
    print(embeddings_dict)

    # Use it in jupyter notebook.
    plot_word2vec_representation(embeddings_dict, words)