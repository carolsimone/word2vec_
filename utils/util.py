"""Utility functions used on the word2vec project"""

__author__ = 'simone carolini'

import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Type, List
import unittest as unit_test


def text_processing(text: Type[List] = None) -> Type[List]:
    """
    Pre-processing some input text

    :param
    ------
    text : List
        List with all the words in the text.

    :return:
    -------
    text : list
        List containing tokens that we are going to feed to our model.
    """

    punctuation_ = string.punctuation
    stopwords_ = stopwords.words('english')
    text = text.lower().strip()
    for x in text:
        if x in punctuation_:
            text = text.replace(x, '')
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = word_tokenize(text)  # Convert words into tokens.
    text = [word for word in text if word not in stopwords_]
    return text


def unit_test_true(X, Y):
    ut = unit_test.TestCase()
    assert_condition = X.shape[0] == Y.shape[0]
    return ut.assertTrue(assert_condition, 'Matrices have different row numbers!')