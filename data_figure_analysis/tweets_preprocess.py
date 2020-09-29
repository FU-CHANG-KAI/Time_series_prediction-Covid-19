import pandas as pd
import os
import json
#1. lang == 'en' and place in 50 states in the united states "place"
#2. preprocess the format of date "created_at"
#3 store and calculate the size of data 

basePath = os.path.dirname(os.path.abspath(__file__))
from collections import Counter

import string
import re

import pandas
import numpy
import spacy

from spellchecker import SpellChecker

from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import train_test_split

from spacy.lemmatizer import Lemmatizer

from tfn.clean import clean
from tfn.helper import _get_training_data_from_csv, _get_stop_words
from tfn import TWEETS_FILE_CSV_MERGE

from transformers import BertTokenizer


_TRAIN_DATA_PATH = TWEETS_FILE_CSV_MERGE
_EMOJI_SEQUENCE = ' xx90'

en = spacy.load('en_core_web_sm')
lemmatize = en.Defaults.create_lemmatizer()

START_SPEC_CHARS = re.compile('^[{}]+'.format(re.escape(string.punctuation)))
END_SPEC_CHARS = re.compile('[{}]+$'.format(re.escape(string.punctuation)))


spell = SpellChecker(distance=1)
def check_spelling(tokens, keep_wrong=False):
    if keep_wrong:
        length_original = len(tokens)
        tokens += [
            spell.correction(token) for token in tokens
            if not spell.correction(token) in [
                token for token in tokens
            ]
        ]
        return tokens, len(tokens) - length_original

    elif not keep_wrong:
        corrections = [
            (token, spell.correction(token)) for token in tokens
            if not token == spell.correction(token)
        ]
        for correction in corrections:
            tokens.remove(correction[0])
            tokens.append(correction[1])

        return tokens, len(corrections)

def _has_digits(token):
    ''' Returns true if the given string contains any digits '''
    return any(char.isdigit() for char in token)


class Dataset():
    def __init__(self, tokenizer, strip_handles=True, 
                                  strip_rt=True, 
                                  strip_digits=True,
                                  strip_hashtags=False,
                                  test_size=0.1,
                                  shuffle=False):

        # Get raw data
        self.corpus = pandas.read_csv(_TRAIN_DATA_PATH)

        if tokenizer == 'twitter':
            self.X = self._tokenize(self.corpus, 
                                    strip_handles, 
                                    strip_rt, 
                                    strip_digits,
                                    strip_hashtags)
        elif tokenizer == 'lemmatize':
            self.corpus = self.corpus.apply(self._tokenize_with_lemma(doc, 
                                               strip_handles, 
                                               strip_rt,
                                               strip_digits))

    def _tokenize_with_lemma(self, doc, strip_handles=True, strip_rt=True, strip_digits=True):
        ''' Tokenize and lemmatize using Spacy '''
        
        stop_words = _get_stop_words(strip_handles, strip_rt)

        # Add special sequence for emojis (??). Needs to be done before any
        # punctuation removal or tokenization
        doc = doc.replace('??', _EMOJI_SEQUENCE)

        # Applies cleaning from clean.py
        #doc = clean(doc)

        # Replace hashtag with <hashtag> token as is encoded in GLoVe
        # doc = doc.replace('#', '<hashtag> ')

        # Tokenize the document.
        tokens = [lemmatize(token.text, token.pos_)[0].lower() for token in en(doc)]

        # Remove punctuation tokens.
        tokens = [token for token in tokens if token not in string.punctuation+'…’']

        # Remove tokens wich contain any number.
        if strip_digits:
            tokens = [token for token in tokens if not _has_digits(token)]

        # Remove tokens without text.
        tokens = [token for token in tokens if bool(token.strip())]

        # Remove punctuation from start of tokens.
        tokens = [re.sub(START_SPEC_CHARS, '', token) for token in tokens]

        # Remove punctuation from end of tokens.
        tokens = [re.sub(END_SPEC_CHARS, '', token) for token in tokens]

        # Remove stopwords from the tokens
        tokens = [token for token in tokens if token not in stop_words]


        return tokens

if __name__ == '__main__':
   ds = Dataset('lemmatize')
   print(ds.text[:10])