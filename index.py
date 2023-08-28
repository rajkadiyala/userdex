'''Utilities for indexing documents.'''

import re
import string

import nltk

_PUNCTUATION = re.compile(f'[{re.escape(string.punctuation)}]')
_STEMMER = nltk.stem.PorterStemmer()
_STOP_WORDS = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                   'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
                   'do', 'at', 'this', 'but', 'his', 'by', 'from', 'wikipedia'])


class _Tokenizer:

    def tokenize(self, text: str) -> list[str]:
        '''Converts text into distinct words for indexing'''
        tokens = self._tokenize(text)
        tokens = self._lowercase(tokens)
        tokens = self._remove_punctuation(tokens)
        tokens = self._remove_common_words(tokens)
        tokens = self._consolidate_to_root_word(tokens)
        return list(set(tokens))

    def _tokenize(self, text: str) -> list[str]:
        return nltk.tokenize.word_tokenize(text)

    def _lowercase(self, tokens: list[str]) -> list[str]:
        return [token.lower() for token in tokens]

    def _consolidate_to_root_word(self, tokens: list[str]) -> list[str]:
        return [_STEMMER.stem(token) for token in tokens]

    def _remove_punctuation(self, tokens: list[str]) -> list[str]:
        return [_PUNCTUATION.sub('', token) for token in tokens]

    def _remove_common_words(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token not in _STOP_WORDS]
