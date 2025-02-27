'''Utilities for indexing documents.'''

import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Optional, TypeAlias

import nltk

Index: TypeAlias = dict[str, list[int]]

_PUNCTUATION = re.compile(f'[{re.escape(string.punctuation)}]')
_STEMMER = nltk.stem.PorterStemmer()
_STOP_WORDS = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                   'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
                   'do', 'at', 'this', 'but', 'his', 'by', 'from', 'wikipedia'])


class _Tokenizer:

    def tokenize(self, text: str) -> list[str]:
        """Converts text into distinct words for indexing."""
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


class DocumentIndexer:
    """Utility class for indexing documents."""

    def index_by_page_number(
            self, document: list[str],
            persist_filepath: Optional[Path] = None) -> Index:
        """Creates a page number index for various words within a document."""
        page_index: Index = defaultdict(list)
        for page_number, page in enumerate(document):
            self._index_page(page, page_number, page_index)
        self._persist_index_if_requested(page_index, persist_filepath)
        return page_index

    def _index_page(
            self, page: str, page_number: int, page_index: Index) -> None:
        for token in _Tokenizer().tokenize(page):
            page_index[token].append(page_number)

    def _persist_index_if_requested(
            self, page_index: Index, filepath: Optional[Path]) -> None:
        if filepath is None:
            return
        with open(filepath, 'w', encoding='utf-8') as persistent_index:
            persistent_index.write(page_index)
