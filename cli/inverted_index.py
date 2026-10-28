import json
import math
import os
import pickle
import string
from collections import Counter
from functools import lru_cache

import snowballstemmer

try:
    from constants import BM25_B, BM25_K1, CACHE_DIR
    from lib.config import (
        DOC_LENGTHS_PATH, DOCMAP_PATH, INDEX_PATH, MOVIES_PATH,
        STOPWORDS_PATH, TERM_FREQUENCIES_PATH, ensure_cache_dir,
    )
except ModuleNotFoundError:
    from .constants import BM25_B, BM25_K1, CACHE_DIR
    from .lib.config import (
        DOC_LENGTHS_PATH, DOCMAP_PATH, INDEX_PATH, MOVIES_PATH,
        STOPWORDS_PATH, TERM_FREQUENCIES_PATH, ensure_cache_dir,
    )


stemmer = snowballstemmer.stemmer("english")
translation_table = str.maketrans("", "", string.punctuation)


def load_movies() -> list[dict]:
    with MOVIES_PATH.open(encoding="utf-8") as file:
        return json.load(file)["movies"]


def load_stop_words() -> list[str]:
    with STOPWORDS_PATH.open(encoding="utf-8") as file:
        return [
            word.lower().translate(translation_table)
            for word in file.read().splitlines()
        ]


stop_words = set(load_stop_words())


@lru_cache(maxsize=65536)
def stem_word(token: str) -> str:
    return stemmer.stemWord(token)


def tokenize_text(text: str) -> list[str]:
    tokens = text.lower().translate(translation_table).split()
    return [stem_word(token) for token in tokens if token not in stop_words]


def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must tokenize to exactly one token")
    return tokens[0]


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.doc_lengths_path = str(DOC_LENGTHS_PATH)

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_documents(self, term):
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def get_bm25_idf(self, term: str) -> float:
        document_count = len(self.docmap)
        document_frequency = len(self.get_documents(term))
        return math.log(
            (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
            + 1
        )

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        term_frequency = self.get_tf(doc_id, term)
        average_doc_length = self.__get_avg_doc_length()
        if average_doc_length == 0:
            length_normalization = 1.0
        else:
            length_normalization = 1 - b + b * (
                self.doc_lengths.get(doc_id, 0) / average_doc_length
            )
        return (term_frequency * (k1 + 1)) / (
            term_frequency + k1 * length_normalization
        )

    def bm25(self, doc_id, term):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query, limit):
        query_tokens = tokenize_text(query)
        scores = {doc_id: 0.0 for doc_id in self.docmap}
        average_length = self.__get_avg_doc_length()
        # Corpus-wide statistics are constant throughout a search. Visit only
        # postings that contain a query term, rather than every movie/term pair.
        for token in query_tokens:
            idf = self.get_bm25_idf(token)
            for doc_id in self.index.get(token, ()):
                frequency = self.get_tf(doc_id, token)
                normalization = (1 - BM25_B + BM25_B *
                                 self.doc_lengths.get(doc_id, 0) / average_length
                                 if average_length else 1.0)
                scores[doc_id] += idf * frequency * (BM25_K1 + 1) / (
                    frequency + BM25_K1 * normalization)

        ranked_documents = sorted(
            ((doc_id, score) for doc_id, score in scores.items() if score > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
        return [(self.docmap[doc_id], score) for doc_id, score in ranked_documents]

    def build(self):
        self.index.clear()
        self.docmap.clear()
        self.term_frequencies.clear()
        self.doc_lengths.clear()
        for movie in load_movies():
            self.docmap[movie["id"]] = movie
            self.__add_document(
                movie["id"], f"{movie['title']} {movie['description']}"
            )

    def save(self):
        ensure_cache_dir()
        with INDEX_PATH.open("wb") as file:
            pickle.dump(self.index, file)
        with DOCMAP_PATH.open("wb") as file:
            pickle.dump(self.docmap, file)
        with TERM_FREQUENCIES_PATH.open("wb") as file:
            pickle.dump(self.term_frequencies, file)
        with DOC_LENGTHS_PATH.open("wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        with INDEX_PATH.open("rb") as file:
            self.index = pickle.load(file)
        with DOCMAP_PATH.open("rb") as file:
            self.docmap = pickle.load(file)
        with TERM_FREQUENCIES_PATH.open("rb") as file:
            self.term_frequencies = pickle.load(file)
        with DOC_LENGTHS_PATH.open("rb") as file:
            self.doc_lengths = pickle.load(file)
