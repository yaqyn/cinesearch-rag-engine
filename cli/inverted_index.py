import json
import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer

try:
    from constants import BM25_B, BM25_K1, CACHE_DIR
except ModuleNotFoundError:
    from .constants import BM25_B, BM25_K1, CACHE_DIR


stemmer = PorterStemmer()
translation_table = str.maketrans("", "", string.punctuation)


def load_movies() -> list[dict]:
    with open("data/movies.json") as file:
        return json.load(file)["movies"]


def load_stop_words() -> list[str]:
    with open("data/stopwords.txt") as file:
        return [
            word.lower().translate(translation_table)
            for word in file.read().splitlines()
        ]


stop_words = set(load_stop_words())


def tokenize_text(text: str) -> list[str]:
    tokens = text.lower().translate(translation_table).split()
    return [stemmer.stem(token) for token in tokens if token not in stop_words]


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
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

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
        for doc_id in scores:
            for token in query_tokens:
                scores[doc_id] += self.bm25(doc_id, token)

        ranked_documents = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )[:limit]
        return [(self.docmap[doc_id], score) for doc_id, score in ranked_documents]

    def build(self):
        for movie in load_movies():
            self.docmap[movie["id"]] = movie
            self.__add_document(
                movie["id"], f"{movie['title']} {movie['description']}"
            )

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)
        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)
        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)
        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)
        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)
        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)
