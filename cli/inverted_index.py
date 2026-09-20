import json
import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer


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

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
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

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)
        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)
        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)
