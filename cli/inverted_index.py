import json
import os
import pickle
import string

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


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        for token in tokenize_text(text):
            self.index.setdefault(token, set()).add(doc_id)

    def get_documents(self, term):
        return sorted(self.index.get(term, set()))

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

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)
        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)
