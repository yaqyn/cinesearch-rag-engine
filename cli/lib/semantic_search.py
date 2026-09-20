import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("text must not be empty or whitespace only")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {document["id"]: document for document in documents}
        movie_texts = [
            f"{document['title']}: {document['description']}"
            for document in documents
        ]
        self.embeddings = self.model.encode(movie_texts, show_progress_bar=True)
        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {document["id"]: document for document in documents}
        embeddings_path = "cache/movie_embeddings.npy"
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)


def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    with open("data/movies.json") as file:
        documents = json.load(file)["movies"]
    semantic_search = SemanticSearch()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors "
        f"in {embeddings.shape[1]} dimensions"
    )
