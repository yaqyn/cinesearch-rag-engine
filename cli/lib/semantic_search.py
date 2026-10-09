import json
import os
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    from lib.config import (
        CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH, MOVIE_EMBEDDINGS_PATH,
        MOVIES_PATH, ensure_cache_dir,
    )
except ModuleNotFoundError:
    from .config import (
        CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH, MOVIE_EMBEDDINGS_PATH,
        MOVIES_PATH, ensure_cache_dir,
    )

try:
    from lib.search_utils import format_search_result
except ModuleNotFoundError:
    from .search_utils import format_search_result


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def load_movies() -> list[dict]:
    with MOVIES_PATH.open(encoding="utf-8") as file:
        return json.load(file)["movies"]


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk size")

    words = text.split()
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def semantic_chunk_text(
    text: str, max_chunk_size: int, overlap: int = 0
) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError("max chunk size must be greater than zero")
    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError("overlap must be non-negative and smaller than max chunk size")

    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    chunks = []
    start = 0
    step = max_chunk_size - overlap
    while start < len(sentences):
        chunks.append(" ".join(sentences[start : start + max_chunk_size]))
        if start + max_chunk_size >= len(sentences):
            break
        start += step
    return chunks


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name, device="cpu")
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
        ensure_cache_dir()
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {document["id"]: document for document in documents}
        if MOVIE_EMBEDDINGS_PATH.exists():
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        scored_documents = [
            (cosine_similarity(query_embedding, embedding), document)
            for embedding, document in zip(self.embeddings, self.documents)
        ]
        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "score": score,
                "title": document["title"],
                "description": document["description"],
            }
            for score, document in scored_documents[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {document["id"]: document for document in documents}
        all_chunks = []
        chunk_metadata = []

        for movie_idx, document in enumerate(documents):
            description = document["description"]
            if not description.strip():
                continue
            chunks = semantic_chunk_text(description, 4, 1)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )

        torch.set_num_threads(os.cpu_count() or 1)
        self.chunk_embeddings = self.model.encode(
            all_chunks, batch_size=256, show_progress_bar=True
        )
        self.chunk_metadata = chunk_metadata
        ensure_cache_dir()
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with CHUNK_METADATA_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                file,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {document["id"]: document for document in documents}
        if CHUNK_EMBEDDINGS_PATH.exists() and CHUNK_METADATA_PATH.exists():
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with CHUNK_METADATA_PATH.open(encoding="utf-8") as file:
                self.chunk_metadata = json.load(file)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for metadata, embedding in zip(self.chunk_metadata, self.chunk_embeddings):
            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": cosine_similarity(query_embedding, embedding),
                }
            )

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]["score"]
            ):
                movie_scores[movie_idx] = chunk_score

        ranked_movies = sorted(
            movie_scores.values(), key=lambda result: result["score"], reverse=True
        )[:limit]
        results = []
        for movie_score in ranked_movies:
            document = self.documents[movie_score["movie_idx"]]
            results.append(
                format_search_result(
                    document["id"],
                    document["title"],
                    document["description"],
                    movie_score["score"],
                    movie_score,
                )
            )
        return results


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


def embed_query_text(query) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings() -> None:
    documents = load_movies()
    semantic_search = SemanticSearch()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors "
        f"in {embeddings.shape[1]} dimensions"
    )
