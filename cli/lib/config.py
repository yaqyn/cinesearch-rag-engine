"""Project paths shared by the search backends."""

from pathlib import Path
import hashlib
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
MOVIES_PATH = DATA_DIR / "movies.json"
STOPWORDS_PATH = DATA_DIR / "stopwords.txt"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.json"
MOVIE_EMBEDDINGS_PATH = CACHE_DIR / "movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = CACHE_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = CACHE_DIR / "chunk_metadata.json"
INDEX_PATH = CACHE_DIR / "index.pkl"
DOCMAP_PATH = CACHE_DIR / "docmap.pkl"
TERM_FREQUENCIES_PATH = CACHE_DIR / "term_frequencies.pkl"
DOC_LENGTHS_PATH = CACHE_DIR / "doc_lengths.pkl"
EMBEDDING_MANIFEST_PATH = CACHE_DIR / "embedding_manifest.json"


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def documents_signature(documents: list[dict]) -> str:
    payload = [
        {"id": doc["id"], "title": doc["title"], "description": doc["description"]}
        for doc in documents
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
