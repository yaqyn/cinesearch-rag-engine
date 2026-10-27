"""Prepare the ignored dataset and keyword artifacts for the launcher."""

import json
from pathlib import Path
import sys
import tempfile
import urllib.request

from cli.lib.config import (
    MOVIES_PATH, INDEX_PATH, DOCMAP_PATH, TERM_FREQUENCIES_PATH, DOC_LENGTHS_PATH,
)

# Fixed object generation keeps fresh installs on the same Boot.dev corpus.
DATASET_URL = (
    "https://storage.googleapis.com/qvault-webapp-dynamic-assets/"
    "course_assets/movies.json?generation=1759751668026541"
)


def download_dataset(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        with temporary.open(encoding="utf-8") as source:
            payload = json.load(source)
        movies = payload.get("movies") if isinstance(payload, dict) else None
        if not isinstance(movies, list) or not movies or any(
            not isinstance(movie, dict)
            or not isinstance(movie.get("id"), int)
            or not isinstance(movie.get("title"), str)
            or not movie.get("title", "").strip()
            or not isinstance(movie.get("description"), str)
            for movie in movies
        ):
            raise ValueError("Downloaded dataset must contain movie IDs, titles, and descriptions.")
        movie_ids = [movie["id"] for movie in movies]
        if len(movie_ids) != len(set(movie_ids)):
            raise ValueError("Downloaded dataset must contain unique movie IDs.")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    downloaded = not MOVIES_PATH.exists()
    if downloaded:
        print("Downloading the Boot.dev movie catalog…", file=sys.stderr)
        download_dataset(MOVIES_PATH)
    artifacts = (INDEX_PATH, DOCMAP_PATH, TERM_FREQUENCIES_PATH, DOC_LENGTHS_PATH)
    if downloaded or not all(path.exists() for path in artifacts):
        from cli.inverted_index import InvertedIndex
        print("Building the local keyword index (first launch)…", file=sys.stderr)
        search = InvertedIndex()
        search.build()
        search.save()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"CineSearch setup failed: {exc}\nCheck connectivity and data/movies.json, then run ./launch again.", file=sys.stderr)
        raise SystemExit(1)
