"""Unified CineSearch entrypoint and interactive movie browser."""

import argparse
import contextlib
import json
import sys
import textwrap
from functools import lru_cache


@lru_cache(maxsize=1)
def _keyword_index(signature):
    """Reuse loaded artifacts until an index file changes on disk."""
    from cli.inverted_index import InvertedIndex
    from cli.lib.config import INDEX_PATH
    search = InvertedIndex()
    if INDEX_PATH.exists():
        search.load()
    else:
        search.build()
        search.save()
    return search


def search_catalog(query, mode, limit):
    # Lazy imports keep help and the browser instant, without loading PyTorch.
    from cli.lib.config import INDEX_PATH
    if mode == "keyword":
        from cli.lib.config import DOCMAP_PATH, TERM_FREQUENCIES_PATH, DOC_LENGTHS_PATH
        signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                          for path in (INDEX_PATH, DOCMAP_PATH, TERM_FREQUENCIES_PATH, DOC_LENGTHS_PATH)
                          if path.exists())
        search = _keyword_index(signature)
        return [{"title": doc["title"], "description": doc["description"], "score": score}
                for doc, score in search.bm25_search(query, limit)]
    from cli.lib.semantic_search import load_movies
    documents = load_movies()
    if mode == "hybrid":
        from cli.lib.hybrid_search import HybridSearch
        return [{"title": row["document"]["title"], "description": row["document"]["description"],
                 "score": row["rrf"], "raw": row}
                for row in HybridSearch(documents).rrf_search(query, k=60, limit=limit)[:limit]]
    from cli.lib.semantic_search import ChunkedSemanticSearch
    search = ChunkedSemanticSearch()
    search.load_or_create_chunk_embeddings(documents)
    return [{"title": row["title"], "description": row["document"], "score": row["score"], "raw": row}
            for row in search.search_chunks(query, limit)]


def positive_limit(value):
    number = int(value)
    if number < 1 or number > 100:
        raise argparse.ArgumentTypeError("Choose a result limit between 1 and 100.")
    return number


def main(argv=None) -> None:
    # Also support `python cli/cinesearch.py` from a source checkout.
    if not __package__:
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cli.terminal import THEMES, clean, error_message, launch, plain_browser
    parser = argparse.ArgumentParser(description="CineSearch · Find movies by words, scenes, and meaning")
    subparsers = parser.add_subparsers(dest="command")
    browse = subparsers.add_parser("browse", help="Open the keyboard-driven movie browser")
    browse.add_argument("--plain", action="store_true", help="Linear text interface for screen readers and scrollback")
    browse.add_argument("--theme", choices=THEMES, help="Override the saved browser theme")
    search_parser = subparsers.add_parser("search", help="Search the movie catalog")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=positive_limit, default=5)
    search_parser.add_argument("--mode", choices=("hybrid", "keyword", "semantic"), default="hybrid")
    search_parser.add_argument("--json", action="store_true", help="Emit results as JSON")
    subparsers.add_parser("health", help="Inspect local search artifacts")
    args = parser.parse_args(argv)
    if args.command is None or args.command == "browse":
        if args.command == "browse" and args.plain:
            plain_browser(search_catalog)
        elif sys.stdin.isatty() and sys.stdout.isatty():
            launch(search_catalog, getattr(args, "theme", None))
        elif args.command == "browse":
            plain_browser(search_catalog)
        else:
            parser.print_help()
        return
    if args.command == "health":
        from cli.lib.config import CACHE_DIR, MOVIES_PATH
        artifacts = sorted(path.name for path in CACHE_DIR.glob("*")) if CACHE_DIR.exists() else []
        print(f"Movie dataset: {'ready' if MOVIES_PATH.exists() else 'missing: ' + str(MOVIES_PATH)}")
        print(f"Cache directory: {CACHE_DIR}")
        print(f"Artifacts: {', '.join(artifacts) if artifacts else 'none'}")
        return
    if not args.query.strip():
        parser.error("Enter a nonempty search query.")
    try:
        # Backend progress belongs on stderr, keeping JSON parseable.
        with contextlib.redirect_stdout(sys.stderr):
            results = search_catalog(args.query, args.mode, args.limit)
        if args.json:
            rows = [row.get("raw", {"title": row["title"], "score": row["score"], "document": row["description"]}) for row in results]
            print(json.dumps(rows, indent=2, default=str))
            return
        print(f"CineSearch · {len(results)} matches · {args.mode}\n")
        if not results:
            print("No matches. Try fewer words or another --mode.")
        for index, row in enumerate(results, 1):
            print(f"{index}. {clean(row['title'])} (score: {row['score']:.4f})")
            print(textwrap.fill(clean(row['description']), width=78, initial_indent="   ", subsequent_indent="   "))
            print()
    except (OSError, ValueError, RuntimeError) as exc:
        print(error_message(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
