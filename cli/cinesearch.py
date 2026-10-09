"""Unified CineSearch entrypoint for the most useful product workflows."""

import argparse
import json

try:
    from lib.config import CACHE_DIR, INDEX_PATH
    from lib.hybrid_search import HybridSearch
    from lib.semantic_search import ChunkedSemanticSearch
    from lib.semantic_search import load_movies
    from inverted_index import InvertedIndex
except ModuleNotFoundError:
    from .lib.config import CACHE_DIR, INDEX_PATH
    from .lib.hybrid_search import HybridSearch
    from .lib.semantic_search import ChunkedSemanticSearch, load_movies
    from .inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="CineSearch movie retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search", help="Search the movie catalog")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument(
        "--mode", choices=("hybrid", "keyword", "semantic"), default="hybrid"
    )
    search_parser.add_argument(
        "--json", action="store_true", help="Emit results as JSON"
    )
    health_parser = subparsers.add_parser("health", help="Inspect local search artifacts")
    health_parser.set_defaults(command="health")
    args = parser.parse_args()

    if args.command == "health":
        artifacts = sorted(path.name for path in CACHE_DIR.glob("*")) if CACHE_DIR.exists() else []
        print(f"Cache directory: {CACHE_DIR}")
        print(f"Artifacts: {', '.join(artifacts) if artifacts else 'none'}")
        return

    documents = load_movies()
    if args.mode == "hybrid":
        search = HybridSearch(documents)
        results = search.rrf_search(args.query, k=60, limit=args.limit)[: args.limit]
        if args.json:
            print(json.dumps(results, indent=2, default=str))
            return
        for index, result in enumerate(results, 1):
            document = result["document"]
            print(f"{index}. {document['title']} (score: {result['rrf']:.4f})")
            print(f"   {document['description'][:160]}...")
    elif args.mode == "keyword":
        search = InvertedIndex()
        if not INDEX_PATH.exists():
            search.build()
            search.save()
        else:
            search.load()
        results = [
            {"title": document["title"], "score": score, "document": document["description"]}
            for document, score in search.bm25_search(args.query, args.limit)
        ]
        if args.json:
            print(json.dumps(results, indent=2))
            return
        for index, (document, score) in enumerate(
            search.bm25_search(args.query, args.limit), 1
        ):
            print(f"{index}. {document['title']} (score: {score:.4f})")
            print(f"   {document['description'][:160]}...")
    else:
        search = ChunkedSemanticSearch()
        search.load_or_create_chunk_embeddings(documents)
        results = search.search_chunks(args.query, args.limit)
        if args.json:
            print(json.dumps(results, indent=2, default=str))
            return
        for index, result in enumerate(
            results, 1
        ):
            print(f"{index}. {result['title']} (score: {result['score']:.4f})")
            print(f"   {result['document']}...")


if __name__ == "__main__":
    main()
