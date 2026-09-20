import argparse

from lib.hybrid_search import normalize_scores
from lib.hybrid_search import HybridSearch
from lib.semantic_search import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument("scores", type=float, nargs="*")
    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Search using weighted hybrid scoring"
    )
    weighted_parser.add_argument("query", type=str)
    weighted_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize_scores(args.scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            hybrid_search = HybridSearch(load_movies())
            results = hybrid_search.weighted_search(
                args.query, args.alpha, args.limit
            )[: args.limit]
            for i, result in enumerate(results, start=1):
                document = result["document"]
                print(f"{i}. {document['title']}")
                print(f"  Hybrid Score: {result['hybrid']:.3f}")
                print(
                    f"  BM25: {result['bm25']:.3f}, "
                    f"Semantic: {result['semantic']:.3f}"
                )
                print(f"  {document['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
