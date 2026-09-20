import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.semantic_search import load_movies


def enhance_query_with_spelling(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f'''Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"''',
            }
        ],
    )
    return response.choices[0].message.content.strip()


def rewrite_query(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f'''Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"''',
            }
        ],
    )
    return response.choices[0].message.content.strip()


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
    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Search using reciprocal rank fusion"
    )
    rrf_parser.add_argument("query", type=str)
    rrf_parser.add_argument("-k", type=int, default=60)
    rrf_parser.add_argument("--limit", type=int, default=5)
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="Query enhancement method",
    )

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
        case "rrf-search":
            query = args.query
            if args.enhance in ("spell", "rewrite"):
                if args.enhance == "spell":
                    enhanced_query = enhance_query_with_spelling(query)
                else:
                    enhanced_query = rewrite_query(query)
                print(
                    f"Enhanced query ({args.enhance}): "
                    f"'{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query
            hybrid_search = HybridSearch(load_movies())
            results = hybrid_search.rrf_search(query, args.k, args.limit)[
                : args.limit
            ]
            for i, result in enumerate(results, start=1):
                document = result["document"]
                print(f"{i}. {document['title']}")
                print(f"  RRF Score: {result['rrf']:.3f}")
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, "
                    f"Semantic Rank: {result['semantic_rank']}"
                )
                print(f"  {document['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
