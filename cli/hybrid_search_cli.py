import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.semantic_search import load_movies


LLM_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "liquid/lfm-2.5-2.6b:free"
)


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
        model=LLM_MODEL,
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
        model=LLM_MODEL,
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


def expand_query(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": f'''Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

User query: "{query}"''',
            }
        ],
    )
    return response.choices[0].message.content.strip()


def rerank_document(query: str, result: dict) -> float:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    document = result["document"]
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=30.0
    )
    prompt = f'''Rate how well this movie matches the search query.

Query: "{query}"
Movie: {document.get("title", "")} - {document.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:'''
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            score = float(response.choices[0].message.content.strip())
            if 0 <= score <= 10:
                return score
        except Exception:
            if attempt == 2:
                raise
        time.sleep(3)
    raise RuntimeError("Unable to get a valid rerank score")


def rerank_batch(query: str, results: list[dict]) -> list[dict]:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    doc_list = "\n".join(
        f"ID: {result['document']['id']} - {result['document']['title']}: "
        f"{result['document']['description'][:300]}"
        for result in results
    )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=30.0
    )
    prompt = f'''Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

Ranking:'''
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            ranked_ids = json.loads(response.choices[0].message.content.strip())
            ranked_ids = [int(document_id) for document_id in ranked_ids]
            rank_map = {document_id: rank for rank, document_id in enumerate(ranked_ids, 1)}
            for result in results:
                result["rerank_rank"] = rank_map.get(
                    result["document"]["id"], len(results) + 1
                )
            return sorted(results, key=lambda result: result["rerank_rank"])
        except Exception:
            if attempt == 2:
                raise
        time.sleep(3)
    raise RuntimeError("Unable to get valid batch rerank results")


def rerank_cross_encoder(query: str, results: list[dict]) -> list[dict]:
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    pairs = [
        (
            query,
            f"{result['document']['title']}: "
            f"{result['document']['description'][:500]}",
        )
        for result in results
    ]
    scores = model.predict(pairs, show_progress_bar=True)
    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)
    return sorted(results, key=lambda result: result["rerank_score"], reverse=True)


def evaluate_results(query: str, results: list[dict]) -> list[int]:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    formatted_results = [
        f"{result['document']['title']}: "
        f"{result['document']['description'][:200]}"
        for result in results
    ]
    prompt = f'''Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:
[2, 0, 3, 2, 0, 1]'''
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=30.0
    )
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            scores = json.loads(response.choices[0].message.content.strip())
            if len(scores) != len(results) or any(score not in (0, 1, 2, 3) for score in scores):
                raise ValueError("Invalid evaluation scores")
            return scores
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)
    raise RuntimeError("Unable to evaluate search results")


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
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method",
    )
    rrf_parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate search results with an LLM"
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
            if args.enhance in ("spell", "rewrite", "expand"):
                if args.enhance == "spell":
                    enhanced_query = enhance_query_with_spelling(query)
                elif args.enhance == "rewrite":
                    enhanced_query = rewrite_query(query)
                else:
                    expansion = expand_query(query)
                    enhanced_query = f"{query} {expansion}"
                print(
                    f"Enhanced query ({args.enhance}): "
                    f"'{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query
            hybrid_search = HybridSearch(load_movies())
            results = hybrid_search.rrf_search(query, args.k, args.limit)
            if args.rerank_method:
                results = results[: args.limit * 5]
            if args.rerank_method:
                print(
                    f"Re-ranking top {args.limit} results using "
                    f"{args.rerank_method} method..."
                )
                if args.rerank_method == "individual":
                    for start in range(0, len(results), 5):
                        group = results[start : start + 5]
                        with ThreadPoolExecutor(max_workers=5) as executor:
                            scores = list(
                                executor.map(
                                    lambda result: rerank_document(query, result),
                                    group,
                                )
                            )
                        for result, score in zip(group, scores):
                            result["rerank_score"] = score
                        if start + 5 < len(results):
                            time.sleep(3)
                    results.sort(
                        key=lambda result: result["rerank_score"], reverse=True
                    )
                elif args.rerank_method == "batch":
                    results = rerank_batch(query, results)
                else:
                    results = rerank_cross_encoder(query, results)
            results = results[: args.limit]
            print(
                f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):"
            )
            for i, result in enumerate(results, start=1):
                document = result["document"]
                print(f"{i}. {document['title']}")
                if args.rerank_method == "individual":
                    print(f"   Re-rank Score: {result['rerank_score']:.3f}/10")
                elif args.rerank_method == "cross_encoder":
                    print(f"   Re-rank Score: {result['rerank_score']:.3f}")
                elif args.rerank_method == "batch":
                    print(f"   Re-rank Rank: {result['rerank_rank']}")
                print(f"  RRF Score: {result['rrf']:.3f}")
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, "
                    f"Semantic Rank: {result['semantic_rank']}"
                )
                print(f"   {document['description'][:100]}...")
            if args.evaluate:
                scores = evaluate_results(query, results)
                print()
                for i, (result, score) in enumerate(zip(results, scores), start=1):
                    print(f"{i}. {result['document']['title']}: {score}/3")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
