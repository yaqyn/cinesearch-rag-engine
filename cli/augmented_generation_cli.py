import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

from lib.hybrid_search import HybridSearch
from lib.semantic_search import load_movies


LLM_MODEL = os.environ.get("OPENROUTER_MODEL", "liquid/lfm-2.5-2.6b:free")


def generate_answer(query: str, results: list[dict]) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    docs = "\n".join(
        f"{result['document']['title']}: "
        f"{result['document']['description'][:1000]}"
        for result in results
    )
    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=60.0
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def generate_summary(query: str, results: list[dict]) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    formatted_results = "\n".join(
        f"{result['document']['title']}: "
        f"{result['document']['description'][:1000]}"
        for result in results
    )
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{formatted_results}

Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=60.0
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "rag":
            results = HybridSearch(load_movies()).rrf_search(args.query, 60, 5)[:5]
            print("Search Results:")
            for result in results:
                print(f"- {result['document']['title']}")
            print("\nRAG Response:")
            print(generate_answer(args.query, results))
        case "summarize":
            results = HybridSearch(load_movies()).rrf_search(
                args.query, 60, args.limit
            )[: args.limit]
            print("Search Results:")
            for result in results:
                print(f"- {result['document']['title']}")
            print("\nLLM Summary:")
            print(generate_summary(args.query, results))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
