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


def generate_cited_answer(query: str, results: list[dict]) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    documents = "\n".join(
        f"[{index}] {result['document']['title']}: "
        f"{result['document']['description'][:1000]}"
        for index, result in enumerate(results, start=1)
    )
    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=60.0
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def answer_question(question: str, results: list[dict]) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    context = "\n".join(
        f"{result['document']['title']}: "
        f"{result['document']['description'][:1000]}"
        for result in results
    )
    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
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
    citations_parser = subparsers.add_parser(
        "citations", help="Answer a query with source citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=5)
    question_parser = subparsers.add_parser(
        "question", help="Answer a question about movies"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, default=5)

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
        case "citations":
            results = HybridSearch(load_movies()).rrf_search(
                args.query, 60, args.limit
            )[: args.limit]
            print("Search Results:")
            for result in results:
                print(f"- {result['document']['title']}")
            print("\nLLM Answer:")
            print(generate_cited_answer(args.query, results))
        case "question":
            results = HybridSearch(load_movies()).rrf_search(
                args.question, 60, args.limit
            )[: args.limit]
            print("Search Results:")
            for result in results:
                print(f"- {result['document']['title']}")
            print("\nAnswer:")
            print(answer_question(args.question, results))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
