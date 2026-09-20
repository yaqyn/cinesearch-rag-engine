import argparse

from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    chunk_text,
    semantic_chunk_text,
    embed_query_text,
    embed_text,
    load_movies,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the semantic search model")
    embed_parser = subparsers.add_parser("embed_text", help="Generate a text embedding")
    embed_parser.add_argument("text", type=str)
    subparsers.add_parser("verify_embeddings", help="Verify movie embeddings")
    subparsers.add_parser("embed_chunks", help="Generate chunked movie embeddings")
    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search using chunked movie embeddings"
    )
    search_chunked_parser.add_argument("query", type=str)
    search_chunked_parser.add_argument("--limit", type=int, default=5)
    query_parser = subparsers.add_parser("embed_query", help="Generate a query embedding")
    query_parser.add_argument("query", type=str)
    search_parser = subparsers.add_parser("search", help="Search movies semantically")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5)
    chunk_parser = subparsers.add_parser("chunk", help="Split text into word chunks")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=0)
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text into sentence-based chunks"
    )
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_chunks":
            documents = load_movies()
            semantic_search = ChunkedSemanticSearch()
            embeddings = semantic_search.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            documents = load_movies()
            semantic_search = ChunkedSemanticSearch()
            semantic_search.load_or_create_chunk_embeddings(documents)
            results = semantic_search.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search = SemanticSearch()
            documents = load_movies()
            semantic_search.load_or_create_embeddings(documents)
            results = semantic_search.search(args.query, args.limit)
            for result_number, result in enumerate(results, start=1):
                print(f"{result_number}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {result['description']}")
        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for chunk_number, chunk in enumerate(chunks, start=1):
                print(f"{chunk_number}. {chunk}")
        case "semantic_chunk":
            chunks = semantic_chunk_text(
                args.text, args.max_chunk_size, args.overlap
            )
            print(f"Semantically chunking {len(args.text)} characters")
            for chunk_number, chunk in enumerate(chunks, start=1):
                print(f"{chunk_number}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
