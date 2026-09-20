import argparse

from lib.semantic_search import embed_query_text, embed_text, verify_embeddings, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the semantic search model")
    embed_parser = subparsers.add_parser("embed_text", help="Generate a text embedding")
    embed_parser.add_argument("text", type=str)
    subparsers.add_parser("verify_embeddings", help="Verify movie embeddings")
    query_parser = subparsers.add_parser("embed_query", help="Generate a query embedding")
    query_parser.add_argument("query", type=str)
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
