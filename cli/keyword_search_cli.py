import argparse
from inverted_index import InvertedIndex, tokenize_text


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Error: the inverted index has not been built yet.")
                return

            results = []
            seen = set()
            for token in tokenize_text(args.query):
                for doc_id in index.get_documents(token):
                    if doc_id not in seen:
                        seen.add(doc_id)
                        results.append(index.docmap[doc_id])
                    if len(results) == 5:
                        break
                if len(results) == 5:
                    break

            for result_number, movie in enumerate(results, start=1):
                print(f"{result_number}. {movie['title']} (ID: {movie['id']})")
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
