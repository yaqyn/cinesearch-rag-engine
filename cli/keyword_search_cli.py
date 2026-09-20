import argparse
import math

from inverted_index import InvertedIndex, tokenize_term, tokenize_text
from constants import BM25_B, BM25_K1


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(tokenize_term(term))


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, tokenize_term(term), k1, b)


def bm25_search_command(query: str, limit: int):
    index = InvertedIndex()
    index.load()
    return index.bm25_search(query, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get a term frequency")
    tf_parser.add_argument("doc_id", type=int)
    tf_parser.add_argument("term", type=str)
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str)
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25_search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument("--limit", type=int, default=5)

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
        case "tf":
            index = InvertedIndex()
            index.load()
            term = tokenize_term(args.term)
            print(index.get_tf(args.doc_id, term))
        case "idf":
            index = InvertedIndex()
            index.load()
            term = tokenize_term(args.term)
            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(term))
            idf = math.log(
                (total_doc_count + 1) / (term_match_doc_count + 1)
            )
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            index = InvertedIndex()
            index.load()
            term = tokenize_term(args.term)
            term_frequency = index.get_tf(args.doc_id, term)
            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(term))
            idf = math.log(
                (total_doc_count + 1) / (term_match_doc_count + 1)
            )
            tf_idf = term_frequency * idf
            print(
                f"TF-IDF score of '{args.term}' in document "
                f"'{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document "
                f"'{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            results = bm25_search_command(args.query, args.limit)
            for result_number, (movie, score) in enumerate(results, start=1):
                print(
                    f"{result_number}. ({movie['id']}) {movie['title']} "
                    f"- Score: {score:.2f}"
                )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
