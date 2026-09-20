import argparse
import json
import string
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex


stemmer = PorterStemmer()


def preprocess(text: str) -> list[str]:
    translation_table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translation_table).split()


def matches(query: str, title: str, stop_words: list[str]) -> bool:
    query_tokens = [token for token in preprocess(query) if token not in stop_words]
    title_tokens = [token for token in preprocess(title) if token not in stop_words]
    query_stems = [stemmer.stem(token) for token in query_tokens]
    title_stems = [stemmer.stem(token) for token in title_tokens]

    return any(
        query_stem in title_stem
        for query_stem in query_stems
        for title_stem in title_stems
    )


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
    docs = index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


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
            with open("data/movies.json") as file:
                movies = json.load(file)
            with open("data/stopwords.txt") as file:
                stop_words = [
                    token
                    for line in file.read().splitlines()
                    for token in preprocess(line)
                ]

            results = []
            for movie in movies["movies"]:
                if matches(args.query, movie["title"], stop_words):
                    results.append(movie)

            for index, movie in enumerate(results[:5], start=1):
                print(f"{index}. {movie['title']}")
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
