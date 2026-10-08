import argparse
import json

from lib.hybrid_search import HybridSearch
from lib.semantic_search import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json") as file:
        golden_dataset = json.load(file)

    search = HybridSearch(load_movies())
    print(f"k={limit}")
    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant = test_case["relevant_docs"]
        results = search.rrf_search(query, 60, limit)
        retrieved = [result["document"]["title"] for result in results[:limit]]
        precision = sum(title in relevant for title in retrieved) / limit

        print(f"\n- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant)}")


if __name__ == "__main__":
    main()
