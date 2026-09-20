import os

try:
    from inverted_index import InvertedIndex
    from lib.semantic_search import ChunkedSemanticSearch
except ModuleNotFoundError:
    from ..inverted_index import InvertedIndex
    from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float, limit: int = 5
    ) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = normalize_scores([score for _, score in bm25_results])
        semantic_scores = normalize_scores(
            [result["score"] for result in semantic_results]
        )
        combined = {}

        for (document, _), score in zip(bm25_results, bm25_scores):
            document_id = document["id"]
            combined.setdefault(
                document_id,
                {"document": document, "bm25": 0.0, "semantic": 0.0},
            )["bm25"] = score

        for result, score in zip(semantic_results, semantic_scores):
            document_id = result["id"]
            combined.setdefault(
                document_id,
                {"document": self.idx.docmap[document_id], "bm25": 0.0, "semantic": 0.0},
            )["semantic"] = score

        results = []
        for item in combined.values():
            item["hybrid"] = alpha * item["bm25"] + (1 - alpha) * item["semantic"]
            results.append(item)
        return sorted(results, key=lambda item: item["hybrid"], reverse=True)

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if minimum == maximum:
        return [1.0] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]
