import unittest
import io
from contextlib import redirect_stderr
from unittest.mock import patch
from cli.cinesearch import main

from cli.inverted_index import tokenize_text
from cli.lib.hybrid_search import normalize_scores, rrf_score
from cli.lib.semantic_search import semantic_chunk_text


class CoreSearchTests(unittest.TestCase):
    def test_reranking_requests_a_larger_candidate_pool(self):
        import sys
        from pathlib import Path
        from contextlib import redirect_stdout
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "cli"), *sys.path]):
            from cli import hybrid_search_cli
        for method, expected_limit in [(None, 2), ("batch", 10)]:
            argv = ["hybrid-search", "rrf-search", "bear", "--limit", "2"]
            if method:
                argv += ["--rerank-method", method]
            with patch.object(sys, "argv", argv), patch.object(hybrid_search_cli, "load_movies", return_value=[]), patch.object(hybrid_search_cli, "HybridSearch") as search, patch.object(hybrid_search_cli, "rerank_batch", return_value=[]) as rerank, redirect_stdout(io.StringIO()):
                search.return_value.rrf_search.return_value = []
                hybrid_search_cli.main()
                search.return_value.rrf_search.assert_called_once_with("bear", 60, expected_limit)
                if method:
                    rerank.assert_called_once_with("bear", [])

    def test_hybrid_backends_honor_the_requested_result_limit(self):
        from types import SimpleNamespace
        from cli.lib.hybrid_search import HybridSearch
        documents = {i: {"id": i, "title": str(i)} for i in range(1, 5)}
        search = HybridSearch.__new__(HybridSearch)
        search.idx = SimpleNamespace(docmap=documents)
        search._bm25_search = lambda query, limit: [(documents[i], float(5-i)) for i in range(1, 5)]
        search.semantic_search = SimpleNamespace(search_chunks=lambda query, limit: [
            {"id": i, "score": float(5-i)} for i in range(1, 5)])
        weighted = search.weighted_search("test", alpha=0.5, limit=2)
        reciprocal = search.rrf_search("test", k=60, limit=2)
        self.assertEqual([row["document"]["id"] for row in weighted], [1, 2])
        self.assertEqual([row["document"]["id"] for row in reciprocal], [1, 2])

    def test_rebuilding_index_removes_deleted_documents_and_terms(self):
        from cli.inverted_index import InvertedIndex
        index = InvertedIndex()
        with patch("cli.inverted_index.load_movies", return_value=[
            {"id": 1, "title": "Bear", "description": "forest"},
            {"id": 2, "title": "Ocean", "description": "waves"},
        ]):
            index.build()
        with patch("cli.inverted_index.load_movies", return_value=[
            {"id": 1, "title": "Desert", "description": "sand"},
        ]):
            index.build()
        self.assertEqual(set(index.docmap), {1})
        self.assertEqual(set(index.term_frequencies), {1})
        self.assertEqual(set(index.doc_lengths), {1})
        self.assertEqual(index.get_documents("bear"), [])
        self.assertEqual(index.get_documents("ocean"), [])
        self.assertEqual(index.get_documents("desert"), [1])

    def test_blank_queries_fail_before_loading_data(self):
        for query in ["", "  \t\n"]:
            for mode in ["keyword", "semantic", "hybrid"]:
                with self.subTest(query=query, mode=mode), patch("cli.cinesearch.search_catalog") as load:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        main(["search", query, "--mode", mode])
                    self.assertEqual(error.exception.code, 2)
                    load.assert_not_called()

    def test_invalid_limits_fail_before_loading_data(self):
        for value in ["0", "-1", "invalid"]:
            with self.subTest(value=value), patch("cli.cinesearch.search_catalog") as load:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(["search", "bear", "--limit", value])
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
    def test_tokenization_is_normalized_and_stemmed(self):
        self.assertEqual(tokenize_text("The bears, running!"), ["bear", "run"])

    def test_normalization_handles_empty_and_constant_scores(self):
        self.assertEqual(normalize_scores([]), [])
        self.assertEqual(normalize_scores([4.0, 4.0]), [1.0, 1.0])

    def test_rrf_score_decreases_with_rank(self):
        self.assertGreater(rrf_score(1, 60), rrf_score(2, 60))

    def test_semantic_chunking_handles_whitespace(self):
        self.assertEqual(semantic_chunk_text("  ", 4), [])
        self.assertEqual(semantic_chunk_text("Text without punctuation", 4), ["Text without punctuation"])


if __name__ == "__main__":
    unittest.main()


class KeywordPerformanceRegressionTests(unittest.TestCase):
    def test_stemming_reuses_repeated_words(self):
        from cli.inverted_index import stem_word, tokenize_text
        stem_word.cache_clear()
        self.assertEqual(tokenize_text("bears bears bears"), ["bear", "bear", "bear"])
        self.assertEqual(stem_word.cache_info().misses, 1)
        self.assertEqual(stem_word.cache_info().hits, 2)

    def test_optimized_bm25_matches_reference_scores(self):
        from cli.inverted_index import InvertedIndex
        from collections import Counter
        index = InvertedIndex()
        index.docmap = {1: {"id": 1}, 2: {"id": 2}, 3: {"id": 3}}
        index.index = {"bear": {1, 2}, "forest": {2}}
        index.term_frequencies = {1: Counter(bear=2), 2: Counter(bear=1, forest=1), 3: Counter(other=1)}
        index.doc_lengths = {1: 2, 2: 2, 3: 1}
        expected = {doc_id: sum(index.bm25(doc_id, token) for token in ["bear", "bear", "forest"])
                    for doc_id in index.docmap}
        actual = index.bm25_search("bear bear forest", 3)
        for doc, score in actual:
            self.assertAlmostEqual(score, expected[doc["id"]])
        self.assertEqual([doc["id"] for doc, _ in actual], [2, 1, 3])
