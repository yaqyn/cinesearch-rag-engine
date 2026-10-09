import unittest

from cli.inverted_index import tokenize_text
from cli.lib.hybrid_search import normalize_scores, rrf_score
from cli.lib.semantic_search import semantic_chunk_text


class CoreSearchTests(unittest.TestCase):
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
