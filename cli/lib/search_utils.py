SCORE_PRECISION = 4


def format_search_result(doc_id, title, document, score, metadata=None):
    return {
        "id": doc_id,
        "title": title,
        "document": document[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }
