SCORE_PRECISION = 4
DOCUMENT_PREVIEW_LENGTH = 100


def document_preview(document):
    """Return a compact excerpt without cutting the final word in half."""
    compact = " ".join(document.split())
    if len(compact) <= DOCUMENT_PREVIEW_LENGTH:
        return compact
    prefix = compact[: DOCUMENT_PREVIEW_LENGTH - 1]
    if " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.rstrip() + "…"


def format_search_result(doc_id, title, document, score, metadata=None):
    return {
        "id": doc_id,
        "title": title,
        "document": document_preview(document),
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }
