import unicodedata
import re

SCORE_PRECISION = 4
DOCUMENT_PREVIEW_LENGTH = 100
ANSI_OSC_PATTERN = re.compile(
    r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c|$)"
)


def document_preview(document):
    """Return a compact excerpt without cutting the final word in half."""
    document = ANSI_OSC_PATTERN.sub(" ", document)
    document = re.sub(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]", " ", document)
    safe = "".join(" " if unicodedata.category(char).startswith("C") else char for char in document)
    compact = " ".join(safe.split())
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
