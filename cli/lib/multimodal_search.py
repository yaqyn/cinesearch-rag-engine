from PIL import Image
from sentence_transformers import SentenceTransformer

try:
    from lib.search_utils import format_search_result
    from lib.semantic_search import cosine_similarity, load_movies
except ModuleNotFoundError:
    from .search_utils import format_search_result
    from .semantic_search import cosine_similarity, load_movies


class MultimodalSearch:
    def __init__(
        self, documents: list[dict] | None = None, model_name: str = "clip-ViT-B-32"
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.text_embeddings = None
        if self.documents:
            texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
            self.text_embeddings = self.model.encode(texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        return self.model.encode([image])[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        if self.text_embeddings is None:
            raise ValueError("No text embeddings loaded.")

        image_embedding = self.embed_image(image_path)
        scored = [
            (cosine_similarity(image_embedding, embedding), document)
            for embedding, document in zip(self.text_embeddings, self.documents)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            format_search_result(
                document["id"],
                document["title"],
                document["description"],
                score,
            )
            for score, document in scored[:limit]
        ]


def verify_image_embedding(image_path: str) -> None:
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    search = MultimodalSearch(load_movies())
    return search.search_with_image(image_path)
