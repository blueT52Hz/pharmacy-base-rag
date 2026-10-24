"""Wrapper sentence-transformers. Họ model e5 cần tiền tố 'query: ' / 'passage: '."""
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.uses_e5_prefix = "e5" in model_name.lower()

    def _encode(self, texts: list[str], prefix: str, batch_size: int) -> list[list[float]]:
        if self.uses_e5_prefix:
            texts = [f"{prefix}{text}" for text in texts]
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # chuẩn hoá để dùng khoảng cách cosine
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return self._encode(texts, "passage: ", batch_size)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], "query: ", 1)[0]
