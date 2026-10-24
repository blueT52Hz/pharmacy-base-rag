"""ChromaDB chỉ giữ vector + metadata tối thiểu; nội dung chunk đọc từ SQLite theo chunk_id."""
import chromadb
from chromadb.api.models.Collection import Collection

from rag.config import CHROMA_COLLECTION, CHROMA_DIR
from rag.text_chunker import Chunk


def get_collection(reset: bool = False) -> Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass  # collection chưa tồn tại
    # embedding_function=None: luôn tự truyền vector, không để Chroma tải model mặc định
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def existing_ids(collection: Collection, ids: list[str]) -> set[str]:
    """Các chunk_id đã có vector -> bỏ qua khi chạy lại build_index (resume)."""
    return set(collection.get(ids=ids, include=[])["ids"])


def add_chunks(collection: Collection, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        metadatas=[{"ndc": c.ndc, "section": c.section} for c in chunks],
    )


def query_similar(collection: Collection, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
    """Trả về [(chunk_id, cosine_distance)] gần nhất, sắp theo độ gần giảm dần."""
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["distances"],
    )
    return list(zip(result["ids"][0], result["distances"][0]))
