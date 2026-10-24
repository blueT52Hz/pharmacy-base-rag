"""Pipeline baseline: embed câu hỏi -> top-k Chroma -> lấy text từ SQLite -> LLM trả lời."""
from dataclasses import dataclass
from typing import Iterator

from rag.chroma_vector_store import get_collection, query_similar
from rag.config import EMBEDDING_MODEL
from rag.openrouter_llm_client import stream_chat
from rag.sqlite_drug_store import count_drugs, fetch_chunks_with_drug, open_connection
from rag.text_chunker import SECTION_LABELS
from rag.text_embedder import TextEmbedder

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu thông tin thuốc cho nhà thuốc.
Quy tắc:
- Chỉ trả lời dựa trên phần NGỮ CẢNH được cung cấp (trích từ nhãn thuốc; có thể là tiếng Việt hoặc tiếng Anh).
- Luôn trả lời bằng tiếng Việt, rõ ràng, có cấu trúc; giữ nguyên tên thuốc, hoạt chất, liều và đơn vị.
- Ghi số nguồn dạng [1], [2] ngay sau ý lấy từ nguồn đó.
- Nếu ngữ cảnh không đủ để trả lời, nói rõ "Không tìm thấy thông tin này trong dữ liệu" và không được bịa.
- Nếu các nguồn thuộc nhiều sản phẩm / nhà sản xuất khác nhau, nêu rõ thông tin thuộc sản phẩm nào.
- Kết thúc bằng một câu lưu ý: thông tin chỉ để tham khảo, cần hỏi ý kiến bác sĩ hoặc dược sĩ."""

NO_CONTEXT_ANSWER = "Không tìm thấy đoạn dữ liệu nào liên quan. Hãy kiểm tra lại chỉ mục (build_index.py)."


@dataclass
class RetrievedChunk:
    chunk_id: str
    ndc: str
    ten_thuoc: str
    nha_san_xuat: str
    ke_don: str
    section_label: str
    text: str
    score: float  # độ tương đồng cosine (1 - distance)


class RagPipeline:
    def __init__(self):
        self.embedder = TextEmbedder(EMBEDDING_MODEL)
        self.collection = get_collection()

    def stats(self) -> dict:
        with open_connection() as conn:
            drugs = count_drugs(conn)
        return {"drugs": drugs, "chunks": self.collection.count()}

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        hits = query_similar(self.collection, self.embedder.embed_query(question), top_k)
        with open_connection() as conn:
            rows = fetch_chunks_with_drug(conn, [chunk_id for chunk_id, _ in hits])
        results: list[RetrievedChunk] = []
        for chunk_id, distance in hits:
            row = rows.get(chunk_id)
            if row is None:  # vector còn nhưng SQLite thiếu (chỉ mục lệch) -> bỏ qua
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    ndc=row["ndc"],
                    ten_thuoc=row["ten_thuoc"],
                    nha_san_xuat=row["nha_san_xuat"],
                    ke_don=row["ke_don"],
                    section_label=SECTION_LABELS.get(row["section"], row["section"]),
                    text=row["text"],
                    score=1.0 - distance,
                )
            )
        return results

    @staticmethod
    def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
        context = "\n\n".join(
            f"[{i}] {c.ten_thuoc} (NDC {c.ndc}, NSX: {c.nha_san_xuat}, kê đơn: {c.ke_don})\n{c.text}"
            for i, c in enumerate(chunks, start=1)
        )
        user_content = f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def stream_answer(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        if not chunks:
            yield NO_CONTEXT_ANSWER
            return
        yield from stream_chat(self.build_messages(question, chunks))
