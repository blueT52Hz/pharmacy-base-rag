"""Nạp CSV vào SQLite rồi embed các chunk vào ChromaDB.

Ví dụ:
    python build_index.py --limit 300     # chỉ mục nhỏ để demo nhanh
    python build_index.py                 # toàn bộ 4.991 thuốc (lâu, có thể chạy lại để resume)
    python build_index.py --reset         # xoá chỉ mục cũ và tạo lại

Lưu ý:
- Đổi CHUNK_SIZE / CHUNK_OVERLAP / EMBEDDING_MODEL / cách làm sạch text -> BẮT BUỘC chạy với --reset
  (resume chỉ bỏ qua theo chunk_id, không biết nội dung đã đổi).
- Không build khi app Streamlit đang chạy; build xong phải dừng và chạy lại `streamlit run app.py`.
"""
import argparse

from tqdm import tqdm

from rag.chroma_vector_store import add_chunks, existing_ids, get_collection
from rag.config import DATA_CSV_PATH, EMBEDDING_MODEL
from rag.drug_csv_loader import load_drugs
from rag.sqlite_drug_store import init_schema, open_connection, reset_schema, upsert_chunks, upsert_drugs
from rag.text_chunker import chunk_drug
from rag.text_embedder import TextEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xây chỉ mục baseline RAG hỏi đáp thuốc")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ lấy N thuốc đầu tiên")
    parser.add_argument("--batch-size", type=int, default=64, help="Số chunk mỗi lần embed")
    parser.add_argument("--reset", action="store_true", help="Xoá chỉ mục cũ trước khi build")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Đọc dữ liệu: {DATA_CSV_PATH}")
    drugs = load_drugs(DATA_CSV_PATH, args.limit)
    chunks = [chunk for drug in drugs for chunk in chunk_drug(drug)]
    print(f"{len(drugs)} thuốc -> {len(chunks)} chunk")

    with open_connection() as conn:
        if args.reset:
            reset_schema(conn)
        else:
            init_schema(conn)
        upsert_drugs(conn, drugs)
        upsert_chunks(conn, chunks)
    print("Đã lưu thuốc + chunk vào SQLite")

    collection = get_collection(reset=args.reset)
    print(f"Tải model embedding: {EMBEDDING_MODEL}")
    embedder = TextEmbedder(EMBEDDING_MODEL)

    skipped = 0
    for start in tqdm(range(0, len(chunks), args.batch_size), desc="Embedding", unit="batch"):
        batch = chunks[start : start + args.batch_size]
        done = existing_ids(collection, [c.chunk_id for c in batch])
        todo = [c for c in batch if c.chunk_id not in done]
        skipped += len(batch) - len(todo)
        if todo:
            add_chunks(collection, todo, embedder.embed_passages([c.text for c in todo]))

    print(f"Xong. Vector trong Chroma: {collection.count()} (bỏ qua {skipped} chunk đã có)")


if __name__ == "__main__":
    main()
