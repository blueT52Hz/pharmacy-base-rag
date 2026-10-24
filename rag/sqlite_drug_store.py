"""SQLite là nguồn dữ liệu gốc: bảng drugs (thông tin thuốc) + chunks (nội dung từng đoạn)."""
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from rag.config import SQLITE_PATH
from rag.drug_csv_loader import DRUG_COLUMNS
from rag.text_chunker import Chunk


@contextmanager
def open_connection() -> Iterator[sqlite3.Connection]:
    """Mở kết nối mới mỗi lần dùng (an toàn với nhiều thread của Streamlit)."""
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    columns_sql = ", ".join(
        f"{col} TEXT PRIMARY KEY" if col == "ndc" else f"{col} TEXT" for col in DRUG_COLUMNS
    )
    conn.execute(f"CREATE TABLE IF NOT EXISTS drugs ({columns_sql})")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            ndc TEXT NOT NULL REFERENCES drugs(ndc),
            section TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_ndc ON chunks(ndc)")


def reset_schema(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("DROP TABLE IF EXISTS drugs")
    init_schema(conn)


def upsert_drugs(conn: sqlite3.Connection, drugs: list[dict]) -> None:
    cols = ", ".join(DRUG_COLUMNS)
    params = ", ".join(f":{col}" for col in DRUG_COLUMNS)
    conn.executemany(f"INSERT OR REPLACE INTO drugs ({cols}) VALUES ({params})", drugs)


def upsert_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO chunks (chunk_id, ndc, section, chunk_index, text) "
        "VALUES (?, ?, ?, ?, ?)",
        [(c.chunk_id, c.ndc, c.section, c.chunk_index, c.text) for c in chunks],
    )


def fetch_chunks_with_drug(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, sqlite3.Row]:
    """Lấy nội dung chunk + thông tin thuốc tương ứng, trả về dict theo chunk_id."""
    if not chunk_ids:
        return {}
    placeholders = ", ".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"""SELECT c.chunk_id, c.ndc, c.section, c.text,
                   d.ten_thuoc, d.nha_san_xuat, d.ke_don
            FROM chunks c JOIN drugs d ON d.ndc = c.ndc
            WHERE c.chunk_id IN ({placeholders})""",
        chunk_ids,
    ).fetchall()
    return {row["chunk_id"]: row for row in rows}


def count_drugs(conn: sqlite3.Connection) -> int:
    init_schema(conn)
    return conn.execute("SELECT COUNT(*) FROM drugs").fetchone()[0]
