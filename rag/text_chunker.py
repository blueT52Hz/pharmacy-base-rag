"""Chunking baseline: mỗi mục (Chỉ định, Liều lượng...) cắt theo cửa sổ ký tự cố định có overlap."""
from dataclasses import dataclass

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.drug_csv_loader import TEXT_SECTIONS

SECTION_LABELS = {"tong_quan": "Tổng quan", **TEXT_SECTIONS}


@dataclass
class Chunk:
    chunk_id: str  # "{ndc}:{section}:{index}" -> ổn định, cho phép chạy lại không trùng
    ndc: str
    section: str
    chunk_index: int
    text: str  # đã gắn header "Thuốc | Mục" để embedding biết đoạn thuộc thuốc nào


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cắt text thành các đoạn <= size ký tự, ưu tiên cắt tại xuống dòng/khoảng trắng."""
    if len(text) <= size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Tìm điểm cắt tự nhiên ở nửa sau cửa sổ để không cắt giữa từ
            cut = text.rfind("\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def build_overview_text(drug: dict) -> str:
    """Đoạn 'Tổng quan' từ các trường thông tin có cấu trúc (tiếng Việt)."""
    fields = [
        ("Tên thuốc", drug["ten_thuoc"]),
        ("Nhóm thuốc", drug["nhom_thuoc"]),
        ("Thuốc kê đơn", drug["ke_don"]),
        ("Thành phần", drug["thanh_phan"]),
        ("Hàm lượng", drug["ham_luong"]),
        ("Đường dùng", drug["duong_dung"]),
        ("Quy cách đóng gói", drug["quy_cach"]),
        ("Nhà sản xuất", drug["nha_san_xuat"]),
        ("Số đăng ký", drug["so_dang_ky"]),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def chunk_drug(drug: dict) -> list[Chunk]:
    sections = [("tong_quan", build_overview_text(drug))]
    sections += [(key, drug[key]) for key in TEXT_SECTIONS]
    chunks: list[Chunk] = []
    for section, content in sections:
        if not content:
            continue
        # Kèm thành phần vì nhiều tên thương mại chung chung ("ALLERGY", "ACID REDUCER")
        ingredient = f" | Thành phần: {drug['thanh_phan']}" if drug["thanh_phan"] else ""
        header = f"Thuốc: {drug['ten_thuoc']}{ingredient} | Mục: {SECTION_LABELS[section]}"
        for index, piece in enumerate(split_text(content)):
            chunks.append(
                Chunk(
                    chunk_id=f"{drug['ndc']}:{section}:{index}",
                    ndc=drug["ndc"],
                    section=section,
                    chunk_index=index,
                    text=f"{header}\n{piece}",
                )
            )
    return chunks
