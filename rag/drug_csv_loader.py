"""Đọc CSV DailyMed, map cột sang khoá ngắn và làm sạch text ở mức nhẹ (baseline)."""
import csv
import re
from pathlib import Path

# Thứ tự cột trong CSV gốc -> khoá ngắn. None = bỏ cột (rỗng / không mang thông tin).
COLUMN_KEYS = [
    "nhom_thuoc", "ndc", "ten_thuoc", None, "ke_don", "so_dang_ky", "ngay_cap", None,
    "ham_luong", "nha_san_xuat", None, "quy_cach", "duong_dung", "thanh_phan",
    "chi_dinh", "chong_chi_dinh", "bao_quan", "tac_dung", "tac_dung_phu",
    "khuyen_cao", "lieu_luong",
]
DRUG_COLUMNS = [key for key in COLUMN_KEYS if key]

# Các mục nội dung dài sẽ được chunk + embed, kèm nhãn tiếng Việt
TEXT_SECTIONS = {
    "chi_dinh": "Chỉ định",
    "chong_chi_dinh": "Chống chỉ định",
    "tac_dung": "Tác dụng",
    "tac_dung_phu": "Tác dụng phụ",
    "khuyen_cao": "Khuyến cáo",
    "lieu_luong": "Liều lượng",
    "bao_quan": "Cách bảo quản",
}

# Gộp các nhãn đường dùng gần trùng nhau
ROUTE_ALIASES = {"Dùng ngoài da": "Dùng ngoài"}

_BULLET_RE = re.compile(r"^[ \t]*-?[ \t]*•[ \t]*", re.MULTILINE)
_DISTRIBUTOR_LINE_RE = re.compile(
    # Chỉ xoá dòng BẮT ĐẦU bằng cụm này (địa chỉ NPP); không đụng câu lâm sàng có nhắc tới NSX
    r"^[ \t]*(?:repackaged by\s*/\s*)?(?:(?:distributed|manufactured|marketed)\s+(?:by|for)\b"
    r"|(?:phân phối|sản xuất|đóng gói lại)\s+(?:bởi|cho)\b|nhà phân phối\s*:).*$",
    re.IGNORECASE | re.MULTILINE,
)
_SPACES_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Chuẩn hoá bullet '- •', bỏ dòng nhà phân phối, gộp khoảng trắng thừa."""
    text = text.replace("\r", "")
    text = _BULLET_RE.sub("- ", text)
    text = _DISTRIBUTOR_LINE_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def load_drugs(csv_path: Path, limit: int | None = None) -> list[dict]:
    """Trả về danh sách thuốc dạng dict {khoá ngắn: giá trị đã làm sạch}."""
    csv.field_size_limit(10_000_000)  # một số trường dài > 70K ký tự
    drugs: list[dict] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != len(COLUMN_KEYS):
            raise ValueError(
                f"CSV có {len(header)} cột, cần đúng {len(COLUMN_KEYS)} cột theo mẫu DailyMed"
            )
        for row in reader:
            row = row + [""] * (len(COLUMN_KEYS) - len(row))  # dòng thiếu cột -> bù rỗng
            drug = {key: clean_text(value) for key, value in zip(COLUMN_KEYS, row) if key}
            if not drug.get("ndc"):
                continue
            drug["duong_dung"] = ROUTE_ALIASES.get(drug["duong_dung"], drug["duong_dung"])
            drugs.append(drug)
            if limit and len(drugs) >= limit:
                break
    return drugs
