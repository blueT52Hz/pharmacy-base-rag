"""Cấu hình tập trung: đường dẫn, model, tham số RAG (đọc từ file .env nếu có)."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    """Như os.getenv nhưng coi giá trị rỗng (vd 'TOP_K=') là chưa đặt."""
    return os.getenv(name) or default


# Bộ dữ liệu đang dùng (đổi qua DATA_CSV trong .env). Mỗi bộ có thư mục chỉ mục riêng
# storage/<tên file>/ để chuyển qua lại không phải build lại.
DATA_CSV_PATH = PROJECT_ROOT / _env("DATA_CSV", "data/tap-mau-thuoc.csv")
STORAGE_DIR = PROJECT_ROOT / "storage" / DATA_CSV_PATH.stem
SQLITE_PATH = STORAGE_DIR / "drugs.sqlite"
CHROMA_DIR = STORAGE_DIR / "chroma"
CHROMA_COLLECTION = "drug_chunks"

# LLM qua OpenRouter (API tương thích OpenAI)
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = _env("LLM_MODEL", "deepseek/deepseek-v4-flash")
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.2"))

# Embedding + chunking + truy xuất
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
CHUNK_SIZE = int(_env("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "150"))
TOP_K = int(_env("TOP_K", "6"))
