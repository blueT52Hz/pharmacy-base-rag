"""HTTP API cho BE NestJS gọi (hợp đồng v1, xem plans/261105-0118-pos-rag-assistant/design.md).

Bản RAG hoàn chỉnh chỉ cần giữ đúng hợp đồng này (GET /info, POST /chat dạng SSE) là FE + BE không phải sửa.

Chạy: uvicorn api:app --host 127.0.0.1 --port 8790
"""
import json
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Iterator, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.config import RAG_INTERNAL_KEY, TOP_K
from rag.example_questions import EXAMPLE_QUESTIONS
from rag.rag_pipeline import RagPipeline, RetrievedChunk

logger = logging.getLogger("base-rag.api")

# Nguồn gửi về FE chỉ để xem nhanh; giữ ngắn cho stream nhẹ.
SNIPPET_MAX_CHARS = 1500

pipeline: RagPipeline | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Tải model embedding + chỉ mục một lần lúc khởi động, không để câu hỏi đầu tiên phải chờ.
    global pipeline
    pipeline = RagPipeline()
    yield


app = FastAPI(title="Baseline RAG - hỏi đáp thuốc", lifespan=lifespan)


def require_internal_key(x_internal_key: str | None = Header(default=None)) -> None:
    if not RAG_INTERNAL_KEY:
        return
    if not x_internal_key or not secrets.compare_digest(x_internal_key, RAG_INTERNAL_KEY):
        raise HTTPException(status_code=401, detail="Sai X-Internal-Key")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Baseline cố ý không dùng history (mỗi câu độc lập); vẫn nhận để giữ chung hợp đồng với bản hoàn chỉnh.
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    topK: int | None = Field(default=None, ge=1, le=15)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def to_source(chunk: RetrievedChunk) -> dict:
    return {
        "id": chunk.chunk_id,
        "title": chunk.ten_thuoc,
        "section": chunk.section_label,
        "snippet": chunk.text[:SNIPPET_MAX_CHARS],
        "score": round(chunk.score, 4),
        "metadata": {"NDC": chunk.ndc, "Nhà sản xuất": chunk.nha_san_xuat, "Kê đơn": chunk.ke_don},
    }


@app.get("/info", dependencies=[Depends(require_internal_key)])
def info() -> dict:
    return {
        "pipeline": "baseline",
        "version": "1",
        "features": {"history": False, "feedback": False},
        "stats": pipeline.stats(),
        "suggestions": EXAMPLE_QUESTIONS,
    }


@app.post("/chat", dependencies=[Depends(require_internal_key)])
def chat(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()
    top_k = req.topK or TOP_K

    # Generator đồng bộ: Starlette chạy nó trong threadpool nên embed/LLM không chặn event loop.
    def events() -> Iterator[str]:
        try:
            yield sse("status", {"stage": "retrieve", "label": "Đang tìm đoạn nhãn thuốc liên quan..."})
            chunks = pipeline.retrieve(question, top_k)
            yield sse("sources", {"sources": [to_source(c) for c in chunks]})
            yield sse("status", {"stage": "generate", "label": "Đang soạn câu trả lời..."})
            for text in pipeline.stream_answer(question, chunks):
                yield sse("delta", {"text": text})
            yield sse("done", {"answerId": None})
        except Exception:  # lỗi mạng / API key / model slug sai — không lộ chi tiết ra FE
            logger.exception("Lỗi khi trả lời câu hỏi")
            yield sse("error", {"message": "Trợ lý gặp lỗi khi trả lời. Kiểm tra log server RAG."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
