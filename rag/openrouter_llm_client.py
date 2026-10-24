"""Gọi LLM qua OpenRouter (API tương thích OpenAI), trả kết quả dạng stream."""
from typing import Iterator

from openai import OpenAI

from rag.config import LLM_MODEL, LLM_TEMPERATURE, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Thiếu OPENROUTER_API_KEY trong file .env")
    if _client is None:
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    return _client


def stream_chat(messages: list[dict]) -> Iterator[str]:
    """Yield từng đoạn text của câu trả lời."""
    stream = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        stream=True,
        extra_headers={"X-Title": "Baseline RAG hoi dap thuoc"},
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content
