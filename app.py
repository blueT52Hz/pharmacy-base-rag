"""Giao diện Streamlit demo baseline RAG hỏi đáp thuốc.

Chạy: streamlit run app.py
"""
import streamlit as st

from rag.config import DATA_CSV_PATH, EMBEDDING_MODEL, LLM_MODEL, TOP_K
from rag.rag_pipeline import RagPipeline, RetrievedChunk

EXAMPLE_QUESTIONS = [
    "Metformin chống chỉ định với những trường hợp nào?",
    "Liều dùng ibuprofen 200mg cho người lớn là bao nhiêu?",
    "Tác dụng phụ thường gặp của amoxicillin là gì?",
]

st.set_page_config(page_title="Hỏi đáp thuốc - Baseline RAG", page_icon="💊", layout="wide")


@st.cache_resource(show_spinner="Đang tải model embedding và chỉ mục...")
def load_pipeline() -> RagPipeline:
    return RagPipeline()


def render_sources(sources: list[RetrievedChunk]) -> None:
    if not sources:
        return
    with st.expander(f"Nguồn tham khảo ({len(sources)} đoạn)"):
        for i, src in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {src.ten_thuoc}** · {src.section_label} · NDC `{src.ndc}` · "
                f"{src.nha_san_xuat} · kê đơn: {src.ke_don} · tương đồng `{src.score:.3f}`"
            )
            st.text(src.text if len(src.text) <= 1200 else src.text[:1200] + " ...")


pipeline = load_pipeline()
stats = pipeline.stats()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Cấu hình")
    top_k = st.slider("Số đoạn truy xuất (top-k)", min_value=1, max_value=15, value=TOP_K)
    st.metric("Số thuốc trong SQLite", f"{stats['drugs']:,}")
    st.metric("Số vector trong ChromaDB", f"{stats['chunks']:,}")
    st.caption(f"Dữ liệu: `{DATA_CSV_PATH.name}`")
    st.caption(f"LLM: `{LLM_MODEL}`")
    st.caption(f"Embedding: `{EMBEDDING_MODEL}`")
    if st.button("Xoá hội thoại"):
        st.session_state.messages = []
        st.rerun()

st.title("💊 Chatbot hỏi đáp thuốc")
st.caption(
    "Baseline RAG · dữ liệu nhãn thuốc DailyMed · trả lời tiếng Việt · "
    "mỗi câu hỏi được xử lý độc lập"
)

if stats["chunks"] == 0:
    st.warning(
        "Chưa có chỉ mục. Dừng app, chạy `python build_index.py --limit 300`, "
        "rồi chạy lại `streamlit run app.py` (F5 không đủ vì chỉ mục được cache trong tiến trình)."
    )
    st.stop()

# Lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

# Câu hỏi gợi ý khi chưa có hội thoại
example_clicked = None
if not st.session_state.messages:
    st.markdown("**Thử hỏi:**")
    for col, example in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
        if col.button(example):
            example_clicked = example

question = st.chat_input("Nhập câu hỏi về thuốc...") or example_clicked

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm đoạn liên quan..."):
            sources = pipeline.retrieve(question, top_k)
        try:
            answer = st.write_stream(pipeline.stream_answer(question, sources))
        except Exception as exc:  # lỗi mạng / API key / model slug sai
            answer = f"Lỗi khi gọi LLM: {exc}"
            st.error(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": str(answer), "sources": sources}
    )
    if len(st.session_state.messages) == 2:
        st.rerun()  # ẩn khối câu hỏi gợi ý sau lần hỏi đầu tiên
