# Baseline RAG – Chatbot hỏi đáp thuốc

Hỏi tiếng Việt, trả lời tiếng Việt dựa trên nhãn thuốc DailyMed. Hai bộ dữ liệu, chọn qua `DATA_CSV` trong `.env`:

- `data/tap-mau-thuoc.csv` — 50 thuốc, nội dung tiếng Việt (**mặc định**)
- `data/ds-thuoc-dailymed.csv` — 4.991 thuốc, nội dung tiếng Anh

Mỗi bộ có chỉ mục riêng trong `storage/<tên file>/`:

- **SQLite** (`drugs.sqlite`): thông tin thuốc + nội dung các đoạn (chunk)
- **ChromaDB** (`chroma/`): vector của các đoạn
- **Embedding**: `intfloat/multilingual-e5-small` (chạy local)
- **LLM**: DeepSeek qua OpenRouter
- **Giao diện**: Streamlit

## 1. Cài đặt

Cần Python 3.10+. Chạy trong thư mục `projects/base-rag`:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Mở `.env`, điền `OPENROUTER_API_KEY`. Kiểm tra lại `LLM_MODEL` tại https://openrouter.ai/models.

## 2. Tạo chỉ mục

Lệnh build áp dụng cho bộ dữ liệu đang chọn trong `DATA_CSV`.

**Cách A – trên máy (CPU; bản tiếng Việt ~1 phút, bản tiếng Anh ~2 giờ):**

```
.venv\Scripts\python build_index.py               # build; bị ngắt thì chạy lại để làm tiếp
.venv\Scripts\python build_index.py --reset       # xoá chỉ mục cũ, tạo lại từ đầu
.venv\Scripts\python build_index.py --limit 300   # chỉ lấy N thuốc đầu (demo nhanh)
```

**Cách B – Colab GPU (cho bộ tiếng Anh lớn, ~vài phút):**

1. Chạy `build_index.py` trên máy (ít nhất tới khi in "Đã lưu thuốc + chunk vào SQLite", có thể dừng sau đó)
2. Mở `colab-build-chroma.ipynb` trên Colab, chọn GPU T4, Run all, upload `storage/<tên file>/drugs.sqlite`
3. Tải `chroma.zip`, xoá `storage/<tên file>/chroma` cũ, giải nén vào `storage/<tên file>/`

## 3. Chạy demo

```
.venv\Scripts\streamlit run app.py
```

Mở http://localhost:8501. Sidebar hiển thị số thuốc / số vector và chỉnh được top-k.

## Lưu ý

- Build xong (hoặc đổi `.env`) phải **tắt và chạy lại** Streamlit — F5 không đủ.
- Không chạy `build_index.py` khi Streamlit đang mở.
- Đổi `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL` → phải build lại với `--reset` (và chạy lại notebook nếu dùng Colab).
- Đổi `DATA_CSV` không cần build lại nếu bộ đó đã có chỉ mục — chỉ cần chạy lại Streamlit.
- Mỗi câu hỏi xử lý độc lập (không nhớ hội thoại) — đúng thiết kế baseline.
- Thông tin chỉ để tham khảo, không thay thế tư vấn của bác sĩ / dược sĩ.
