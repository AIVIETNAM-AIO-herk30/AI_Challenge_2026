# Trạng thái triển khai — Multi-Agent Video Retrieval

Tài liệu này ghi lại các thay đổi đã thực hiện để đưa mã nguồn về cùng một
kiến trúc truy xuất đang dùng: **Turbovec + Elasticsearch + SigLIP**, sau đó
bổ sung lớp điều phối ReAct và giao diện Streamlit.

## 1. Sửa đường truy vấn trực tuyến

Trước đây, `src/inference.py` vẫn tham chiếu tới thiết kế FAISS/M/M/c cũ,
trong khi pipeline indexing hiện tại ghi dữ liệu vào Turbovec và
Elasticsearch. Đường truy vấn đã được chuyển sang kiến trúc hiện hành:

```text
Text query
  ├─ SigLIP text embedding → Turbovec (tương đồng ngữ nghĩa/hình ảnh)
  └─ Elasticsearch BM25 → OCR + ASR text
                    ↓
         Reciprocal Rank Fusion (RRF)
                    ↓
   frame_id → metadata → video_id, frame_idx, timestamp_sec, score
```

### Các file liên quan

- `src/inference.py`
  - Dùng `VisualAgent` ở text mode để tạo SigLIP embedding.
  - Nạp index `data/index/turbovec/siglip`.
  - Tìm kiếm song song theo semantic vector và OCR/ASR text.
  - Gộp hai bảng xếp hạng bằng Reciprocal Rank Fusion, không cộng trực tiếp
    score từ hai hệ thống có thang đo khác nhau.
  - Giữ nguyên API công khai:
    `search(query, config, top_k) -> list[dict]`.

- `src/retrieval/es_store.py`
  - Thêm `search_text()` để BM25 trên `ocr_text` và `asr_text`.
  - Thêm `get_many_by_frame_ids()` để lấy metadata theo lô thay vì một request
    cho từng frame.

Kết quả trả về vẫn tương thích với `src/eval.py`:

```python
{
    "video_id": "L21_V001",
    "frame_idx": 1234,
    "timestamp_sec": 41.13,
    "score": 0.0325,
}
```

## 2. Giao diện tìm kiếm Streamlit

`src/ui/app.py` đã được thay từ màn hình placeholder thành trình duyệt kết
quả tìm kiếm:

- Ô nhập mô tả truy vấn và lựa chọn số kết quả.
- Hiển thị keyframe, frame index, timestamp và fusion score.
- Preview video tại timestamp tương ứng nếu file video gốc có sẵn.
- Thông báo lỗi có hướng dẫn khi Turbovec index, model hoặc Elasticsearch chưa
  hoạt động.

## 3. ReAct Multi-Agent Orchestrator

File mới `src/agents/orchestrator.py` hiện thực baseline theo bản đặc tả
Multi-Agent AIC 2026.

### Năng lực hiện có

- Phân loại loại bài toán: `KIS`, `AVS`, `VQA`, `KISC`.
- Với KISC mơ hồ, yêu cầu người dùng bổ sung đặc điểm phân biệt thay vì đoán.
- Hỗ trợ inject `expand_fn` để nối Query Expansion Agent/LLM sau này; baseline
  không cần API key và dùng trực tiếp query gốc.
- Gộp evidence từ nhiều query expansion bằng RRF.
- Xây dựng các `TemporalClip`: nhóm các frame cùng video, gần nhau trong thời
  gian (mặc định 5 giây), giúp trình bày sự kiện thay vì danh sách frame rời
  rạc.
- Trả về summary ngắn có căn cứ từ evidence; VQA hiện chỉ thu thập evidence,
  chưa sinh câu trả lời tự nhiên bằng LLM.

Giao diện cho phép chọn `Auto` hoặc ép loại bài toán, duy trì lịch sử hội
thoại trong phiên Streamlit, và hiển thị các temporal candidate clips.

## 4. Kiểm tra đã thực hiện

- Biên dịch cú pháp Python cho các file thay đổi.
- Kiểm tra RRF fusion.
- Kiểm tra phân loại AVS/VQA, temporal grouping, gộp retrieval và clarification
  cho KISC bằng mock search function.
- Kiểm tra tính hợp lệ YAML của `configs/config.yaml`.
- Chạy `git diff --check` để phát hiện whitespace error.

Không thể chạy truy vấn end-to-end trong môi trường phát triển hiện tại vì
chưa có package `turbovec`, vector index đã build, và Elasticsearch đang chạy.

Các visual encoder cũng tự chọn dtype theo thiết bị: CUDA dùng `fp16`, còn CPU
dùng `fp32`. Điều này cho phép chạy fallback CPU khi CUDA/NCCL không sẵn sàng.

## 5. Cách chạy khi đã chuẩn bị hạ tầng

1. Cài dependencies của project và tải model/weights cần thiết.
2. Khởi động Elasticsearch:

   ```bash
   docker compose up -d elasticsearch
   ```

3. Index video để tạo Turbovec index, keyframe và Elasticsearch documents:

   ```bash
   python -m src.retrieval.video_indexer
   ```

4. Mở giao diện:

   ```bash
   docker compose up --build
   ```

   Hoặc chạy trực tiếp:

   ```bash
   streamlit run src/ui/app.py
   ```

## 6. Hướng triển khai tiếp theo

1. Cắm LLM vào `expand_fn` và planner để sinh query expansion thực sự theo
   system prompt AIC.
2. Thêm spatial tools (object detection, OCR zoom) và temporal grounding để
   xác minh top candidates theo vòng lặp ReAct.
3. Thêm Answer Generator cho VQA và feedback/refine cho KISC.
4. Đánh giá Recall@K, MRR và temporal tolerance trên ground-truth dataset.
5. Chỉ sau khi accuracy baseline ổn định mới triển khai queue-aware routing
   cho tối ưu concurrency/latency.
