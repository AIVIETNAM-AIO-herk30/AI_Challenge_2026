# Kế hoạch Triển khai & Tích hợp — Vòng Sơ khảo

Tài liệu này vạch ra chiến lược triển khai, trách nhiệm của các đội và hợp đồng API (API contract) cần thiết để xây dựng hệ thống mà không gặp lỗi tích hợp. Tài liệu này đi kèm với `ARCHITECTURE_VI.md`.

**Nguyên tắc chỉ đạo:** Vòng sơ khảo chấm điểm dựa trên **độ chính xác (accuracy)**, không phải độ trễ (latency). Mỗi thành phần đều có phạm vi **Giai đoạn 1 (baseline)** (giải pháp đơn giản và đúng nhất) và **Giai đoạn 2 (refine)** (tối ưu hóa cho độ trễ/mở rộng). Không bắt đầu làm Giai đoạn 2 cho đến khi Giai đoạn 1 đã hoạt động trơn tru từ đầu đến cuối.

---

## 1. Trách nhiệm của các đội

Công việc được chia thành 4 phần cho 2 đội:

| Đội | Trách nhiệm |
|---|---|
| **Đội 1 — Dữ liệu & Lập chỉ mục** | Phần 1 (Lập chỉ mục Offline), Phần 2 (Tập dữ liệu Đánh giá & Metrics) |
| **Đội 2 — Truy xuất & Phục vụ** | Phần 3 (Thư viện Agent Dùng chung), Phần 4 (Truy xuất, Xếp hạng, Điều phối) |

> **QUAN TRỌNG**
> **Phần 3 (Thư viện Agent Dùng chung)** là phụ thuộc của cả Phần 1 và Phần 4. Giao diện (interface) của nó phải được chốt và xây dựng đầu tiên (dù chỉ là bản stub/mock) trước khi các phần khác bắt đầu phát triển.

---

## 2. Hợp đồng API (Đội 1 🤝 Đội 2)

Vì Đội 1 chuẩn bị cơ sở dữ liệu offline và Đội 2 truy vấn online, cả hai đội **BẮT BUỘC** phải thống nhất các quy tắc sau.

### 2.1 Core ID (Khóa chính)
Mỗi keyframe được trích xuất phải có một định danh thống nhất. Cả cơ sở dữ liệu vector (Turbovec/FAISS) và văn bản (Elasticsearch) đều phải dùng chung định dạng này.
- **Định dạng:** `{video_id}_{frame_index}`
- **Ví dụ:** `L01_V001_0145` (Video L01_V001, Frame 145)

### 2.2 Schema Tài liệu Elasticsearch
Khi Đội 1 chèn văn bản (ASR/OCR) vào Elasticsearch, họ phải dùng chính xác schema này để thuật toán tìm kiếm BM25 của Đội 2 có thể hoạt động:
```json
{
  "frame_id": "L01_V001_0145",
  "video_id": "L01_V001",
  "timestamp_seconds": 14.5,
  "ocr_text": "Trường Đại học Bách Khoa", 
  "asr_text": "Xin chào các bạn sinh viên"
}
```

### 2.3 Thỏa thuận Lưu trữ Vector (Turbovec / FAISS)
Cơ sở dữ liệu vector lưu các embedding tạo ra từ SigLIP và BEiT-3. Đội 1 phải đính kèm `frame_id` cùng với vector khi thêm vào cơ sở dữ liệu. Đối với FAISS, `embedding_id` (số nguyên) phải ánh xạ tới `frame_id` trong kho lưu trữ siêu dữ liệu (metadata store).

### 2.4 Đường dẫn Tệp tin (Dành cho UI)
Khi Đội 2 lấy được `frame_id`, Web UI cần hiển thị hình ảnh. Đội 1 phải lưu keyframe theo một cấu trúc thư mục chuẩn đoán được:
- **Đường dẫn:** `data/keyframes/{video_id}/{frame_id}.jpg`
- **Ví dụ:** `data/keyframes/L01_V001/L01_V001_0145.jpg`

---

## 3. Chi tiết Triển khai theo từng Phần

### Phần 1: Pipeline Lập chỉ mục Offline (Đội 1)
**Mục tiêu:** Chuyển đổi video thành các keyframe có thể tìm kiếm cùng siêu dữ liệu.
- **Lấy mẫu Khung hình (Frame Sampling):** Cố định FPS (`decord`).
- **Embeddings:** Dùng SigLIP `ViT-SO400M-14-384` qua `open_clip` (L2-normalized, 1152-d).
- **Âm thanh/Văn bản:** Whisper `large-v3` cho ASR, Gemini `1.5-flash` cho OCR.
- **Lưu trữ:** Dùng `faiss-cpu` (IndexIDMap kết hợp IndexFlatIP) và `pandas` (Parquet) cho siêu dữ liệu.

### Phần 2: Tập dữ liệu Đánh giá & Metrics (Đội 1)
**Mục tiêu:** Đo lường độ chính xác của hệ thống.
- **Ground Truth:** Tệp JSON ánh xạ các truy vấn tới `video_id` và `timestamp_sec` mong đợi.
- **Metrics (Chỉ số đo lường):** Recall@K (1, 5, 10) và Mean Reciprocal Rank (MRR).
- **Harness:** Script Python gọi hàm tìm kiếm để tự động tính toán các chỉ số này.

### Phần 3: Thư viện Agent Dùng chung (Đội 2)
**Mục tiêu:** Cung cấp các wrapper model cho cả quá trình lập chỉ mục và truy xuất.
- **VisualAgent:** Trả về mảng `(1152,)` kiểu float32 đã chuẩn hóa L2. Phải dùng **cùng một instance của model** cho cả truy vấn ảnh và truy vấn chữ để đảm bảo không gian embedding chung (joint embedding space).
- **ASRAgent:** Trả về dict dạng `{"text": str, "segments": [...]}`.
- **OCRAgent:** Trả về đoạn text trích xuất dạng `str` (hoặc chuỗi rỗng `""` nếu không có).

### Phần 4: Truy xuất & Xếp hạng tại thời điểm Truy vấn (Đội 2)
**Mục tiêu:** Truy xuất top-k chuẩn xác cho câu truy vấn văn bản.
- **Định dạng hàm tìm kiếm:** `search(query: str, config: dict, top_k: int = 10) -> list[dict]`
- **Phân loại (Classification):** Sử dụng rule-based classifier hiện có cho Giai đoạn 1.
- **Truy xuất:** Mã hóa câu truy vấn bằng `VisualAgent`, sau đó tìm kiếm trong FAISS.
- **Điều phối (Dispatcher):** Gọi hàm trực tiếp, đồng bộ ở Giai đoạn 1.

---

## 4. Chiến lược Tích hợp

**Xây dựng Tập con Vàng (Golden Mini-set):** Đừng đợi đến khi toàn bộ dữ liệu được lập chỉ mục xong.
1. Đội 1 lập chỉ mục trước 10-20 video và xuất các tệp FAISS và Parquet.
2. Đội 2 xây dựng và kiểm thử Phần 4 dựa trên tập con này.
3. Khi việc tích hợp đã được xác nhận thành công, tiến hành mở rộng (scale up) cho toàn bộ tập dữ liệu.
