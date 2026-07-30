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

## 4. Schema Cơ sở Dữ liệu Chính xác (Điểm Hội tụ)

Cả hai file đều nằm trong `data/processed/embeddings/`.

### 4.1 `index.faiss`
- Kiểu: `faiss.IndexIDMap(faiss.IndexFlatIP(1152))` (Giai đoạn 1).
- Thêm dữ liệu bằng `index.add_with_ids(embeddings, ids)` với `ids` là các giá trị `embedding_id` từ bảng metadata.
- **Không bao giờ dùng ID tuần tự ngầm định của FAISS** — luôn gán tường minh qua `IndexIDMap` để join với metadata luôn chính xác dù các dòng được thêm không theo thứ tự.

### 4.2 `metadata.parquet`

| Cột | Kiểu dữ liệu | Ghi chú |
|---|---|---|
| `embedding_id` | `int64` | Khóa chính; khớp chính xác với id trong FAISS |
| `video_id` | `string` | VD: `L21_V001` |
| `frame_idx` | `int64` | Số frame trong video gốc |
| `timestamp_sec` | `float64` | **Đây là giá trị được dùng để chấm điểm** |
| `keyframe_path` | `string` | Đường dẫn đến file `.jpg` |
| `asr_text` | `string` (nullable) | Từ Whisper, ghép qua temporal overlap |
| `ocr_text` | `string` (nullable) | Từ Gemini, theo từng keyframe |
| `source_type` | `string` | `"surveillance"` / `"sousveillance"` |

### 4.3 Phần 4 (Đội 2) đọc và trả về gì
- Đọc: `index.faiss` + `metadata.parquet`, join theo `embedding_id`.
- Trả về (theo chữ ký hàm §2.4): `{"video_id", "frame_idx", "timestamp_sec", "score"}` — một tập con có chủ đích. Harness đánh giá và UI chỉ nhìn thấy định dạng này, không bao giờ thấy các cột Parquet thô.

---

## 5. Chiến lược Tích hợp

**Xây dựng Tập con Vàng (Golden Mini-set):** Đừng đợi đến khi toàn bộ dữ liệu được lập chỉ mục xong.
1. Đội 1 lập chỉ mục trước 10–20 video và xuất `index.faiss` + `metadata.parquet` chỉ cho tập con đó.
2. Đội 2 xây dựng và kiểm thử Phần 4 dựa trên tập con này trong khi Đội 1 tiếp tục lập chỉ mục song song.
3. File ground-truth của Đội 1 chỉ cần bao phủ tập con ban đầu — mở rộng sau khi toàn bộ dữ liệu được lập chỉ mục xong.

Điều này có nghĩa là lịch làm việc của hai đội không bị xếp hàng theo kiểu "Phần 1 xong hết rồi Phần 4 mới bắt đầu" — cả hai đội hội tụ tại hợp đồng chung (§2) và xác nhận nó sớm trên một phần nhỏ dữ liệu.

---

## 6. Bảng Thư viện & Model Tổng hợp

| Mục đích | Thư viện / Model | Giai đoạn |
|---|---|---|
| Lấy mẫu frame | `decord` | 1 |
| Lấy mẫu thích ứng | DAKE (dựa trên kích thước JPEG, không cần model) | 2 |
| Visual embedding | `open-clip-torch`, SigLIP `ViT-SO400M-14-384` | 1 |
| ASR | `openai-whisper`, `large-v3` | 1 |
| OCR | `google-generativeai`, Gemini `gemini-1.5-flash` | 1 (tùy chọn) |
| Chỉ mục vector | `faiss-cpu` (`IndexFlatIP` → `IndexIVFFlat`) | 1 → 2 |
| Kho metadata | `pandas` (Parquet) → Milvus/Elasticsearch | 1 → 2 |
| Tìm kiếm văn bản hybrid | `rank_bm25` → Elasticsearch | 1 → 2 |
| Captioning | *(không có)* → ReCap-style Gemini captioning | 2 |
| Phân loại truy vấn | rule-based (hiện có) → trained MLP | 1 → 2 |
| Xếp hạng clip | top-k theo điểm → Unified Clipping Algorithm | 1 → 2 |
