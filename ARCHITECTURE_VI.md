# Kiến trúc Hệ thống — AIC 2026

Tài liệu này định nghĩa Kiến trúc Truy xuất Đa phương thức (Multimodal Retrieval Architecture) cho cuộc thi AI Challenge 2026, dựa trên hệ thống tham chiếu ("Cascaded Embedding-Reranking and Temporal-Aware Score Fusion") và các hướng dẫn chính thức từ Ban tổ chức.

---

## 1. Tổng quan Cuộc thi & Sự thay đổi Dữ liệu

Dữ liệu AIC 2026 thể hiện một sự dịch chuyển lớn từ **Surveillance** (camera an ninh cố định, tin tức truyền hình) sang **Sousveillance** (góc nhìn thứ nhất / Ego-centric từ thiết bị cá nhân như kính thông minh, camera hành trình).

**Hệ quả thực tế:**
- **Video rung lắc & Biến động:** Không thể dựa vào các khung hình tĩnh, sạch sẽ. Visual embeddings phải thật sự mạnh mẽ (robust).
- **Âm thanh nhiều tiếng ồn (Noisy Audio):** Khác với giọng MC truyền hình, âm thanh ego-centric lẫn tiếng gió, tiếng ồn môi trường và nhiều khoảng im lặng.
- **Ba Thách thức Cốt lõi:**
  1. **Semantic Gap:** Truy vấn của con người mang tính trừu tượng; pixel chỉ là dữ liệu thô.
  2. **Data Sparsity & Scale:** Tìm một đoạn clip 2 giây trong hàng trăm giờ video đòi hỏi một bộ lọc ban đầu cực kỳ nhanh.
  3. **Temporal Logic Constraints:** Thứ tự thời gian của các hành động rất quan trọng ("bước vào phòng rồi mới cởi mũ"). Tìm kiếm thông thường bỏ qua yếu tố này.

**Nhiệm vụ mới - KISC (Conversational KIS):** 
Dữ liệu 2026 giới thiệu bài toán Conversational Known-Item Search, bắt buộc phải sử dụng các Agent hội thoại. Các đội phải xây dựng hệ thống có khả năng tinh chỉnh truy vấn qua các cuộc hội thoại hỏi - đáp, thay vì chỉ trả về một danh sách kết quả tĩnh.

---

## 2. Các Khu vực Chức năng (GitNexus Clusters)

Cơ sở mã có **3 cụm chức năng chính** được xác định bằng phân tích tĩnh:

| Cụm | Vai trò |
|:---|:---|
| **Agents (A1-A6)** | Hệ thống đa agent điều phối việc suy luận, bộ nhớ và lập kế hoạch. |
| **Truy xuất (Retrieval)** | Lập chỉ mục video, kho lưu trữ TurboVec, kho lưu trữ Elasticsearch. |
| **Giao diện & Phản hồi (UI & Feedback)** | Phản hồi mức độ liên quan, giới hạn đa dạng (diversity caps), và khám phá/khai phá concept. |

### 🧩 A. Hệ thống 6-Agent (Đội 2)
Hệ thống truy xuất trực tuyến được điều khiển bởi sáu agent chuyên biệt:
- **A1 (Task Router):** Phân loại truy vấn (KIS, AVS, VQA, KISC) và điều hướng luồng thực thi.
- **A2 (Query Planner):** Tạo đối tượng ràng buộc kiểu (trọng số modality, thứ tự thời gian). Thực thi chạy thử (dry-run) bằng ES `_count` để tránh các truy vấn quá chặt (0 kết quả).
- **A3 (Concept Grounding):** Bộ nhớ ngữ nghĩa (semantic cache). Mở rộng concept thành các mô tả thị giác chi tiết.
- **A4 (Temporal Verifier):** Kiểm tra các ràng buộc thời gian ("bước vào phòng, rồi cởi mũ").
- **A5 (VLM Judge):** Cross-encoder reranker trên top-50 ứng viên, có quyền phủ quyết (hard veto).
- **A6 (Clarification Agent):** Dành cho KISC. Tính toán entropy trên tập ứng viên để hỏi người dùng một câu hỏi làm rõ tối ưu nhất.

### 🗄️ B. Truy xuất & Lưu trữ (Đội 1)
- **VideoIndexer:** Trình điều phối pipeline offline (Tách biệt CPU/GPU/API pool).
- **Kho lưu trữ Vector (FAISS/Turbovec):** Lưu trữ các visual embedding.
- **Kho lưu trữ Elasticsearch:** Lưu trữ siêu dữ liệu, OCR/ASR, **thời gian, địa điểm, và sự kiện âm thanh**.

### 💻 C. Giao diện & Phản hồi
- **Giới hạn Đa dạng (Diversity Cap):** Giới hạn kết quả trả về ≤2 sự kiện mỗi video trên trang đầu.
- **"More Like This" (Tìm ảnh tương tự):** Truy vấn bằng hình ảnh sử dụng vector có sẵn (không tốn chi phí model).
- **Phản hồi Rocchio:** Phản hồi mức độ liên quan để cập nhật vector truy vấn mà không chịu độ trễ của LLM.

---

## 3. Pipeline Kiến trúc Agentic

Chúng tôi đã triển khai **Pipeline Đa phương thức điều hướng bởi Agent (Agent-guided Multimodal Pipeline)** kết hợp với khả năng **Suy luận Sự kiện Thời gian** và khung **Suy luận Không-Thời gian (STAR)** dành cho VQA.

### Lập chỉ mục Offline (Đội 1)
1. **Decode Phân tán (Fan-Out Decoding):** `ffmpeg` giải mã các frame (chạy trên CPU pool) và track âm thanh một lần duy nhất.
2. **Gắn thẻ Sự kiện Âm thanh:** `BEATs` hoặc `CLAP` trích xuất sự kiện âm thanh (ví dụ: "tiếng giao thông", "tiếng nấu ăn") để cung cấp ưu tiên mạnh mẽ về vị trí/hoạt động khi ASR thất bại trên video egocentric.
3. **Phân đoạn Embedding-Drift:** Thay thế phát hiện ranh giới cảnh quay truyền thống. Chúng tôi phân đoạn các cảnh quay không qua chỉnh sửa bằng cách đo độ lệch (drift) giữa các visual embedding đã tính toán (Liên kết Cảnh Tương tự).
4. **Lập chỉ mục Siêu dữ liệu:** Mỗi sự kiện được lưu trong Elasticsearch cùng với các bộ lọc cắt tỉa (pruning) quan trọng: `date` (ngày), `hour_of_day` (giờ), `place_category` (loại địa điểm), và `audio_events` (sự kiện âm thanh).

### Truy xuất Online & VQA (Đội 2)
1. **Lập kế hoạch Truy vấn bằng Agent:** `A2` mở rộng truy vấn và phân bổ trọng số động giữa hình ảnh, OCR, và âm thanh. Nó sử dụng cơ chế **Chạy thử Mô hình Thế giới (World Model Dry-run)** bằng cách gọi ES `_count` để linh hoạt nới lỏng hoặc siết chặt các ràng buộc trước khi thực thi.
2. **Tìm kiếm Song song:** Hệ thống truy vấn Elasticsearch (Metadata + Văn bản) và TurboVec (Hình ảnh) đồng thời thông qua `asyncio.gather`, đi kèm với timeout riêng biệt cho từng công cụ (tool).
3. **Làm rõ KISC bằng Entropy:** Đối với truy vấn hội thoại, `A6` tính toán entropy của các khía cạnh (facet) như trong nhà/ngoài trời trên tập ứng viên. Sau đó nó hỏi người dùng một câu hỏi xoáy vào facet có entropy cao nhất để cắt đôi không gian tìm kiếm.
4. **Khung VQA STAR:** Đối với Video QA, một Planner điều phối các **Công cụ Thời gian** (mở rộng cửa sổ thời gian) và **Công cụ Không gian** (bao gồm một **công cụ ZOOM** để cắt ảnh và OCR vùng đó ở độ phân giải gốc đầy đủ).

---

## 4. Các Luồng Thực thi Chính

Các luồng gọi hàm quan trọng nhất trong codebase:

### Lập chỉ mục Offline
```
_build_and_run (video_indexer.py)
  └─ index_directory
       └─ index_video
            ├─ _sample_frame_indices — fixed-FPS sampling qua decord
            ├─ _transcribe       — Whisper chạy trên toàn bộ audio, rồi ghép vào frame
            └─ _extract_text     — OCRAgent.process(keyframe_path) cho từng keyframe
```

### Truy vấn Online
```
evaluate (eval.py)
  └─ search (inference.py)
       └─ _search_async
            ├─ rule_based_classify (classifier.py)
            ├─ dispatch (dispatcher.py)
            └─ _hybrid_rerank    — kết hợp điểm cosine TurboVec với điểm BM25 text
```

### Xác minh Chỉ mục
```
_check_frame (verify_index.py)
  └─ process (base_agent.py)
       └─ _run                   — xác nhận handoff chỉ mục từ Đội 1 sang Đội 2
```

---

## 5. Ngăn xếp Công nghệ Cốt lõi

### Khóa chính: `frame_id`
```
frame_id = "{video_id}_{frame_index:06d}"
Ví dụ:     L01_V001_000145
```
Dùng làm khóa trong **cả hai** TurboVec (qua file JSON sidecar) và Elasticsearch (là `_id`), đồng thời là tên file trên đĩa. Mọi thao tác join giữa các kho đều là tra cứu O(1) trên chuỗi này.

### Hai Cơ sở Dữ liệu

| | TurboVec (×2 instance) | Elasticsearch |
|:---|:---|:---|
| **Lưu trữ** | Vector thực (hình ảnh) | Văn bản (ASR + OCR) + metadata |
| **Kiểu chỉ mục** | ANN nén 4-bit (TurboQuant) | Chỉ mục đảo ngược (BM25) |
| **File trên đĩa** | `*.tvim` + `*.sidecar.json` | Docker volume `es_data` |
| **Kết quả trả về** | `[(frame_id, cosine_score)]` | `[(frame_id, BM25_score)]` |
| **Tại sao 2 TurboVec?** | SigLIP (1152-d) và BEiT-3 (768-d) có số chiều khác nhau; mỗi encoder một chỉ mục riêng |

### Bảng Thư viện Đầy đủ

| Mục đích | Thư viện / Model |
|:---|:---|
| Đọc và lấy mẫu frame | `decord` hoặc `ffmpeg` (CPU pool) |
| Visual embedding | `open-clip-torch`, SigLIP `ViT-SO400M-14-384` |
| Vision-only embedding | `timm`, BEiT-3 `beit3_base_patch16_224.in22k_ft_in1k` |
| Gắn thẻ Sự kiện Âm thanh | `BEATs` hoặc `CLAP` (GPU) |
| ASR | `openai-whisper`, `large-v3` |
| OCR | `google-genai`, Gemini 2.0/3.5 Flash |
| Kho Metadata & Văn bản | `elasticsearch>=8.13` (Gồm Thời gian, Địa điểm, Sự kiện Âm thanh) |
| Kho Vector | `turbovec` (Rust, 4-bit TurboQuant) |
| Reranker / Judge VLM | Gemini 2.5 Flash / Qwen3-VL (Chỉ Top-50) |
| Web UI | `streamlit` |

---

## 6. Phân chia File theo Đội

```
Đội 1 (Data & Indexing)           Đội 2 (NLP & Retrieval)
─────────────────────────         ────────────────────────────
Sở hữu:                           Sở hữu:
  src/agents/       ← dùng chung→   src/agents/ (chế độ text)
  src/retrieval/                    src/routing/
  scripts/                          src/inference.py
  configs/config.yaml               src/eval.py
                                    src/ui/app.py

Bàn giao:                         Tiêu thụ:
  data/index/turbovec/siglip.*      Kho TurboVec (đọc)
  data/index/turbovec/beit3.*       Chỉ mục Elasticsearch (đọc)
  Chỉ mục Elasticsearch             data/keyframes/**/*.jpg
  data/keyframes/**/*.jpg
```
