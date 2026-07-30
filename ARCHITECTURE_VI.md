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

## 2. Các Nhóm Chức năng (GitNexus Clusters)

Codebase được tổ chức thành **3 functional clusters**:

| Cluster | Vai trò |
|:---|:---|
| **Agents** | Tất cả model wrappers (SigLIP, BEiT-3, Whisper, Gemini, BaseAgent) |
| **Retrieval** | Shot boundary detection, video indexing pipeline, FAISS/TurboVec store, Elasticsearch store |
| **Routing** | Query classifier, rule-based classify, dynamic dispatcher |

### 🧩 A. Agents
- **BaseAgent:** Base class cung cấp cơ chế kiểm soát đồng thời và đo lường độ trễ (latency).
- **VisualAgent:** Mã hóa cả **hình ảnh lẫn văn bản** vào chung một không gian embedding 1152-d thông qua **SigLIP ViT-SO400M-14-384**.
- **BEiT3Agent:** Bộ mã hóa chỉ dành cho hình ảnh 768-d sử dụng **BEiT-3 base_patch16_224**.
- **ASRAgent:** Chạy **Whisper large-v3** cục bộ; trích xuất văn bản từ âm thanh.
- **OCRAgent:** Gọi **API Gemini 2.0/3.5 Flash**; trích xuất văn bản từ hình ảnh.

### 🗄️ B. Retrieval & Storage
- **ShotDetector:** Bọc mô hình **TransNet V2** để phát hiện ranh giới cảnh quay (shot boundaries).
- **VideoIndexer:** Bộ điều phối offline pipeline.
- **Vector Store (FAISS/Turbovec):** Lưu trữ các embedding hình ảnh.
- **Elasticsearch Store:** Kho lưu trữ văn bản dạng chỉ mục đảo ngược (inverted-index) cho văn bản OCR/ASR.

### 🧠 C. Routing & Classification
- **rule_based_classify:** Bộ phân loại truy vấn theo từ khóa ở Giai đoạn 1.
- **QueryClassifier:** Bộ phân loại MLP cho Giai đoạn 2.
- **DynamicDispatcher:** Ánh xạ truy vấn tới các agent cụ thể và chạy chúng song song.

---

## 3. Đường ống Kiến trúc Agentic (Agentic Pipeline)

Hệ thống đã triển khai một **Agent-guided Multimodal Pipeline** (Đường ống Đa phương thức điều hướng bởi Agent) kết hợp với **Temporal Event Reasoning** (Suy luận Sự kiện theo thời gian).

### Agentic Pipeline khác gì so với Ad-hoc hoặc Zero-shot?
- **Hệ thống Zero-shot / Ad-hoc:** Thường hoạt động theo một chuỗi cứng nhắc duy nhất (VD: "Nhận câu truy vấn $\rightarrow$ biến thành vector $\rightarrow$ tìm trong database $\rightarrow$ trả về kết quả"). Chúng không thể tự sửa lỗi, không thể chia nhỏ các truy vấn phức tạp, và không biết đặt câu hỏi làm rõ.
- **Agentic Pipeline (Đường ống hướng Agent):** Hoạt động một cách linh hoạt. Khi nhận một truy vấn, bộ điều phối (thường là LLM) sẽ quyết định gọi *những sub-agent chuyên biệt nào* (Visual, ASR, OCR). Nó có thể mở rộng câu truy vấn, kết hợp nhiều loại hình dữ liệu (modalities) tùy theo ngữ cảnh. Quan trọng nhất, đối với bài toán KISC mới, nó có thể đo lường độ nhiễu (entropy) trong tập kết quả dự tuyển và **đặt câu hỏi ngược lại cho người dùng** để làm rõ thông tin trước khi đưa ra câu trả lời cuối cùng.

```mermaid
flowchart TD
    subgraph Team1 ["🗄️ Team 1: Data Preparation & Indexing (Offline)"]
        direction TB

        RAW["📹 Video AIC 2026"]

        RAW --> SD["🎬 ShotDetector\n(TransNet V2)"]
        SD -->|"Shot boundaries"| VI["⚙️ VideoIndexer\n(Pipeline Orchestrator)"]

        RAW -->|"Raw audio"| ASR["🎤 ASRAgent\n(Whisper large-v3)"]
        ASR -->|"segments"| VI

        VI -->|"Keyframe images"| SigLIP["🖼️ VisualAgent\n(SigLIP — 1152-d)"]
        VI -->|"Keyframe images"| BEiT3["🧠 BEiT3Agent\n(BEiT-3 — 768-d)"]
        VI -->|"Keyframe images"| OCR["📝 OCRAgent\n(Gemini 2.0/3.5 Flash)"]

        SigLIP -->|"float32 L2-normalised"| TVS[("💾 FAISS/TurboVec\nSigLIP Index")]
        BEiT3  -->|"float32 L2-normalised"| TVB[("💾 FAISS/TurboVec\nBEiT-3 Index")]

        VI -->|"temporal overlap"| ESW[("🔎 Elasticsearch\ntrường asr_text")]
        OCR -->|"ocr_text string"| ESO[("🔎 Elasticsearch\ntrường ocr_text")]
    end

    subgraph Team2 ["🧠 Team 2: NLP, Query Processing & Retrieval (Online)"]
        direction TB

        TQ["👤 User Text Query"]

        TQ --> LLM["🤖 Agent Router\nQuery Expansion & Routing"]

        LLM -->|"Visual weight"| TVS
        LLM -->|"Visual weight"| TVB
        LLM -->|"Text/Audio weights"| ESW
        LLM -->|"Text/Audio weights"| ESO
    end
```
