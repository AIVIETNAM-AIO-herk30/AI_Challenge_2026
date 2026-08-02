# Lộ trình — Hệ thống Vòng Sơ Tuyển AIC 2026

## Bối cảnh

Team đang xây dựng theo một kiến trúc tham vọng (`ARCHITECTURE.md`): hệ thống truy xuất
hội thoại 6-agent hỗ trợ KIS, AVS (tìm kiếm tự do), VQA, và KISC (KIS hội thoại). Nhưng quy
chế vòng sơ tuyển chính thức (`Thong-tin-vong-So-tuyen-AIC2026.pdf`) chỉ định nghĩa **3 dạng
truy vấn, mỗi truy vấn được cung cấp trọn vẹn ngay từ đầu** — không có tìm kiếm tự do, không
có hỏi lại làm rõ. Trong khi đó, việc rà soát code phát hiện codebase đã lệch khỏi tài liệu
của chính nó ở nhiều chỗ (code chết, module mồ côi, comment lỗi thời, sai tên biến môi
trường), và thiết kế indexing offline hiện tại đang tính lại những gì cuộc thi đã cung cấp
sẵn cho batch 1 (shot detection, object detection).

Lộ trình này tái định phạm vi công sức kỹ thuật vào đúng những gì được chấm điểm, tái sử
dụng những gì ban tổ chức đã cung cấp thay vì tính lại từ đầu, và sắp xếp thứ tự công việc
sao cho có một pipeline KIS đúng, nộp được trước khi đầu tư vào phần Q&A/TRAKE khó hơn.

Các quyết định làm nền tảng cho lộ trình này: tập trung thuần vào **KIS / Q&A / TRAKE** (chưa
làm AVS/KISC), xây dựng và kiểm chứng pipeline trên **tập con L21 cục bộ (29 video)** trước
khi mở rộng ra phần còn lại của batch 1 (844 video nữa, chưa tải về).

---

## 1. Những gì thực sự được chấm điểm (căn cứ theo PDF chính thức)

Ba dạng truy vấn, mỗi truy vấn trả lời dựa trên một mô tả văn bản cố định, cung cấp trọn vẹn:

| Dạng | Nộp mỗi câu trả lời | Cách chấm |
|---|---|---|
| **Textual KIS** | `video_id, frame_id` | 1 nếu đúng video VÀ `frame_id ∈ [s,e]`, ngược lại 0 |
| **Q&A** | `video_id, frame_id, answer` | 1 nếu đúng video + đúng khoảng frame VÀ answer khớp ngữ nghĩa, ngược lại 0 |
| **TRAKE** | `video_id, frame_id_1..frame_id_N` | 0 nếu sai video; nếu đúng thì tính theo tỉ lệ số khoảnh khắc (trong N khoảnh khắc) khớp đúng khoảng — mỗi khoảng thường **hẹp dưới 10 frame** |

Tối đa **100 câu trả lời được xếp hạng cho mỗi truy vấn**. Điểm cuối cùng = trung bình của
`R@k` với `k ∈ {1,5,20,50,100}`, trong đó `R@k` = R-Score cao nhất trong k câu trả lời đầu
tiên. **Kỷ luật xếp hạng quan trọng không kém khả năng tìm đúng (recall)** — một câu trả lời
đúng nhưng nằm ở vị trí 80 sẽ có điểm thấp hơn nhiều so với cùng câu trả lời đó ở vị trí 3.
Điều này cần trực tiếp định hình cách hệ thống xây dựng danh sách 100 câu trả lời (xem §5,
Giai đoạn 2).

---

## 2. Thực trạng dữ liệu (đã kiểm chứng trực tiếp trên file, không chỉ dựa vào tài liệu)

- Batch 1 gồm **873 video** trải trên các nhóm L21–L30 (`media-info`/`objects`/
  `clip-features`/`map-keyframes` đều bao phủ đủ 873 video). Chỉ có **29 video của nhóm L21**
  là đã có sẵn file `.mp4` + keyframe `.jpg` thật sự ở máy cục bộ (`Videos_L21_a/`,
  `Keyframes_L21/`) — 844 video còn lại chưa có file media, cần tải về từ link Google Sheet
  trong PDF (ước lượng quy mô: 3.2GB cho 29 video → có thể 80–100GB+ cho toàn bộ 873 video).
- Ban tổ chức đã **cung cấp sẵn, theo từng keyframe**: ảnh keyframe đã trích xuất, `frame_idx`
  thật của video + mốc thời gian (qua file `map-keyframes-aic25-b1/{video_id}.csv`, các cột
  `n,pts_time,fps,frame_idx`), kết quả phát hiện vật thể bằng Faster R-CNN/OpenImages
  (`objects-aic25-b1/{video_id}/{n}.json`), và đặc trưng CLIP ViT-B/32
  (`clip-features-32-aic25-b1/{video_id}.npy`, shape `(số_keyframe, 512)`).
- **Ánh xạ quan trọng** (đã kiểm chứng thực nghiệm): file keyframe `NNN.jpg` đánh số
  **bắt đầu từ 1**, bằng với dòng CSV `n=NNN`. Cột `frame_idx` của dòng CSV đó chính là
  **số frame thật trong video** — đây là giá trị cần nộp, KHÔNG phải `NNN`. Mảng `.npy` đánh
  số **bắt đầu từ 0**, nên dòng `NNN-1` tương ứng với keyframe `NNN.jpg`. Bất kỳ code ingest
  nào cũng phải xử lý đúng độ lệch off-by-one này.
- PDF xác nhận: "dữ liệu thi chính thức là Video; keyframes/objects/CLIP features/metadata
  chỉ nhằm hỗ trợ xây dựng giải pháp mẫu." Vậy nên với L21, ta nên **tái sử dụng trực tiếp
  keyframes/objects/features đã cung cấp** thay vì tính lại từ đầu — điều này loại bỏ nhu cầu
  chạy TransNetV2 shot detection hay object detector riêng cho batch này. (Vẫn nên giữ một
  đường xử lý từ-đầu làm phương án dự phòng, phòng khi batch sau chỉ cung cấp video thô.)
- Chưa có file `queries.json` hay `eval_ground_truth.json` nào tồn tại — cần được tự soạn ở
  local (Team 1, theo `IMPLEMENTATION_PLAN.md` §Part 2), và cũng chưa tìm thấy định dạng file
  nộp bài chính thức nào trong PDF hay trên website cuộc thi — **cần xác nhận với ban tổ chức
  hoặc chờ file nộp bài mẫu**, không nên tự đoán.

---

## 3. Thực trạng codebase (đã kiểm chứng bằng cách đọc code thật, không chỉ đọc tài liệu)

**Thực tế và hoạt động được ngay hôm nay** (sẽ chạy được nếu có hạ tầng: Turbovec index +
Elasticsearch):
- `src/agents/{base_agent,visual_agent,asr_agent,ocr_agent,beit3_agent}.py` — các wrapper
  SigLIP/Whisper/Gemini-OCR/BEiT-3 thật sự hoạt động.
- `src/retrieval/{shot_detector,vector_store,es_store,video_indexer}.py` — một pipeline
  offline thật (TransNetV2 → mid-frame → SigLIP(+BEiT3) embed → Turbovec insert → Whisper ASR
  + Gemini OCR → ES bulk upsert). Cách tính `frame_index` là đúng (số frame thật từ chính quá
  trình decode của nó), nhưng lại tính lại toàn bộ từ video thô thay vì tái sử dụng những gì
  ban tổ chức đã tính sẵn cho batch 1.
- `src/inference.py` — `search()` thực hiện SigLIP text-embed → Turbovec ANN + ES BM25 → hợp
  nhất RRF → hydrate metadata → trả về `{video_id, frame_idx, timestamp_sec, score}`. Không
  còn phụ thuộc FAISS dù có một comment lỗi thời; khớp với thiết kế Turbovec+ES hiện tại.
- `src/agents/orchestrator.py` — bộ phân loại truy vấn rule-based thật (dựa từ khóa/regex,
  không phải LLM) + hợp nhất RRF + gom nhóm temporal clip. Đây là orchestration mà UI thực sự
  đang dùng.
- `src/ui/app.py` — một Streamlit app thật, hoạt động, được nối với `search()` + orchestrator.
- `src/eval.py` — harness Recall@K/MRR thật, nhưng tính **metric IR tổng quát, không phải
  công thức R-Score/R@k/Final-Score thật của cuộc thi**.
- `scripts/verify_index.py` — một cổng bàn giao Team1→Team2 thực sự hữu ích.

**Thật nhưng mồ côi** (đã xây xong nhưng không ai gọi tới):
- `src/query_processing/llm_pipeline.py` (2045 dòng) — một pipeline lập kế hoạch truy vấn
  bằng Gemini khá đồ sộ và thật sự hoạt động (T1–T6: preprocess → analyze → expand → build
  plan → fuse → rerank). Không được import bởi `inference.py`, `orchestrator.py`, hay
  `ui/app.py` ở bất kỳ đâu. Nơi duy nhất dùng nó là một trang test mock độc lập
  (`llm_ui_patch/ui/llm_phase_tester.py`). Đây là logic mở rộng truy vấn có giá trị, đang
  hoạt động nhưng bị bỏ không dùng.
- `src/routing/{classifier,dispatcher}.py` — code thật, nhưng không có tham chiếu nào khác
  trong toàn bộ codebase.

**Stub/legacy, có thể bỏ qua hoặc xóa sau:**
- `src/data_loader.py`, `src/model.py`, `src/train.py` — mọi method đều là
  `raise NotImplementedError`; các file này từng dùng để huấn luyện một MLP
  `QueryClassifier` mà bộ phân loại rule-based đang hoạt động đã khiến trở nên không cần thiết.

**Các mâu thuẫn tài liệu/code/config cần dọn dẹp** (chưa gấp, nhưng đang gây hiểu lầm):
- `README.md` mô tả định tuyến GPT-4o + FAISS + rerank BLIP-2 — không cái nào tồn tại thật.
- `README.md` ghi `export GOOGLE_API_KEY=...`; code thật lại yêu cầu `GEMINI_API_KEY`.
- Comment trong `configs/config.yaml` khẳng định `inference.py`/`eval.py` "vẫn tham chiếu
  FAISS VectorStore cũ và sẽ lỗi" — sai, cả hai đã được viết lại từ trước.
- `docs/JSON CONTRACT.md` và `docs/SCHEMA CONTRACT.md` là hai file giống hệt nhau từng byte,
  cả hai đều mô tả field/ES index `caption` không hề tồn tại trong `es_store.py` hay
  `video_indexer.py`.
- `ARCHITECTURE.md` định hình toàn bộ hệ thống xoay quanh AVS/KISC, vốn (theo §1 ở trên)
  không phải thứ vòng này chấm điểm.

---

## 4. Nguyên tắc chỉ đạo cho các giai đoạn dưới đây

**Không tính lại những gì ban tổ chức đã tính sẵn cho batch 1.** Tái sử dụng trực tiếp
keyframes/objects/CLIP-features đã cung cấp; chỉ bổ sung những gì thực sự còn thiếu (embedding
hình ảnh mạnh hơn qua SigLIP, ASR từ track âm thanh, và — có chọn lọc — OCR). Xây dựng đường
KIS trước vì đây là nền tảng chung cho cả 3 dạng truy vấn (Q&A và TRAKE đều vẫn cần "tìm đúng
video/frame" trước khi làm thêm bước riêng của chúng).

---

## 5. Lộ trình theo giai đoạn

### Giai đoạn 0 — Chỉnh lại phạm vi & sửa tài liệu gây hiểu lầm (Cả team, ~1 ngày)
- Rút gọn `ARCHITECTURE.md`/`README.md` để mô tả KIS/Q&A/TRAKE là phạm vi được chấm điểm;
  chuyển nội dung AVS/KISC sang một mục "vòng sau" rõ ràng thay vì xóa hẳn.
- Sửa `README.md`: `GOOGLE_API_KEY` → `GEMINI_API_KEY`, bỏ các tuyên bố về GPT-4o/FAISS/
  BLIP-2, và xóa/sửa comment lỗi thời về FAISS trong `configs/config.yaml`.
- Quyết định số phận của `src/routing/*` (mồ côi) và `src/{data_loader,model,train}.py`
  (stub) — khuyến nghị tạm để nguyên (không chặn tiến độ) và xem xét xóa sau.

### Giai đoạn 1 — Indexing offline tái sử dụng dữ liệu batch 1 đã cung cấp (Team 1)
Đường ingest mới cho tập con L21 (mở rộng `src/retrieval/video_indexer.py` hoặc thêm module
song song — quyết định thiết kế dành cho người triển khai):
1. Với mỗi `video_id` trong `Videos_L21_a`: duyệt qua
   `Keyframes_L21/keyframes/{video_id}/*.jpg`, lấy chỉ số `NNN`, tra cứu dòng `n=NNN` trong
   `map-keyframes-aic25-b1/{video_id}.csv` để lấy `frame_idx`/`pts_time` thật — xây dựng
   `frame_id = f"{video_id}_{frame_idx:06d}"` theo đúng quy ước đã có trong
   `docs/JSON CONTRACT.md` / `es_store.py`.
2. Encode mỗi keyframe bằng `VisualAgent` (SigLIP, đã có sẵn) — đây là tín hiệu vector chính,
   mạnh hơn CLIP-B/32 đã cung cấp. Có thể tùy chọn nạp thêm dòng `.npy` CLIP-B/32 tương ứng
   (`NNN-1`) làm tín hiệu phụ/dự phòng — không tốn chi phí inference vì đã được tính sẵn.
3. Tái sử dụng trực tiếp `objects-aic25-b1/{video_id}/{NNN}.json` thay vì chạy object
   detector riêng — parse các trường dạng chuỗi theo format TF-OD-API thành cấu trúc
   `objects[]` mà `es_store.py`/`docs/SCHEMA CONTRACT.md` đã kỳ vọng.
4. Chạy `ASRAgent` (Whisper) một lần cho mỗi video trên track âm thanh thật của `.mp4` (không
   có sẵn dữ liệu nào bao phủ phần này) và nối các đoạn transcript với keyframe theo mốc thời
   gian, như `IMPLEMENTATION_PLAN.md` §Part 1 đã mô tả.
5. **Gate OCR, không chạy trên mọi keyframe.** ~270 keyframe/video × 29 video ≈ 7.800 frame
   chỉ riêng L21; toàn bộ batch 873 video sẽ là ~235K frame — gọi Gemini cho mỗi frame là rủi
   ro thật về chi phí/rate-limit. Các lựa chọn cần đánh giá: bỏ hẳn OCR ở giai đoạn này (theo
   đúng khuyến nghị "tùy chọn cho Phase 1" của chính `IMPLEMENTATION_PLAN.md`), hoặc gate bằng
   một heuristic rẻ tiền (ví dụ: nhãn object detection gợi ý có chữ/biển hiệu) trước khi gọi
   Gemini.
6. Insert vào `TurbovecStore` + bulk-upsert vào `ElasticsearchStore`, khớp với schema mà
   `es_store.py` đã ghi/đọc (tối thiểu `frame_id, video_id, timestamp_seconds, ocr_text,
   asr_text`; mở rộng thêm `objects` nếu hữu ích cho việc lọc).
7. Chạy `scripts/verify_index.py` làm cổng bàn giao trước khi ai đó xây dựng dựa trên index.

### Giai đoạn 2 — Sửa đúng hợp đồng câu trả lời KIS (Team 2)
- Xác nhận `frame_idx` mà `inference.py` trả về luôn là **số frame thật của video** (đúng
  bất kể đường ingest của Giai đoạn 1 hay đường `video_indexer.py` cũ nạp dữ liệu vào index —
  cả hai phải thống nhất điều này).
- Xây dựng **danh sách top-100 đa dạng hóa**, không phải top-100 thô theo điểm số: với công
  thức chấm điểm R@k (§1), nộp 100 frame gần như trùng nhau từ cùng một shot sẽ lãng phí các
  vị trí đáng lẽ có thể dùng để đặt cược vào các khoảnh khắc ứng viên khác biệt. Tái sử dụng ý
  tưởng "Diversity Cap" đã có sẵn trong phần UI của `ARCHITECTURE.md` (≤N frame mỗi shot/
  video) làm logic xếp hạng nộp bài thực sự, không chỉ là bộ lọc hiển thị UI.
- Giai đoạn này là nền tảng chung mà cả Q&A và TRAKE đều xây dựng tiếp lên trên.

### Giai đoạn 3 — Sinh câu trả lời Q&A (thành phần mới, Team 2)
Hiện chưa có gì trong codebase sinh câu trả lời kiểu VQA (`orchestrator.py` chỉ thu thập
evidence). Cần xây dựng:
1. Chạy tìm kiếm KIS ở Giai đoạn 2 để lấy (các) frame/khoảng ứng viên tốt nhất cho mô tả sự
   kiện của truy vấn.
2. Với (các) ứng viên hàng đầu, gọi một VLM (Gemini, tái sử dụng pattern API-key/client đã có
   trong `ocr_agent.py`) với ảnh keyframe + câu hỏi, để sinh ra một chuỗi câu trả lời ngắn
   (tiếng Việt hoặc tiếng Anh, theo đúng quy định của PDF).
3. Trả về các bộ `(video_id, frame_id, answer)` được xếp hạng theo cùng cách đa dạng hóa như
   Giai đoạn 2.

### Giai đoạn 4 — Căn chỉnh nhiều sự kiện cho TRAKE (thành phần mới, phần khó nhất, Team 2)
Chưa có gì tương tự tồn tại. Thiết kế hai giai đoạn khớp với chính định nghĩa hai giai đoạn
của PDF:
1. **Giai đoạn truy xuất**: xem toàn bộ mô tả chuỗi sự kiện như một truy vấn kiểu KIS để tìm
   ra video khớp nhất duy nhất (tái sử dụng Giai đoạn 2).
2. **Giai đoạn căn chỉnh**: phân rã truy vấn thành N sự kiện con (một bước parse bằng LLM —
   các phase T2/T3 của `src/query_processing/llm_pipeline.py` là lựa chọn tự nhiên cho việc
   này, xem Giai đoạn 5) và, chỉ giới hạn trong các frame của video đã truy xuất được, tìm
   frame có điểm khớp cao nhất cho mỗi sự kiện con (ví dụ: độ tương đồng SigLIP giữa văn bản
   của mỗi sự kiện con và các keyframe của video đó, ưu tiên frame gần vị trí thời gian kỳ
   vọng nếu có nhiều lựa chọn ngang điểm). Lưu ý cảnh báo của PDF rằng các khoảng này thường
   **hẹp dưới 10 frame** — độ chính xác ở đây cần mức độ chi tiết hơn so với keyframe ở cấp độ
   shot thô có thể cung cấp; có thể cần decode thêm frame quanh mỗi keyframe ứng viên để căn
   chỉnh chặt chẽ hơn.

### Giai đoạn 5 — Kết nối pipeline LLM đang bị bỏ không (Team 2, có thể làm song song)
`src/query_processing/llm_pipeline.py` đã thực hiện mở rộng/viết lại truy vấn thật qua Gemini
nhưng lại không được kết nối với `inference.py`/`orchestrator.py`. Kết nối bước mở rộng của nó
vào đường tìm kiếm thực tế (thay vì mặc định `expand_fn or identity` hiện tại của
`orchestrator.py`) sẽ cải thiện recall cho cả KIS và bước phân rã sự kiện con của TRAKE, đồng
thời tận dụng được giá trị của code đã viết sẵn thay vì xây logic mở rộng truy vấn mới. Cần xử
lý kênh `caption`/`elasticsearch_caption` của nó — hoặc triển khai thật (thêm field `caption`
qua Gemini captioning, theo `docs/SCHEMA CONTRACT.md`) hoặc bỏ khỏi retrieval plan của nó vì
hiện chưa có field/index ES như vậy.

### Giai đoạn 6 — Ground truth cục bộ + metric chấm điểm thật (Team 1, có thể bắt đầu bất cứ lúc nào)
- Gán nhãn thủ công ~30–50 truy vấn (pha trộn KIS/Q&A/TRAKE) trên tập con L21 vào
  `data/raw/queries/eval_ground_truth.json` (schema theo `IMPLEMENTATION_PLAN.md` §2.8, mở
  rộng thêm cho dạng nhiều khoảng của TRAKE).
- Mở rộng `src/eval.py` để tính **đúng** metric của cuộc thi — R-Score theo từng dạng truy
  vấn chính xác như định nghĩa ở §2.1 của PDF, sau đó R@k cho k∈{1,5,20,50,100} và Final
  Score — thay vì Recall@K/MRR tổng quát, để các quyết định tinh chỉnh thực sự tối ưu đúng
  thứ được chấm điểm.

### Giai đoạn 7 — Mở rộng ra ngoài L21 (hậu cần, khi nào sẵn sàng)
Sau khi Giai đoạn 1–2 đã được kiểm chứng đúng trên tập con 29 video L21, tải về 844 video còn
lại (L22–L30) từ link Google Sheet trong PDF và chạy lại ingest của Giai đoạn 1 ở quy mô toàn
batch 1. Cần dự trù: ~80–100GB dung lượng lưu trữ video/keyframe, thời gian tính toán ASR
Whisper trên ~873 video, và (nếu vẫn giữ) chi phí API OCR theo tỉ lệ gate mà Giai đoạn 1 chốt.

---

## 6. Vấn đề cần chốt với ban tổ chức hoặc cả team (không thể tự đoán từ dữ liệu đã có)

- **Định dạng file nộp bài** — không có trong PDF hay website cuộc thi. Cần một file nộp bài
  mẫu được công bố hoặc xác nhận trực tiếp trước khi có thể chốt định dạng output của Giai
  đoạn 2 (CSV theo từng truy vấn? quy ước đặt tên? một file riêng cho mỗi dạng truy vấn?).
- Liệu batch 2 (được PDF nhắc đến là "sắp có") sẽ cung cấp keyframes/objects/CLIP features
  giống batch 1 hay chỉ có video thô — điều này ảnh hưởng đến việc "tái sử dụng dữ liệu cung
  cấp sẵn" của Giai đoạn 1 có còn áp dụng được không, hay đường xử lý từ-đầu bằng TransNetV2
  trong `video_indexer.py` hiện tại cần được giữ sẵn sàng production làm phương án dự phòng.

---

## 7. Cách kiểm chứng khi bắt đầu triển khai

- Sau Giai đoạn 1: chạy `scripts/verify_index.py` trên index của L21 — phải pass trước khi
  ai đó xây dựng tiếp trên đó.
- Sau Giai đoạn 2: chạy thử vài truy vấn KIS chọn lọc trên tập con L21 qua `src/ui/app.py` và
  kiểm tra thủ công rằng `frame_idx` trả về rơi đúng vào khoảnh khắc chính xác trong video
  (kiểm tra toàn bộ ánh xạ CSV trên thực tế, không chỉ trên lý thuyết).
- Sau Giai đoạn 6: chạy `src/eval.py` trên ground truth đã gán nhãn thủ công và báo cáo Final
  Score thật (không phải Recall@K/MRR) làm con số baseline thật của team.
- Sau Giai đoạn 3/4: kiểm tra thủ công một số câu trả lời Q&A và TRAKE đối chiếu với video gốc
  trước khi tin tưởng hoàn toàn vào output của pipeline tự động.
