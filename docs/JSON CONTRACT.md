# SCHEMA CONTRACT — AIC 2026

**Hợp đồng dữ liệu bắt buộc cho cả 2 team (Query ↔ Keyframe).**
Mọi thay đổi field phải được cả 2 team đồng ý. Version: `1.1`
(Cập nhật theo commit `feat(team1)` — indexing pipeline thực tế của )

---

## ⚠️ LƯU Ý QUAN TRỌNG NHẤT CHO QUERY TEAM (đọc trước)

**BEiT-3 hiện là VISION-ONLY (load bằng `timm`, không phải `transformers`).**
Lý do: không có checkpoint BEiT-3 image-text public nào load được — nên 
dùng bản vision-only.

**Hệ quả cho query-side:**
- ❌ **KHÔNG encode được TEXT query → không gian BEiT-3.** BEiT-3 chỉ encode ẢNH.
- ✅ Index `turbovec_beit3` CHỈ dùng cho **image-query** (user upload ảnh mẫu → tìm ảnh giống).
- ✅ Với **text query** (đa số case), CHỈ `turbovec_siglip` dùng được (SigLIP có cả text+image encoder chung không gian).

→ **Quy tắc:** `search_queries.visual[].text` (text) → CHỈ vào `turbovec_siglip`.
BEiT-3 để dành cho tính năng "search bằng ảnh" (nếu làm).

---

## 0. QUY ƯỚC KHÓA (BẮT BUỘC)

```
frame_id = f"{video_id}_{frame_idx:06d}"     # VD: "L21_V001_001234"
```
- `frame_idx` PAD 6 số 0.
- `frame_id` là khóa nối cả 3 kho: Turbovec / Elasticsearch / File.
- Lệch format = KHÔNG join được.

### Tên index (cố định)
| Kho | Index name | Encoder | Dim | Query text được? |
|-----|-----------|---------|-----|:----------------:|
| Vector SigLIP | `turbovec_siglip` | SigLIP (text+image) | 768 | ✅ Có |
| Vector BEiT-3 | `turbovec_beit3` | BEiT-3 (**vision-only, timm**) | *(theo timm model)* | ❌ Chỉ image-query |
| Caption (BM25) | `es_caption` | — | — | ✅ |
| OCR (BM25) | `es_ocr` | — | — | ✅ |
| ASR (BM25) | `es_asr` | — | — | ✅ |

> **Multi-encoder:** SigLIP + BEiT-3, mỗi encoder 1 index Turbovec riêng.
> - **SigLIP**: dùng cho text-query VÀ image-query (có text encoder).
> - **BEiT-3 (vision-only)**: CHỈ image-query. Text query KHÔNG dùng được index này.

### Pipeline cắt keyframe (cố định)
| Bước | Tool | Ghi chú |
|------|------|---------|
| Shot detection | **TransNetV2** | ⚠️ pip package ship Git LFS pointer, weights fetch riêng qua `scripts/fetch_transnetv2_weights.sh` |
| Chọn keyframe | (trong mỗi shot) | `keyframe_method` ghi rõ tool dùng |

> Mỗi shot do TransNetV2 cắt → chọn ≥1 keyframe đại diện. `shot_id` gắn frame với shot của nó.

### Nguồn sinh text field (thực tế build)
| Field | Nguồn | Model/Tool |
|-------|-------|-----------|
| `caption` | LVLM | Gemini |
| `ocr_text` | **Gemini vision** (KHÔNG phải OCR truyền thống) | `gemini-3.5-flash` (google-genai) |
| `asr_text` | ASR | Whisper |

### Biến môi trường API (BẮT BUỘC dùng đúng tên)
```
GEMINI_API_KEY      ← docker-compose.yml set tên NÀY
```
> ⚠️ KHÔNG dùng `GOOGLE_API_KEY` —  đã fix bug lẫn lộn 2 tên này. Cả query team cũng phải dùng `GEMINI_API_KEY` cho đồng bộ.

---

## 1. KEYFRAME — dữ liệu lưu vào 3 kho

### 1.1. → TURBOVEC (chỉ vector — 2 index riêng cho 2 encoder)

**Index `turbovec_siglip`:**
```json
{
  "frame_id": "L21_V001_001234",
  "vector": [0.15, -0.30, "...768 floats..."]
}
```

**Index `turbovec_beit3`:**
```json
{
  "frame_id": "L21_V001_001234",
  "vector": [0.08, 0.42, "...1024 floats..."]
}
```

> ⚠️ **Turbovec dùng `bit_width=4` (quantization có mất mát).** Vector đọc ra KHÔNG khớp 100% vector gốc. Khi verify/test: dùng tiêu chí "frame đúng thắng top-1", KHÔNG dùng ngưỡng score gần 1.0.

| Index | Field | Kiểu | Nguồn |
|-------|-------|------|-------|
| `turbovec_siglip` | `frame_id` | string (key) | — |
| `turbovec_siglip` | `vector` | float[768] | **SigLIP** image encoder(ảnh) |
| `turbovec_beit3` | `frame_id` | string (key) | — |
| `turbovec_beit3` | `vector` | float[?] ⚠️ dim theo timm model — **hỏi ** | **BEiT-3** (vision-only) image encoder(ảnh) |

> Cùng 1 keyframe → encode bằng CẢ 2 model → 2 vector → lưu 2 index. Nối bằng `frame_id`.

### 1.2. → ELASTICSEARCH (text + nhãn + metadata)
```json
{
  "frame_id": "L21_V001_001234",
  "video_id": "L21_V001",
  "frame_idx": 1234,
  "timestamp_sec": 41.13,

  "shot_id": "L21_V001_shot_012",
  "shot_start_sec": 39.5,
  "shot_end_sec": 44.2,
  "keyframe_method": "TransNetV2+DAKE",

  "embeddings": {
    "siglip": {"index": "turbovec_siglip", "dim": 768},
    "beit3":  {"index": "turbovec_beit3",  "dim": null, "note": "vision-only, dim theo timm model - hỏi "}
  },

  "caption": "a man wearing a red shirt rides a bicycle on the street at night",
  "ocr_text": "PHỞ HÀ NỘI ĐƯỜNG LÊ LỢI",
  "ocr_lang": "vi",
  "asr_text": "tối nay đường phố khá vắng",
  "asr_lang": "vi",

  "objects": [
    {"label": "person",  "confidence": 0.98, "color": "red",  "bbox": [0.30, 0.20, 0.50, 0.90]},
    {"label": "bicycle", "confidence": 0.95, "color": null,   "bbox": [0.28, 0.50, 0.55, 0.95]}
  ],

  "attributes": {
    "time_of_day": "night",
    "location_type": "outdoor street",
    "colors": ["black", "red", "yellow"]
  },

  "metadata": {
    "title": "60 Giây Tối - 26/07/2024",
    "channel_id": "UCRjzfa1E0gA50lvDQipbDMg",
    "publish_date": "26/07/2024",
    "keywords": ["HTV", "tin tức"]
  }
}
```

| Field | Kiểu | Search bằng | Bắt buộc |
|-------|------|-------------|:--------:|
| `frame_id` | string (key) | — | ✅ |
| `video_id` | string | trả UI | ✅ |
| `frame_idx` | int | suy prev/next | ✅ |
| `timestamp_sec` | float | trả UI | ✅ |
| `shot_id` | string | dedupe/clip | ✅ |
| `shot_start_sec` | float | build clip | ✅ |
| `shot_end_sec` | float | build clip | ✅ |
| `caption` | string (EN) | 🟡 BM25 | ✅ |
| `ocr_text` | string (VI) | 🟡 BM25 | ✅ |
| `ocr_lang` | string | — | ⬜ |
| `asr_text` | string (VI) | 🟡 BM25 | ✅ |
| `asr_lang` | string | — | ⬜ |
| `objects[].label` | string | 🔴 Term filter | ✅ |
| `objects[].confidence` | float | ngưỡng ≥0.5 | ✅ |
| `objects[].color` | string\|null | 🔴 Term boost | ⬜ |
| `objects[].bbox` | float[4] (0-1) | xác minh vị trí | ⬜ |
| `attributes.time_of_day` | enum: day/night/dawn/dusk | 🔴 Term boost | ✅ |
| `attributes.location_type` | string | 🔴 Term boost | ✅ |
| `attributes.colors` | string[] | 🔴 Term boost | ⬜ |
| `metadata.title` | string | filter/hiển thị | ⬜ |
| `metadata.channel_id` | string | filter | ⬜ |
| `metadata.publish_date` | string (DD/MM/YYYY) | filter | ⬜ |
| `metadata.keywords` | string[] | boost | ⬜ |

### 1.3. → FILE STORAGE (ảnh/video thật)
```
data/keyframes/{video_id}/{frame_id}.jpg
data/videos/{video_id}.mp4
```

---

## 2. QUERY — output query team → input retrieval team

```json
{
  "query_id": "q_20260726_7e812cab",
  "schema_version": "1.0",

  "task": {"type": "AVS", "confidence": 0.9},

  "query": {
    "original":   "tìm những cảnh người áo đỏ đi xe đạp ban đêm không có ô tô",
    "translated": "man in red shirt riding bicycle on street at night, no car"
  },

  "search_queries": {
    "visual": [
      {"id": "v0", "text": "man in red shirt riding bicycle at night", "weight": 1.0,  "focus": "original"},
      {"id": "v1", "text": "cyclist wearing red on city street night",  "weight": 0.9,  "focus": "subject_action"}
    ],
    "caption": [
      {"id": "c0", "text": "a man rides a bicycle on the street at night", "weight": 1.2}
    ],
    "ocr": [],
    "asr": []
  },

  "entities": [
    {"id": "e1", "canonical": "man",     "attributes": {"colors": ["red"]}, "verification": "soft"},
    {"id": "e2", "canonical": "bicycle", "attributes": {},                  "verification": "hard"}
  ],

  "constraints": {
    "hard":     ["bicycle"],
    "soft":     ["red shirt", "at night", "street"],
    "negative": ["car"]
  },

  "scene": {"time_of_day": "night", "location_type": "outdoor street"},

  "retrieval_plan": {
    "channels": {
      "visual":  {"enabled": true,  "index": "turbovec_siglip", "top_k": 100, "weight": 1.0},
      "caption": {"enabled": true,  "index": "es_caption",      "top_k": 50,  "weight": 0.4},
      "ocr":     {"enabled": false, "index": "es_ocr",          "top_k": 0,   "weight": 0.0},
      "asr":     {"enabled": false, "index": "es_asr",          "top_k": 0,   "weight": 0.0}
    },
    "fusion": {"method": "weighted_rrf", "rrf_k": 60, "final_top_k": 50},
    "rerank": {"enabled": true, "top_k": 20}
  },

  "meta": {"llm_model": "gemini-2.5-flash", "total_llm_calls": 2, "latency_ms": 8000}
}
```

| Field | Kiểu | Dùng | Bắt buộc |
|-------|------|------|:--------:|
| `query_id` | string | định danh | ✅ |
| `task.type` | enum: KIS/AVS/VQA/KISC | chọn chiến thuật | ✅ |
| `query.original` | string (VI) | lưu vết | ✅ |
| `query.translated` | string (EN) | lưu vết | ✅ |
| `search_queries.visual[].text` | string (EN) | 🔵 → SigLIP → vector | ✅ |
| `search_queries.visual[].weight` | float | fusion weight | ✅ |
| `search_queries.caption[].text` | string (EN) | 🟡 BM25 caption | ⬜ |
| `search_queries.ocr[].text` | string (VI) | 🟡 BM25 ocr | ⬜ |
| `search_queries.asr[].text` | string (VI) | 🟡 BM25 asr | ⬜ |
| `entities[].canonical` | string (EN) | 🔴 object match | ✅ |
| `entities[].verification` | enum: hard/soft | filter vs boost | ✅ |
| `entities[].attributes.colors` | string[] | 🔴 color boost | ⬜ |
| `constraints.hard` | string[] | 🔴 filter (phải có) | ✅ |
| `constraints.soft` | string[] | 🟢 boost | ⬜ |
| `constraints.negative` | string[] | 🔴 filter ngược (loại) | ⬜ |
| `scene.time_of_day` | enum | 🔴 boost | ⬜ |
| `scene.location_type` | string | 🔴 boost | ⬜ |
| `retrieval_plan.channels` | object | bật/tắt + weight | ✅ |
| `retrieval_plan.fusion` | object | RRF config | ✅ |

---

## 3. MAPPING: Query field → Keyframe field

**Bảng nối — code join dựa vào bảng này.**

| Query field | → Keyframe field | Cơ chế | Loại |
|-------------|------------------|--------|------|
| `search_queries.visual[].text` | `turbovec_siglip` vector | Vector (SigLIP text enc) | 🔵 search |
| *(chỉ nếu có ảnh mẫu)* image query | `turbovec_beit3` vector | Vector (BEiT-3, image-only) | 🔵 image-search |
| `search_queries.caption[].text` | `caption` | BM25 | 🟡 search |
| `search_queries.ocr[].text` | `ocr_text` | BM25 | 🟡 search |
| `search_queries.asr[].text` | `asr_text` | BM25 | 🟡 search |
| `entities[canonical, verification=hard]` | `objects[].label` | Term | 🔴 filter |
| `constraints.negative` | `objects[].label` | Term | 🔴 filter ngược |
| `entities[canonical, verification=soft]` | `objects[].label` | Term | 🟢 boost |
| `entities[].attributes.colors` | `objects[].color` | Term | 🟢 boost |
| `scene.time_of_day` | `attributes.time_of_day` | Term | 🟢 boost |
| `scene.location_type` | `attributes.location_type` | Term | 🟢 boost |

---

## 4. OUTPUT sau search (retrieval → UI)

```json
{
  "query_id": "q_20260726_7e812cab",
  "results": [
    {
      "rank": 1,
      "frame_id": "L21_V001_001234",
      "video_id": "L21_V001",
      "timestamp_sec": 41.13,
      "score": 0.02228,
      "keyframe_path": "data/keyframes/L21_V001/L21_V001_001234.jpg",
      "caption": "a man rides a bicycle at night",
      "evidence": {
        "matched_channels": ["vector", "caption"],
        "vector_score": 0.87,
        "bm25_score": 2.13,
        "objects": ["person", "bicycle"]
      }
    }
  ]
}
```

| Field | Kiểu | Nguồn |
|-------|------|-------|
| `rank` | int | thứ hạng cuối |
| `frame_id` | string | fusion |
| `video_id` | string | ES hydrate |
| `timestamp_sec` | float | ES hydrate |
| `score` | float | RRF fusion |
| `keyframe_path` | string | suy từ frame_id |
| `caption` | string | ES hydrate |
| `evidence.matched_channels` | string[] | luồng nào khớp |

---

## 5. FUSION config

```
method: "weighted_rrf"
RRF_score(frame) = Σ [ channel.weight × 1/(rrf_k + rank) ]
rrf_k = 60
final_top_k = 50
```

---

## 6. ENUM chuẩn (dùng chung)

| Enum | Giá trị hợp lệ |
|------|----------------|
| `task.type` | `KIS`, `AVS`, `VQA`, `KISC` |
| `verification` | `hard`, `soft` |
| `time_of_day` | `day`, `night`, `dawn`, `dusk` |
| `fusion.method` | `weighted_rrf`, `weighted_sum` |
| language | `vi`, `en`, `mixed` |

---


### Handoff gate (đã có)
`verify_index.py` là cổng bàn giao index từ Team 1 sang query team:
- Check self-retrieval Turbovec: "frame đúng thắng top-1" (KHÔNG dùng ngưỡng ~1.0 vì `bit_width=4` lossy).
- Query team chỉ bắt đầu test khi gate này pass.

---

## 8. CÂU HỎI CẦN CHỐT 

- [ ] **BEiT-3 dim** thực tế là bao nhiêu? (timm model nào?)
- [ ] BEiT-3 vision-only → có làm tính năng **image-query** không, hay tạm bỏ index `turbovec_beit3` với text query?
- [ ] `keyframe_method` giá trị chính xác? (TransNetV2 + gì để chọn frame trong shot?)
- [ ] Tên field ES thực tế có khớp contract không (`ocr_text`, `asr_text`, `objects[].label`...)?
- [ ] `objects` lấy từ đâu — YOLO/FasterRCNN hay Gemini? (commit chỉ nói OCR dùng Gemini)