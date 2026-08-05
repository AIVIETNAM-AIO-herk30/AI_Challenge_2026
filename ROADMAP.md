# Roadmap — AIC 2026 Preliminary Round System

## Context

The team has been building toward an ambitious architecture (`ARCHITECTURE.md`): a 6-agent
conversational retrieval system supporting KIS, AVS (ad-hoc search), VQA, and KISC
(conversational KIS). But the official preliminary-round rules
(`Thong-tin-vong-So-tuyen-AIC2026.pdf`) define only **3 query types, each given fully upfront**
— no ad-hoc search, no back-and-forth clarification. Meanwhile, a code audit found the
codebase has drifted from its own docs in several places (dead code, orphaned modules,
stale comments, wrong env var names), and the offline indexing design re-derives work
(shot detection, object detection) that the competition already provides for batch 1.

This roadmap re-scopes engineering effort to what is actually graded, re-uses what's
already provided by the organizers instead of re-computing it, and sequences work so a
correct, submittable KIS pipeline exists before investing in the harder Q&A/TRAKE pieces.

Decisions this roadmap is built on: focus purely on **KIS / Q&A / TRAKE** (no AVS/KISC for
now), build and prove the pipeline against the **local L21 subset (29 videos)** before
scaling to the rest of batch 1 (844 more videos, not yet downloaded).

---

## 1. What's actually graded (ground truth from the official PDF)

Three query types, all answered against a fixed, fully-specified text query:

| Type | Submit per answer | Scored by |
|---|---|---|
| **Textual KIS** | `video_id, frame_id` | 1 if video matches AND `frame_id ∈ [s,e]` else 0 |
| **Q&A** | `video_id, frame_id, answer` | 1 if video+frame window match AND answer semantically matches, else 0 |
| **TRAKE** | `video_id, frame_id_1..frame_id_N` | 0 if wrong video; else fraction of the N event-windows hit (each window is often **<10 frames wide**) |

Up to **100 ranked answers per query**. Final score = average of `R@k` for
`k ∈ {1,5,20,50,100}`, where `R@k` = best R-Score among the first *k* submitted answers.
**Ranking discipline matters as much as recall** — a correct answer buried at position 80
scores far worse than the same answer at position 3. This should directly shape how the
system builds its top-100 answer list (see §5, Phase 2).

---

## 2. Data reality (verified against the actual files, not just docs)

- Batch 1 spans **873 videos** across groups L21–L30 (`media-info`/`objects`/`clip-features`/
  `map-keyframes` all cover all 873). Only **L21's 29 videos** have actual `.mp4` + keyframe
  `.jpg` files downloaded locally (`Videos_L21_a/`, `Keyframes_L21/`) — the other 844 videos'
  media files aren't present yet and would need downloading from the Google Sheet link in
  the PDF (rough scale: 3.2GB for 29 videos → likely 80–100GB+ for all 873).
- The organizers **already provide, per keyframe**: the extracted keyframe JPG, its real
  video `frame_idx` + timestamp (via `map-keyframes-aic25-b1/{video_id}.csv`, columns
  `n,pts_time,fps,frame_idx`), Faster R-CNN/OpenImages object detections
  (`objects-aic25-b1/{video_id}/{n}.json`), and CLIP ViT-B/32 features
  (`clip-features-32-aic25-b1/{video_id}.npy`, shape `(num_keyframes, 512)`).
- **Critical mapping** (confirmed empirically): keyframe file `NNN.jpg` is **1-indexed**,
  equal to CSV row `n=NNN`. That CSV row's `frame_idx` column is the **real video frame
  number** — this is what must be submitted, not `NNN`. The `.npy` feature array is
  **0-indexed**, so row `NNN-1` corresponds to keyframe `NNN.jpg`. Any ingestion code must
  thread this off-by-one correctly.
- The PDF confirms: "the official contest data is the Video; keyframes/objects/CLIP
  features/metadata are provided only to help build a sample solution." So for L21 we can
  and should **reuse the provided keyframes/objects/features** rather than re-deriving them
  — this eliminates the need to run TransNetV2 shot detection or a from-scratch object
  detector against this batch. (A from-scratch path should still exist as a fallback in
  case a future batch ships video-only.)
- No `queries.json` or `eval_ground_truth.json` exists anywhere yet — needs to be authored
  locally (Team 1, per `IMPLEMENTATION_PLAN.md` §Part 2), and no official submission-file
  format was found in the PDF or on the competition website — **this needs to be confirmed
  with organizers or a released example submission**, it's not something to guess at.

---

## 3. Codebase reality (verified by reading the actual code, not the docs)

**Real and working today** (would run given infra: Turbovec index + Elasticsearch):
- `src/agents/{base_agent,visual_agent,asr_agent,ocr_agent,beit3_agent}.py` — genuine
  SigLIP/Whisper/Gemini-OCR/BEiT-3 wrappers.
- `src/retrieval/{shot_detector,vector_store,es_store,video_indexer}.py` — a real offline
  pipeline (TransNetV2 → mid-frame → SigLIP(+BEiT3) embed → Turbovec insert → Whisper ASR +
  Gemini OCR → ES bulk upsert). Its `frame_index` computation is correct (real frame number
  from its own decode pass), but it re-derives everything from raw video instead of reusing
  what the organizers already computed for batch 1.
- `src/inference.py` — `search()` does SigLIP text-embed → Turbovec ANN + ES BM25 → RRF
  fusion → metadata hydration → returns `{video_id, frame_idx, timestamp_sec, score}`. No
  FAISS dependency despite a stale comment; matches the current Turbovec+ES design.
- `src/agents/orchestrator.py` — real rule-based task classifier (keyword/regex, not LLM)
  + RRF fusion + temporal clip grouping. This is the orchestration the UI actually uses.
- `src/ui/app.py` — a real, working Streamlit app wired to `search()` + the orchestrator.
- `src/eval.py` — real Recall@K/MRR harness, but computes **generic IR metrics, not the
  competition's actual R-Score/R@k/Final-Score formula**.
- `scripts/verify_index.py` — a genuinely useful Team1→Team2 handoff gate.

**Real but orphaned** (built, but nothing calls them):
- `src/query_processing/llm_pipeline.py` (2045 lines) — a substantial, real Gemini-based
  query-planning pipeline (T1–T6: preprocess → analyze → expand → build plan → fuse →
  rerank). Not imported by `inference.py`, `orchestrator.py`, or `ui/app.py` anywhere. Its
  only consumer is a standalone mock test page (`llm_ui_patch/ui/llm_phase_tester.py`).
  This is valuable, working query-expansion logic sitting unused.
- `src/routing/{classifier,dispatcher}.py` — real code, zero references anywhere else in
  the codebase.

**Stub/legacy, safe to ignore or delete later:**
- `src/data_loader.py`, `src/model.py`, `src/train.py` — every method is
  `raise NotImplementedError`; these were for training a `QueryClassifier` MLP that the
  working rule-based classifier has made unnecessary.

**Known doc/code/config disagreements to clean up** (not urgent, but actively misleading):
- `README.md` describes GPT-4o routing + FAISS + BLIP-2 reranking — none of that exists.
- `README.md` says `export GOOGLE_API_KEY=...`; the actual code requires `GEMINI_API_KEY`.
- `configs/config.yaml`'s inline comment claims `inference.py`/`eval.py` "still reference
  the old FAISS VectorStore and will fail" — false, both were already rewritten.
- `docs/JSON CONTRACT.md` and `docs/SCHEMA CONTRACT.md` are byte-identical duplicate files,
  both describing a `caption` field/ES index that doesn't exist in `es_store.py` or
  `video_indexer.py` anywhere.
- `ARCHITECTURE.md` frames the whole system around AVS/KISC, which (per §1 above) isn't
  what this round grades.

---

## 4. Guiding principle for the phases below

**Do not re-derive what the organizers already computed for batch 1.** Reuse provided
keyframes/objects/CLIP-features directly; only add what's genuinely missing (a stronger
visual embedding via SigLIP, ASR from the audio track, and — selectively — OCR). Build the
KIS path first since it's the foundation all three query types share (Q&A and TRAKE both
still need "find the right video/frame" before they can do their extra step).

---

## 5. Phased roadmap

### Phase 0 — Realign scope & fix misleading docs (Team-wide, ~1 day)
- Trim `ARCHITECTURE.md`/`README.md` to describe KIS/Q&A/TRAKE as the graded scope; move
  AVS/KISC content to a clearly-labeled "future rounds" section instead of deleting it.
- Fix `README.md`'s `GOOGLE_API_KEY` → `GEMINI_API_KEY`, remove the GPT-4o/FAISS/BLIP-2
  claims, and delete/correct `configs/config.yaml`'s stale FAISS comment.
- Decide fate of `src/routing/*` (orphaned) and `src/{data_loader,model,train}.py` (stub) —
  recommend leaving them untouched for now (not blocking) and revisiting deletion later.

### Phase 1 — Offline indexing that reuses provided batch-1 assets (Team 1)
New ingestion path for the L21 subset (extend `src/retrieval/video_indexer.py` or add a
sibling module — design decision for whoever implements this):
1. For each `video_id` in `Videos_L21_a`: iterate `Keyframes_L21/keyframes/{video_id}/*.jpg`,
   parse the `NNN` index, look up `map-keyframes-aic25-b1/{video_id}.csv` row `n=NNN` for the
   real `frame_idx`/`pts_time` — build `frame_id = f"{video_id}_{frame_idx:06d}"` per the
   existing convention in `docs/JSON CONTRACT.md` / `es_store.py`.
2. Embed each keyframe with `VisualAgent` (SigLIP, already built) — this is the primary
   vector signal, stronger than the provided CLIP-B/32 features. Optionally also load the
   provided `.npy` CLIP-B/32 row (`NNN-1`) as a cheap secondary/fallback signal — no model
   inference cost since it's already computed.
3. Reuse `objects-aic25-b1/{video_id}/{NNN}.json` directly instead of running any object
   detector — parse the string-encoded TF-OD-API fields into the `objects[]` shape
   `es_store.py`/`docs/SCHEMA CONTRACT.md` already expect.
4. Run `ASRAgent` (Whisper) once per video against the real `.mp4` audio track (nothing
   provided covers this) and join segments to keyframes by timestamp, as
   `IMPLEMENTATION_PLAN.md` §Part 1 already describes.
5. **Gate OCR, don't run it on every keyframe.** ~270 keyframes/video × 29 videos ≈ 7,800
   frames just for L21; the full 873-video batch would be ~235K frames — a Gemini call per
   frame is a real cost/rate-limit risk. Options to evaluate: skip OCR in this phase
   entirely (per `IMPLEMENTATION_PLAN.md`'s own "optional for Phase 1" guidance), or gate on
   a cheap heuristic (e.g., object-detection labels suggesting text/signage) before calling
   Gemini.
6. Insert into `TurbovecStore` + bulk-upsert into `ElasticsearchStore`, matching the schema
   `es_store.py` already writes/reads (`frame_id, video_id, timestamp_seconds, ocr_text,
   asr_text` at minimum; extend with `objects` if useful for filtering).
7. Run `scripts/verify_index.py` as the handoff gate before anyone builds against the index.

### Phase 2 — Correct the KIS answer contract (Team 2)
- Confirm `inference.py`'s returned `frame_idx` is always the **real video frame number**
  (true regardless of whether Phase 1's ingestion path or the legacy `video_indexer.py` path
  populated the index — both must agree on this).
- Build a **diversified top-100 list**, not a naive top-100 by raw score: given the R@k
  scoring formula (§1), submitting 100 near-duplicate frames from the same shot wastes
  slots that could instead hedge across distinct candidate moments. Reuse the "Diversity
  Cap" idea already sketched in `ARCHITECTURE.md`'s UI section (≤N frames per shot/video) as
  the actual submission-ranking logic, not just a UI display filter.
- This phase is the shared foundation Q&A and TRAKE both build on next.

### Phase 3 — Q&A answer generation (new component, Team 2)
Nothing in the codebase generates a VQA-style answer today (`orchestrator.py` only collects
evidence). Build:
1. Run Phase 2's KIS search to get the best candidate frame(s)/window for the query's
   event description.
2. For the top candidate(s), call a VLM (Gemini, reusing the API-key/client pattern already
   in `ocr_agent.py`) with the keyframe image + the question, to produce a short answer
   string (Vietnamese or English, per the PDF's rule).
3. Return `(video_id, frame_id, answer)` tuples ranked the same diversified way as Phase 2.

### Phase 4 — TRAKE multi-event alignment (new component, hardest piece, Team 2)
Nothing like this exists yet. Two-stage design matching the PDF's own two-stage definition:
1. **Retrieval stage**: treat the whole event-sequence description as a KIS-style query to
   find the single best-matching video (reuse Phase 2).
2. **Alignment stage**: decompose the query into its N sub-events (an LLM parsing step —
   `src/query_processing/llm_pipeline.py`'s T2/T3 phases are a natural fit here, see Phase 5)
   and, restricted to the retrieved video's frames only, find the best-scoring frame per
   sub-event (e.g., SigLIP similarity between each sub-event's text and that video's
   keyframes, taking the frame nearest the expected temporal position if there are ties).
   Note the PDF's warning that these windows are often **<10 frames wide** — precision here
   needs finer granularity than the coarse shot-level keyframes may offer; may need to
   decode extra frames around each candidate keyframe for tighter alignment.

### Phase 5 — Wire up the orphaned LLM query pipeline (Team 2, parallel-able)
`src/query_processing/llm_pipeline.py` already does real query expansion/rewriting via
Gemini but is disconnected from `inference.py`/`orchestrator.py`. Wiring its expansion step
into the actual search path (instead of `orchestrator.py`'s current `expand_fn or identity`
default) should improve recall on both KIS and TRAKE's sub-event decomposition, and gets
value out of already-written code rather than building new query-expansion logic. Needs its
`caption`/`elasticsearch_caption` channel either implemented (add a `caption` field via
Gemini captioning, per `docs/SCHEMA CONTRACT.md`) or stripped from its retrieval plan since
no such ES field/index currently exists.

### Phase 6 — Local ground truth + real scoring metric (Team 1, can start anytime)
- Hand-label ~30–50 queries (mix of KIS/Q&A/TRAKE) against the L21 subset into
  `data/raw/queries/eval_ground_truth.json` (schema per `IMPLEMENTATION_PLAN.md` §2.8,
  extended with TRAKE's multi-window shape).
- Extend `src/eval.py` to compute the **actual** competition metric — R-Score per query type
  exactly as defined in §2.1 of the PDF, then R@k for k∈{1,5,20,50,100} and Final Score —
  instead of generic Recall@K/MRR, so tuning decisions are optimizing the thing that's
  actually graded.

### Phase 7 — Scale out beyond L21 (logistics, whenever ready)
Once Phases 1–2 are proven correct on the 29-video L21 subset, download the remaining 844
videos (L22–L30) from the Google Sheet link in the PDF and re-run Phase 1's ingestion at
full batch-1 scale. Budget for: ~80–100GB of video/keyframe storage, Whisper ASR compute
time across ~873 videos, and (if kept) OCR API costs at whatever gating rate Phase 1 lands
on.

---

## 6. Open items to resolve with organizers or the team (not guessable from what's provided)

- **Submission file format** — not in the PDF or on the competition website. Needs a
  released example submission or direct confirmation before Phase 2's output format can be
  finalized (CSV per query? naming convention? one file per query type?).
- Whether batch 2 (mentioned in the PDF as "coming soon") will ship keyframes/objects/CLIP
  features the same way batch 1 did, or video-only — affects whether Phase 1's "reuse
  provided assets" shortcut generalizes or whether the from-scratch TransNetV2 path in the
  existing `video_indexer.py` needs to stay production-ready as the fallback.

---

## 7. Verification approach once implementation starts

- After Phase 1: run `scripts/verify_index.py` against the L21 index — must pass before
  anyone builds on top.
- After Phase 2: run a handful of hand-picked KIS queries against the L21 subset through
  `src/ui/app.py` and manually confirm the returned `frame_idx` lands in the visually
  correct moment of the video (sanity check the CSV mapping end-to-end, not just in theory).
- After Phase 6: run `src/eval.py` against the hand-labeled ground truth and report the
  actual Final Score (not Recall@K/MRR) as the team's real baseline number.
- After Phase 3/4: spot-check a handful of Q&A and TRAKE answers by hand against the source
  video before trusting the automated pipeline's output.
