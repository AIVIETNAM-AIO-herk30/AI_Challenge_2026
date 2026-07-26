"""Streamlit UI để kiểm thử từng phase của AIC 2026 Query Pipeline.

Chạy từ thư mục gốc project:
    py -3.12 -m streamlit run ui/llm_phase_tester.py

Trang này KHÔNG phụ thuộc visual_agent, beit3_agent, vector index hoặc
Elasticsearch. T5 dùng mock retrieval cục bộ để kiểm tra contract/fusion.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import streamlit as st


# -----------------------------------------------------------------------------
# Import pipeline mà không phụ thuộc cách project đang đóng gói src/.
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

_IMPORT_ERROR: Exception | None = None
try:
    # Khuyến nghị: src/query_processing/llm_pipeline.py
    from query_processing.llm_pipeline import (  # type: ignore
        GeminiClient,
        ProcessedQuery,
        T2Analysis,
        T3ExpandedItem,
        analyze_query,
        build_processed_query,
        expand_query,
        preprocess,
        rerank_candidates,
        retrieve_and_fuse,
    )
except Exception as first_error:
    try:
        # Hỗ trợ trường hợp src là package.
        from src.query_processing.llm_pipeline import (  # type: ignore
            GeminiClient,
            ProcessedQuery,
            T2Analysis,
            T3ExpandedItem,
            analyze_query,
            build_processed_query,
            expand_query,
            preprocess,
            rerank_candidates,
            retrieve_and_fuse,
        )
    except Exception:
        try:
            # Hỗ trợ file standalone ở root project.
            from aic_query_pipeline import (  # type: ignore
                GeminiClient,
                ProcessedQuery,
                T2Analysis,
                T3ExpandedItem,
                analyze_query,
                build_processed_query,
                expand_query,
                preprocess,
                rerank_candidates,
                retrieve_and_fuse,
            )
        except Exception as final_error:
            _IMPORT_ERROR = RuntimeError(
                "Không import được LLM pipeline. Hãy đặt file tại "
                "src/query_processing/llm_pipeline.py hoặc root/aic_query_pipeline.py. "
                f"Lỗi đầu: {first_error}; lỗi cuối: {final_error}"
            )


st.set_page_config(
    page_title="AIC 2026 · LLM Phase Tester",
    page_icon="🧪",
    layout="wide",
)

if _IMPORT_ERROR is not None:
    st.error(str(_IMPORT_ERROR))
    st.code(
        """AI_Challenge_2026/
├── src/
│   └── query_processing/
│       ├── __init__.py
│       └── llm_pipeline.py
└── ui/
    └── llm_phase_tester.py""",
        language="text",
    )
    st.stop()


# -----------------------------------------------------------------------------
# Mock data cho T5/T6 khi Team 2 chưa hoàn thiện retrieval.
# -----------------------------------------------------------------------------
DEFAULT_MOCK_CANDIDATES: list[dict[str, Any]] = [
    {
        "frame_id": "mock_001",
        "video_id": "video_mock_01",
        "timestamp": 12.4,
        "score": 0.82,
        "caption": "A man in a red shirt rides a bicycle on a city street at night.",
        "ocr": "CITY CENTER",
        "asr": "He is riding toward the intersection.",
        "objects": ["man", "red shirt", "bicycle", "street"],
    },
    {
        "frame_id": "mock_002",
        "video_id": "video_mock_01",
        "timestamp": 26.1,
        "score": 0.74,
        "caption": "A cyclist wearing a dark jacket waits beside a traffic light.",
        "ocr": "STOP",
        "asr": "The light is still red.",
        "objects": ["cyclist", "bicycle", "traffic light"],
    },
    {
        "frame_id": "mock_003",
        "video_id": "video_mock_02",
        "timestamp": 8.8,
        "score": 0.69,
        "caption": "A woman prepares vegetables in a bright indoor kitchen.",
        "ocr": "",
        "asr": "Add the onions to the pan.",
        "objects": ["woman", "vegetables", "kitchen", "pan"],
    },
    {
        "frame_id": "mock_004",
        "video_id": "video_mock_03",
        "timestamp": 41.0,
        "score": 0.65,
        "caption": "Several people walk along a sunny beach near the sea.",
        "ocr": "BEACH",
        "asr": "The weather is beautiful today.",
        "objects": ["people", "beach", "sea"],
    },
    {
        "frame_id": "mock_005",
        "video_id": "video_mock_04",
        "timestamp": 17.3,
        "score": 0.61,
        "caption": "A red motorcycle is parked outside a convenience store at dusk.",
        "ocr": "OPEN 24H",
        "asr": "",
        "objects": ["motorcycle", "store", "red"],
    },
]

PHASE_KEYS = {
    "T1": ["cleaned_query", "analysis", "expansions", "processed", "fused", "reranked"],
    "T2": ["analysis", "expansions", "processed", "fused", "reranked"],
    "T3": ["expansions", "processed", "fused", "reranked"],
    "T4": ["processed", "fused", "reranked"],
    "T5": ["fused", "reranked"],
    "T6": ["reranked"],
}


def _clear_from(phase: str) -> None:
    for key in PHASE_KEYS[phase]:
        st.session_state.pop(key, None)
    latencies = st.session_state.setdefault("phase_latency_ms", {})
    phase_number = int(phase[1])
    for key in list(latencies):
        if key.startswith("T") and int(key[1]) >= phase_number:
            latencies.pop(key, None)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _secret_or_env(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, ""))


def _client(api_key: str) -> GeminiClient:
    """Tạo lại client chỉ khi API key thay đổi."""
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    if st.session_state.get("client_fingerprint") != fingerprint:
        old_client = st.session_state.get("gemini_client")
        if old_client is not None:
            try:
                old_client.close()
            except Exception:
                pass
        st.session_state.gemini_client = GeminiClient(api_key=api_key)
        st.session_state.client_fingerprint = fingerprint
    return st.session_state.gemini_client


def _require_api_key(api_key: str) -> bool:
    if api_key.strip():
        return True
    st.error("T2, T3 và T6 cần GEMINI_API_KEY. Nhập key ở sidebar hoặc cấu hình secrets.")
    return False


def _history_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _as_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_as_json(item) for item in value]
    return value


def _parse_candidate_pool(raw: str) -> list[dict[str, Any]]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("Mock candidates phải là một JSON array.")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate #{index} phải là JSON object.")
        if not str(item.get("frame_id", "")).strip():
            raise ValueError(f"Candidate #{index} thiếu frame_id.")
        normalized = dict(item)
        normalized["frame_id"] = str(normalized["frame_id"])
        normalized["score"] = float(normalized.get("score", 0.0))
        output.append(normalized)
    return output


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _candidate_text(candidate: dict[str, Any]) -> str:
    values = [
        candidate.get("caption", ""),
        candidate.get("ocr", ""),
        candidate.get("asr", ""),
        " ".join(map(str, candidate.get("objects", []))),
    ]
    return " ".join(map(str, values))


def _mock_search_factory(
    candidate_pool: list[dict[str, Any]],
) -> Callable[[str, int], list[dict[str, Any]]]:
    """Mock search cục bộ để test T5 contract và fusion.

    Đây không phải vector retrieval. Score chỉ dựa trên lexical overlap, base score
    và một jitter ổn định theo query/frame để các subquery có ranking khác nhau.
    """

    def search(query_text: str, top_k: int) -> list[dict[str, Any]]:
        query_tokens = _tokens(query_text)
        results: list[dict[str, Any]] = []

        for candidate in candidate_pool:
            doc_tokens = _tokens(_candidate_text(candidate))
            union = query_tokens | doc_tokens
            overlap = len(query_tokens & doc_tokens) / max(1, len(union))
            base_score = max(0.0, min(1.0, float(candidate.get("score", 0.0))))
            digest = hashlib.sha256(
                f"{query_text}|{candidate['frame_id']}".encode("utf-8")
            ).hexdigest()
            jitter = int(digest[:8], 16) / 0xFFFFFFFF
            mock_score = 0.70 * overlap + 0.25 * base_score + 0.05 * jitter

            results.append(
                {
                    **candidate,
                    "score": round(mock_score, 6),
                    "mock_overlap": round(overlap, 6),
                }
            )

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

    return search


def _status_panel() -> None:
    columns = st.columns(6)
    states = [
        ("T1", "cleaned_query"),
        ("T2", "analysis"),
        ("T3", "expansions"),
        ("T4", "processed"),
        ("T5", "fused"),
        ("T6", "reranked"),
    ]
    latencies = st.session_state.get("phase_latency_ms", {})
    for column, (phase, key) in zip(columns, states, strict=True):
        done = key in st.session_state
        label = "DONE" if done else "WAIT"
        latency = latencies.get(phase)
        detail = f"{latency} ms" if latency is not None else "—"
        column.metric(phase, label, detail)


def _show_candidates(candidates: list[dict[str, Any]], score_field: str) -> None:
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        rows.append(
            {
                "rank": rank,
                "frame_id": candidate.get("frame_id"),
                score_field: candidate.get(score_field),
                "video_id": candidate.get("video_id"),
                "timestamp": candidate.get("timestamp"),
                "caption": candidate.get("caption", ""),
                "reason": candidate.get("rerank_reason", ""),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Header và cấu hình.
# -----------------------------------------------------------------------------
st.title("🧪 AIC 2026 · LLM Phase Tester")
st.caption(
    "Test T1 → T6 độc lập. Không gọi agents, vector index hoặc Elasticsearch của Team 2."
)

with st.sidebar:
    st.header("Cấu hình")
    default_key = _secret_or_env("GEMINI_API_KEY")
    api_key = st.text_input(
        "GEMINI_API_KEY",
        value=default_key,
        type="password",
        help="Không commit key vào GitHub.",
    )
    analyze_model = st.selectbox(
        "Model T2",
        ["gemini-3.6-flash", "gemini-3.6-flash"],
        index=0,
    )
    expand_model = st.selectbox(
        "Model T3",
        ["gemini-3.6-flash", "gemini-3.6-flash"],
        index=0,
    )
    rerank_model = st.selectbox(
        "Model T6",
        ["gemini-3.6-flash", "gemini-3.6-flash"],
        index=0,
    )

    if st.button("Xóa toàn bộ kết quả", use_container_width=True):
        for key in [
            "cleaned_query",
            "analysis",
            "expansions",
            "processed",
            "fused",
            "reranked",
            "phase_latency_ms",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

query = st.text_area(
    "Query",
    value="người đàn ông mặc áo đỏ đi xe đạp trên đường phố vào ban đêm",
    height=90,
)
history_raw = st.text_area(
    "History, mỗi dòng một lượt trước đó (tùy chọn)",
    value="",
    height=70,
)
history = _history_lines(history_raw)

_status_panel()

st.info(
    "Luồng test: chạy T1, sau đó T2, T3 và T4. T5 dùng mock candidates để test "
    "fusion. T6 dùng metadata mock để test Gemini rerank."
)

phase_t1, phase_t2, phase_t3, phase_t4, phase_t5, phase_t6 = st.tabs(
    [
        "T1 · Preprocess",
        "T2 · Analyze",
        "T3 · Expand",
        "T4 · Build schema",
        "T5 · Mock retrieval",
        "T6 · Rerank",
    ]
)


# -----------------------------------------------------------------------------
# T1
# -----------------------------------------------------------------------------
with phase_t1:
    st.subheader("T1 — deterministic preprocessing")
    st.write("Chuẩn hóa Unicode, xóa control character, gộp khoảng trắng và dấu câu lặp.")

    if st.button("Chạy T1", key="run_t1", type="primary"):
        _clear_from("T1")
        started = time.perf_counter()
        try:
            cleaned = preprocess(query)
            if not cleaned:
                raise ValueError("Query rỗng sau preprocess.")
            st.session_state.cleaned_query = cleaned
            st.session_state.setdefault("phase_latency_ms", {})["T1"] = _elapsed_ms(started)
        except Exception as exc:
            st.exception(exc)

    if "cleaned_query" in st.session_state:
        st.success("T1 hoàn tất")
        st.code(st.session_state.cleaned_query, language="text")


# -----------------------------------------------------------------------------
# T2
# -----------------------------------------------------------------------------
with phase_t2:
    st.subheader("T2 — classify, translate, extract")
    st.write("Một Gemini call để phân loại task, dịch query và trích entities/temporal/negation.")

    if "cleaned_query" not in st.session_state:
        st.warning("Chạy T1 trước.")

    if st.button(
        "Chạy T2",
        key="run_t2",
        type="primary",
        disabled="cleaned_query" not in st.session_state,
    ):
        if _require_api_key(api_key):
            _clear_from("T2")
            started = time.perf_counter()
            try:
                client = _client(api_key)
                client.reset_usage()
                analysis: T2Analysis = analyze_query(
                    query=st.session_state.cleaned_query,
                    client=client,
                    history=history,
                    model=analyze_model,
                )
                st.session_state.analysis = analysis
                st.session_state.setdefault("phase_latency_ms", {})["T2"] = _elapsed_ms(started)
            except Exception as exc:
                st.exception(exc)

    if "analysis" in st.session_state:
        analysis = st.session_state.analysis
        col1, col2, col3 = st.columns(3)
        col1.metric("Task type", analysis.task_type.value)
        col2.metric("Language", analysis.language)
        col3.metric("Ambiguous", str(analysis.ambiguous))
        if analysis.clarification_needed:
            st.warning(analysis.clarification_needed)
        st.json(_as_json(analysis), expanded=True)


# -----------------------------------------------------------------------------
# T3
# -----------------------------------------------------------------------------
with phase_t3:
    st.subheader("T3 — semantic expansion")
    st.write("Sinh các mô tả close-up, wide-shot, action và background để tăng recall.")

    if "analysis" not in st.session_state:
        st.warning("Chạy T2 trước.")

    if st.button(
        "Chạy T3",
        key="run_t3",
        type="primary",
        disabled="analysis" not in st.session_state,
    ):
        if _require_api_key(api_key):
            _clear_from("T3")
            started = time.perf_counter()
            try:
                client = _client(api_key)
                analysis = st.session_state.analysis
                expansions: list[T3ExpandedItem] = expand_query(
                    translated_en=analysis.translated_en,
                    entities=analysis.entities,
                    task_type=analysis.task_type,
                    client=client,
                    model=expand_model,
                )
                st.session_state.expansions = expansions
                st.session_state.setdefault("phase_latency_ms", {})["T3"] = _elapsed_ms(started)
            except Exception as exc:
                st.exception(exc)

    if "expansions" in st.session_state:
        expansions = st.session_state.expansions
        st.success(f"T3 hoàn tất: {len(expansions)} expansion")
        st.dataframe(
            [item.model_dump(mode="json") for item in expansions],
            use_container_width=True,
            hide_index=True,
        )


# -----------------------------------------------------------------------------
# T4
# -----------------------------------------------------------------------------
with phase_t4:
    st.subheader("T4 — build ProcessedQuery")
    st.write("Đóng gói T2/T3 thành JSON contract để bàn giao cho retrieval của Team 2.")

    if "analysis" not in st.session_state:
        st.warning("Chạy T2 trước. T3 có thể bỏ qua nếu muốn chỉ dùng query gốc.")

    if st.button(
        "Chạy T4",
        key="run_t4",
        type="primary",
        disabled="analysis" not in st.session_state,
    ):
        _clear_from("T4")
        started = time.perf_counter()
        try:
            analysis = st.session_state.analysis
            expansions = st.session_state.get("expansions", [])
            client = st.session_state.get("gemini_client")
            usage = client.usage() if client is not None else {
                "calls": 0,
                "total": 0,
            }
            processed: ProcessedQuery = build_processed_query(
                original_query=query,
                analysis=analysis,
                expansions=expansions,
                llm_model=f"{analyze_model}, {expand_model}",
                total_llm_calls=int(usage.get("calls", 0)),
                total_tokens=int(usage.get("total", 0)),
                latency_ms=sum(st.session_state.get("phase_latency_ms", {}).values()),
                phase_latency_ms=dict(st.session_state.get("phase_latency_ms", {})),
            )
            t4_latency = _elapsed_ms(started)
            st.session_state.setdefault("phase_latency_ms", {})["T4"] = t4_latency
            processed.meta.phase_latency_ms = dict(
                st.session_state.get("phase_latency_ms", {})
            )
            processed.meta.latency_ms = sum(processed.meta.phase_latency_ms.values())
            st.session_state.processed = processed
        except Exception as exc:
            st.exception(exc)

    if "processed" in st.session_state:
        processed = st.session_state.processed
        st.success("T4 hoàn tất — đây là payload Team 2 sẽ nhận.")
        st.json(processed.to_dict(), expanded=True)
        st.download_button(
            "Tải ProcessedQuery JSON",
            data=json.dumps(processed.to_dict(), ensure_ascii=False, indent=2),
            file_name=f"{processed.query_id}.json",
            mime="application/json",
        )


# -----------------------------------------------------------------------------
# T5 mock
# -----------------------------------------------------------------------------
with phase_t5:
    st.subheader("T5 — mock retrieval + fusion")
    st.warning(
        "Đây chỉ là mock cục bộ để test input/output của T5. Không dùng VisualAgent, "
        "BEiT3, TurboVec hay Elasticsearch."
    )

    if "processed" not in st.session_state:
        st.warning("Chạy T4 trước.")

    candidate_json = st.text_area(
        "Mock candidate pool (JSON array)",
        value=json.dumps(DEFAULT_MOCK_CANDIDATES, ensure_ascii=False, indent=2),
        height=360,
        key="mock_candidate_editor",
    )
    top_k_per_query = st.number_input(
        "Top-K mỗi expanded query",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
    )

    if st.button(
        "Chạy T5 mock",
        key="run_t5",
        type="primary",
        disabled="processed" not in st.session_state,
    ):
        _clear_from("T5")
        started = time.perf_counter()
        try:
            pool = _parse_candidate_pool(candidate_json)
            search_fn = _mock_search_factory(pool)
            fused = retrieve_and_fuse(
                st.session_state.processed,
                search_fn,
                top_k_per_query=int(top_k_per_query),
                final_top_k=len(pool),
            )
            st.session_state.fused = fused
            st.session_state.setdefault("phase_latency_ms", {})["T5"] = _elapsed_ms(started)
        except Exception as exc:
            st.exception(exc)

    if "fused" in st.session_state:
        st.success(f"T5 mock hoàn tất: {len(st.session_state.fused)} candidates")
        _show_candidates(st.session_state.fused, "fusion_score")
        with st.expander("Xem JSON T5"):
            st.json(st.session_state.fused, expanded=False)


# -----------------------------------------------------------------------------
# T6
# -----------------------------------------------------------------------------
with phase_t6:
    st.subheader("T6 — optional LLM rerank")
    st.write("Gemini chỉ đánh giá caption/OCR/ASR/objects của candidates, không xem ảnh gốc.")

    if "processed" not in st.session_state:
        st.warning("Chạy T4 trước.")
    if "fused" not in st.session_state:
        st.warning("Chạy T5 mock trước để có candidates.")

    rerank_top_k = st.number_input(
        "Số candidate gửi Gemini",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )

    can_run_t6 = "processed" in st.session_state and "fused" in st.session_state
    if st.button(
        "Chạy T6",
        key="run_t6",
        type="primary",
        disabled=not can_run_t6,
    ):
        if _require_api_key(api_key):
            _clear_from("T6")
            started = time.perf_counter()
            try:
                client = _client(api_key)
                reranked = rerank_candidates(
                    processed=st.session_state.processed,
                    candidates=st.session_state.fused,
                    client=client,
                    model=rerank_model,
                    top_k=int(rerank_top_k),
                )
                st.session_state.reranked = reranked
                st.session_state.setdefault("phase_latency_ms", {})["T6"] = _elapsed_ms(started)
            except Exception as exc:
                st.exception(exc)

    if "reranked" in st.session_state:
        st.success(f"T6 hoàn tất: {len(st.session_state.reranked)} candidates đã rerank")
        _show_candidates(st.session_state.reranked, "rerank_score")
        with st.expander("Xem JSON T6"):
            st.json(st.session_state.reranked, expanded=False)


st.divider()
st.caption(
    "Khi Team 2 hoàn thiện, chỉ thay mock search trong T5 bằng callback retrieval thật. "
    "Các phase T1–T4 và T6 không cần chuyển vào folder agents."
)
