"""A dependency-light ReAct-style planner for AIC multimedia retrieval.

The planner deliberately separates planning from tools: production deployments
can inject an LLM query-expander, while this baseline remains useful without
an API key by calling the existing multimodal ``search`` function directly.
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable


ORCHESTRATOR_SYSTEM_PROMPT = """Bạn là AGENT ROUTER của hệ thống Video Retrieval AIC 2026.
Phân loại truy vấn thành KIS, AVS, VQA hoặc KISC; chỉ gọi tool khi evidence
chưa đủ; ưu tiên truy vấn vector/text diện rộng, sau đó kiểm chứng theo thời
gian và không gian. Không bịa chi tiết: trả lời bằng video, timestamp và bằng
chứng. Với KISC mơ hồ, hỏi làm rõ thay vì đoán. Với AVS, trả về tất cả đoạn
đạt ngưỡng thay vì chỉ một kết quả."""


class TaskType(StrEnum):
    KIS = "KIS"
    AVS = "AVS"
    VQA = "VQA"
    KISC = "KISC"


@dataclass(frozen=True)
class RetrievalPlan:
    task_type: TaskType
    queries: list[str]
    actions: list[str]
    clarification: str | None = None


@dataclass(frozen=True)
class TemporalClip:
    video_id: str
    start_sec: float
    end_sec: float
    frames: list[dict]
    score: float


@dataclass(frozen=True)
class AgentResponse:
    plan: RetrievalPlan
    results: list[dict]
    clips: list[TemporalClip]
    summary: str


SearchFunction = Callable[[str, dict, int], list[dict]]
ExpandFunction = Callable[[str], list[str]]


def classify_task(query: str, history: list[str] | None = None) -> TaskType:
    """Small, inspectable fallback until an LLM classifier is configured."""
    text = query.lower()
    if history and any(phrase in text for phrase in ("that one", "it", "instead", "còn", "đó")):
        return TaskType.KISC
    if any(word in text for word in ("all ", "every ", "tất cả", "mọi ", "những cảnh")):
        return TaskType.AVS
    if "?" in query or any(word in text for word in ("how many", "why", "what", "who", "khi nào", "bao nhiêu")):
        return TaskType.VQA
    return TaskType.KIS


def group_temporal_results(results: list[dict], window_sec: float = 5.0) -> list[TemporalClip]:
    """Turn frame hits into contiguous, explainable candidate video moments."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_video[result["video_id"]].append(result)

    clips: list[TemporalClip] = []
    for video_id, video_results in by_video.items():
        current: list[dict] = []
        for result in sorted(video_results, key=lambda item: item["timestamp_sec"]):
            if current and result["timestamp_sec"] - current[-1]["timestamp_sec"] > window_sec:
                clips.append(_make_clip(video_id, current))
                current = []
            current.append(result)
        if current:
            clips.append(_make_clip(video_id, current))
    return sorted(clips, key=lambda clip: clip.score, reverse=True)


def _make_clip(video_id: str, frames: list[dict]) -> TemporalClip:
    return TemporalClip(
        video_id=video_id,
        start_sec=frames[0]["timestamp_sec"],
        end_sec=frames[-1]["timestamp_sec"],
        frames=frames,
        score=max(frame["score"] for frame in frames),
    )


class ReActOrchestrator:
    """Plan → retrieve → aggregate evidence, exposing only concise observations."""

    def __init__(self, search_fn: SearchFunction, expand_fn: ExpandFunction | None = None):
        self._search = search_fn
        self._expand = expand_fn or (lambda query: [query])

    def plan(
        self,
        query: str,
        history: list[str] | None = None,
        task_override: TaskType | None = None,
    ) -> RetrievalPlan:
        task_type = task_override or classify_task(query, history)
        if task_type is TaskType.KISC and self._is_ambiguous(query):
            return RetrievalPlan(
                task_type=task_type,
                queries=[],
                actions=["clarify"],
                clarification=(
                    "Bạn có thể thêm một đặc điểm phân biệt—màu sắc, hành động, "
                    "địa điểm hoặc mốc thời gian tương đối—để tôi thu hẹp kết quả không?"
                ),
            )

        queries = self._deduplicate_queries(query, self._expand(query))
        return RetrievalPlan(
            task_type=task_type,
            queries=queries,
            actions=["query_expand", "late_fusion_search", "temporal_group", "frame_selector"],
        )

    def run(
        self,
        query: str,
        config: dict,
        top_k: int = 10,
        history: list[str] | None = None,
        task_override: TaskType | None = None,
    ) -> AgentResponse:
        plan = self.plan(query, history, task_override)
        if plan.clarification:
            return AgentResponse(plan=plan, results=[], clips=[], summary=plan.clarification)

        candidate_k = top_k if plan.task_type is TaskType.KIS else max(top_k * 3, top_k)
        results = self._fuse_expansion_results(plan.queries, config, candidate_k)
        clips = group_temporal_results(results)
        if plan.task_type is TaskType.KIS:
            results = results[:top_k]
            clips = clips[:top_k]
        return AgentResponse(
            plan=plan,
            results=results,
            clips=clips,
            summary=self._summarize(plan.task_type, results, clips),
        )

    @staticmethod
    def _is_ambiguous(query: str) -> bool:
        return len(query.split()) < 4 or query.lower().strip() in {"find it", "tìm nó", "cái đó"}

    @staticmethod
    def _deduplicate_queries(query: str, expanded: list[str]) -> list[str]:
        queries: list[str] = []
        for candidate in [query, *expanded]:
            normalized = candidate.strip()
            if normalized and normalized not in queries:
                queries.append(normalized)
        return queries[:5]

    def _fuse_expansion_results(self, queries: list[str], config: dict, top_k: int) -> list[dict]:
        fused: dict[tuple[str, int], tuple[dict, float]] = {}
        for query in queries:
            for rank, result in enumerate(self._search(query, config, top_k), start=1):
                key = (result["video_id"], result["frame_idx"])
                score = fused.get(key, (result, 0.0))[1] + 1.0 / (60 + rank)
                fused[key] = (result, score)
        return [
            {**result, "score": score}
            for result, score in sorted(fused.values(), key=lambda item: item[1], reverse=True)
        ][:top_k]

    @staticmethod
    def _summarize(task_type: TaskType, results: list[dict], clips: list[TemporalClip]) -> str:
        if not results:
            return "Chưa tìm thấy evidence đủ tin cậy trong các chỉ mục hiện có."
        if task_type is TaskType.VQA:
            return "Đã thu thập evidence; cần Answer Generator chuyên dụng để trả lời câu hỏi tự nhiên."
        if task_type is TaskType.AVS:
            return f"Tìm thấy {len(clips)} đoạn ứng viên trên {len({r['video_id'] for r in results})} video."
        best = results[0]
        return f"Ứng viên tốt nhất: {best['video_id']} tại {best['timestamp_sec']:.2f}s."
