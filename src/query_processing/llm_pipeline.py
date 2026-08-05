"""AIC 2026 query-planning pipeline for video retrieval.

Phases
------
T1: deterministic preprocessing
T2: Gemini query planning (task, translation, entity graph, constraints, temporal)
T3: Gemini visual-query expansion
T4: build the shared ProcessedQuery v3 contract
T5: call Team 2 retrieval channels and fuse ranked lists
T6: optional metadata-based Gemini reranking

The module intentionally keeps LLM planning separate from ``src/agents``.
Team 2 encoders/search stores are injected into T5 through callback functions.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import unicodedata
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from google import genai
from google.genai import errors, types
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Shared schema v3
# =============================================================================


class TaskType(str, Enum):
    KIS = "KIS"
    AVS = "AVS"
    VQA = "VQA"
    KISC = "KISC"
    TRAKE = "TRAKE"


class VerificationLevel(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class TaskInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: TaskType = TaskType.KIS
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    reason: str = ""


class LanguageInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = "vi"
    target: str = "en"
    needs_translation: bool = True


class QueryText(BaseModel):
    model_config = ConfigDict(extra="ignore")

    original: str
    normalized: str
    translated: str


class SearchQuery(BaseModel):
    """A query sent to one retrieval channel.

    ``query`` is retained as a compatibility alias for integrations that already
    use that key. Internally, code should read ``value`` or ``text``.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    text: str | None = None
    query: str | None = None
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    focus: str = "general"

    @model_validator(mode="after")
    def _synchronize_text_fields(self) -> SearchQuery:
        text = (self.text or self.query or "").strip()
        if not text:
            raise ValueError("SearchQuery phải có text hoặc query")
        self.text = text
        self.query = text
        return self

    @property
    def value(self) -> str:
        return self.text or self.query or ""


class SearchQueries(BaseModel):
    model_config = ConfigDict(extra="ignore")

    visual: list[SearchQuery] = Field(default_factory=list)
    ocr: list[SearchQuery] = Field(default_factory=list)
    asr: list[SearchQuery] = Field(default_factory=list)
    caption: list[SearchQuery] = Field(default_factory=list)


class EntityAttributes(BaseModel):
    """Visual attributes without free-form dictionaries.

    Gemini Developer API structured output does not support JSON Schema
    ``additionalProperties``. Keeping the common attributes explicit avoids
    that unsupported keyword while ``model_dump()`` still produces a normal
    JSON object for Team 2.
    """

    model_config = ConfigDict(extra="ignore")

    colors: list[str] = Field(default_factory=list)
    clothing: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)
    gender: str | None = None
    age_group: str | None = None
    count: int | None = Field(default=None, ge=0)
    position: list[str] = Field(default_factory=list)
    state: list[str] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)


class Entity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    type: str
    canonical: str
    attributes: EntityAttributes = Field(default_factory=EntityAttributes)
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    verification: VerificationLevel = VerificationLevel.SOFT


class Relation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject: str
    predicate: str
    object: str
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    verification: VerificationLevel = VerificationLevel.SOFT


class Constraints(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hard: list[str] = Field(default_factory=list)
    soft: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)


class TemporalSlot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_ref: str | None = None
    description: str = ""


class TemporalSlots(BaseModel):
    """Fixed temporal slots supported by Gemini structured output."""

    model_config = ConfigDict(extra="ignore")

    before: TemporalSlot | None = None
    current: TemporalSlot | None = None
    after: TemporalSlot | None = None


class EventStep(BaseModel):
    """One ordered semantic keyframe target within a TRAKE event sequence.

    Unlike TemporalSlots (fixed before/current/after), TRAKE queries describe
    an arbitrary-length ordered chain of moments (e.g. the 4-step high-jump
    example in the AIC2026 rules: approach/take-off/clearance/landing), so
    this is a list rather than fixed fields.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    order: int = Field(ge=0)
    label: str = ""
    description: str


class TemporalInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["single_moment", "ordered_events", "range", "unknown"] = (
        "single_moment"
    )
    has_order: bool = False
    motion_required: bool = False
    clip_window_sec: float = Field(default=4.0, ge=1.0, le=120.0)
    slots: TemporalSlots = Field(default_factory=TemporalSlots)

    @field_validator("clip_window_sec", mode="before")
    @classmethod
    def _normalize_clip_window_sec(cls, value: Any) -> float:
        """Convert invalid LLM durations to a safe retrieval window.

        Gemini occasionally returns 0, null, an empty string, or a non-finite
        number. Those values should not invalidate the whole T2 response.
        """

        if value is None or value == "":
            return 4.0

        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return 4.0

        if not math.isfinite(seconds) or seconds <= 0:
            return 4.0

        return min(max(seconds, 1.0), 120.0)


class ChannelPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    query_refs: list[str] = Field(default_factory=list)
    index: str | None = None
    top_k_per_query: int = Field(default=50, ge=0)
    top_k: int = Field(default=50, ge=0)
    weight: float = Field(default=1.0, ge=0.0, le=2.0)


class FusionPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    method: Literal["weighted_rrf", "rrf", "weighted_sum"] = "weighted_rrf"
    rrf_k: int = Field(default=60, ge=1)
    use_query_weights: bool = True
    final_top_k: int = Field(default=50, ge=1)


class RerankPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    method: Literal[
        "none",
        "metadata_llm",
        "vlm",
        "metadata_vlm_or_llm",
    ] = "metadata_vlm_or_llm"
    use_fields: list[str] = Field(
        default_factory=lambda: [
            "caption",
            "recap",
            "ocr",
            "asr",
            "objects",
            "timestamp",
        ]
    )
    top_k: int = Field(default=20, ge=1)


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    channels: dict[str, ChannelPlan] = Field(default_factory=dict)
    fusion: FusionPlan = Field(default_factory=FusionPlan)
    rerank: RerankPlan = Field(default_factory=RerankPlan)


class AmbiguityInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ambiguous: bool = False
    clarification_needed: str | None = None


class ModelUse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stage: str
    model: str


class Meta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    planner_version: str = "v3.0"
    models_used: list[ModelUse] = Field(default_factory=list)
    total_llm_calls: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    phase_latency_ms: dict[str, int] = Field(default_factory=dict)


class ProcessedQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = "3.0"
    query_id: str
    task: TaskInfo
    language: LanguageInfo
    query: QueryText
    search_queries: SearchQueries
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    temporal: TemporalInfo = Field(default_factory=TemporalInfo)
    events: list[EventStep] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan
    ambiguity: AmbiguityInfo = Field(default_factory=AmbiguityInfo)
    meta: Meta = Field(default_factory=Meta)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    # Compatibility helpers for code written against the older contract.
    @property
    def task_type(self) -> TaskType:
        return self.task.type

    @property
    def original_query(self) -> str:
        return self.query.original

    @property
    def translated_query(self) -> str:
        return self.query.translated

    @property
    def expanded_queries(self) -> list[SearchQuery]:
        return self.search_queries.visual

    @property
    def ambiguous(self) -> bool:
        return self.ambiguity.ambiguous

    @property
    def clarification_needed(self) -> str | None:
        return self.ambiguity.clarification_needed

    @property
    def negation(self) -> list[str]:
        return self.constraints.negative


# =============================================================================
# Structured Gemini response schemas
# =============================================================================


class T2Analysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: TaskInfo = Field(default_factory=TaskInfo)
    language: LanguageInfo = Field(default_factory=LanguageInfo)
    query: QueryText
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    temporal: TemporalInfo = Field(default_factory=TemporalInfo)
    events: list[EventStep] = Field(default_factory=list)
    ambiguity: AmbiguityInfo = Field(default_factory=AmbiguityInfo)
    ocr_queries: list[SearchQuery] = Field(default_factory=list)
    asr_queries: list[SearchQuery] = Field(default_factory=list)
    caption_queries: list[SearchQuery] = Field(default_factory=list)

    # Compatibility helpers for older UI/debug code.
    @property
    def task_type(self) -> TaskType:
        return self.task.type

    @property
    def task_reason(self) -> str:
        return self.task.reason

    @property
    def translated_en(self) -> str:
        return self.query.translated

    @property
    def ambiguous(self) -> bool:
        return self.ambiguity.ambiguous

    @property
    def clarification_needed(self) -> str | None:
        return self.ambiguity.clarification_needed


class T3ExpandedItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Keep each expansion compact. These constraints are also exported to the
    # Gemini JSON schema as minLength/maxLength.
    text: str = Field(min_length=1, max_length=180)
    focus: Literal[
        "original",
        "full_scene",
        "subject_action",
        "visual_attributes",
        "scene_context",
        "temporal_context",
    ] = "full_scene"
    weight: float = Field(default=1.0, ge=0.0, le=2.0)

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("expanded query must not be empty")
        return value


class T3Expansion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Keep the LLM response deliberately tiny. The pipeline assigns focus and
    # weight deterministically after parsing, so Gemini only has to return
    # three short strings instead of a list of nested objects.
    expanded: list[str] = Field(min_length=3, max_length=3)


class T6RerankItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    frame_id: str
    score: float = Field(ge=0.0, le=100.0)
    reason: str = ""


class T6RerankOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reranked: list[T6RerankItem] = Field(default_factory=list)


# =============================================================================
# Gemini client
# =============================================================================


_GEMINI_JSON_SCHEMA_KEYS = {
    "$id",
    "$defs",
    "$ref",
    "$anchor",
    "type",
    "format",
    "title",
    "description",
    "enum",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "anyOf",
    "oneOf",
    "properties",
    "required",
    "propertyOrdering",
}


def _gemini_json_schema(response_model: type[BaseModel]) -> dict[str, Any]:
    """Return the Developer API-compatible JSON Schema for a Pydantic model.

    Gemini Developer API supports only a subset of JSON Schema and rejects
    ``additionalProperties``. T2 therefore uses fixed nested models instead of
    free-form dictionaries, and this sanitizer also removes unsupported
    Pydantic metadata such as ``default``.
    """

    def clean(node: Any, *, container_key: str | None = None) -> Any:
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node

        # Under properties/$defs, dictionary keys are user field/model names,
        # not JSON Schema keywords, so preserve all of them.
        if container_key in {"properties", "$defs"}:
            return {key: clean(value) for key, value in node.items()}

        output: dict[str, Any] = {}
        for key, value in node.items():
            if key not in _GEMINI_JSON_SCHEMA_KEYS:
                continue
            if key in {"properties", "$defs"}:
                output[key] = clean(value, container_key=key)
            else:
                output[key] = clean(value)
        return output

    schema = clean(response_model.model_json_schema())
    if not isinstance(schema, dict):
        raise TypeError("Pydantic model_json_schema() phải trả về dictionary")
    return schema


class GeminiClient:
    """Wrapper around ``google-genai`` with structured output and retry."""

    RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str | None = None,
        fallback_models: Sequence[str] = (
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
        ),
        max_retries: int = 3,
        base_backoff_sec: float = 2.0,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Chưa có GEMINI_API_KEY. Hãy khai báo biến môi trường "
                "GEMINI_API_KEY hoặc truyền api_key khi tạo QueryPipeline."
            )

        self._client = genai.Client(api_key=resolved_key)
        self._fallback_models = tuple(fallback_models)
        self._max_retries = max_retries
        self._base_backoff_sec = base_backoff_sec
        self._usage = {"input": 0, "output": 0, "total": 0, "calls": 0}

    def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[BaseModel],
        model: str,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        thinking_level: str = "minimal",
    ) -> BaseModel:
        """Generate and validate structured JSON.

        Gemini 3 models think by default. With a small output budget, hidden
        thinking can consume most of the budget and leave a truncated JSON
        string. For structured extraction we therefore request minimal thinking,
        use Developer API ``response_json_schema``, and increase the
        token budget on each validation retry.
        """
        last_error: Exception | None = None

        for model_name in self._model_candidates(model):
            for attempt in range(1, self._max_retries + 1):
                # A malformed EOF response is commonly caused by MAX_TOKENS.
                # Give each retry progressively more room, capped at 8192.
                attempt_max_tokens = min(
                    max_output_tokens * (2 ** (attempt - 1)),
                    8192,
                )

                try:
                    self._usage["calls"] += 1

                    config_kwargs: dict[str, Any] = {
                        "system_instruction": system_instruction,
                        "max_output_tokens": attempt_max_tokens,
                        "response_mime_type": "application/json",
                        # Use standard JSON Schema instead of the legacy
                        # OpenAPI-style response_schema. This is required for
                        # Pydantic dict fields that emit additionalProperties.
                        "response_json_schema": _gemini_json_schema(
                            response_model
                        ),
                    }

                    if model_name.startswith("gemini-3"):
                        config_kwargs["thinking_config"] = types.ThinkingConfig(
                            thinking_level=thinking_level,
                        )
                        # Gemini 3.6 migration guidance recommends omitting
                        # sampling parameters for deterministic extraction.
                    else:
                        config_kwargs["temperature"] = temperature
                        # Gemini 2.5 uses a token budget rather than a level.
                        config_kwargs["thinking_config"] = types.ThinkingConfig(
                            thinking_budget=0,
                        )

                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(**config_kwargs),
                    )
                    self._track_usage(response)

                    parsed = getattr(response, "parsed", None)
                    if isinstance(parsed, response_model):
                        return parsed
                    if parsed is not None:
                        return response_model.model_validate(parsed)

                    raw = (response.text or "").strip()
                    if not raw:
                        raise ValueError(
                            "Gemini trả về nội dung rỗng "
                            f"(finish_reason={self._finish_reason(response)})"
                        )

                    try:
                        return response_model.model_validate_json(raw)
                    except ValidationError as exc:
                        finish_reason = self._finish_reason(response)
                        preview = raw[-240:].replace("\n", " ")
                        raise ValueError(
                            "Structured JSON không hoàn chỉnh hoặc không đúng schema; "
                            f"finish_reason={finish_reason}, "
                            f"max_output_tokens={attempt_max_tokens}, "
                            f"tail={preview!r}"
                        ) from exc

                except errors.APIError as exc:
                    last_error = exc
                    code = getattr(exc, "code", None)

                    if code == 404:
                        logger.warning(
                            "Model %s không khả dụng; thử model fallback.",
                            model_name,
                        )
                        break

                    if code in self.RETRYABLE_CODES and attempt < self._max_retries:
                        self._sleep_before_retry(model_name, attempt, exc)
                        continue

                    raise RuntimeError(
                        f"Gemini API error với model {model_name}: {exc}"
                    ) from exc

                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    last_error = exc
                    if attempt < self._max_retries:
                        self._sleep_before_retry(model_name, attempt, exc)
                        continue
                    break

                except Exception as exc:  # transport/SDK defensive fallback
                    last_error = exc
                    if attempt < self._max_retries:
                        self._sleep_before_retry(model_name, attempt, exc)
                        continue
                    break

        raise RuntimeError(
            "Không nhận được structured output hợp lệ từ Gemini. "
            f"Lỗi cuối: {last_error}"
        )

    def generate_text(
        self,
        *,
        prompt: str,
        model: str,
        system_instruction: str | None = None,
        max_output_tokens: int = 2048,
        thinking_level: str = "minimal",
    ) -> str:
        """Generate plain text with Gemini 3.x-safe settings.

        T3 only needs three short lines, so using plain text avoids a server-side
        partial JSON response while still using Gemini 3.5 Flash.
        """
        last_error: Exception | None = None

        for model_name in self._model_candidates(model):
            for attempt in range(1, self._max_retries + 1):
                attempt_max_tokens = min(
                    max(max_output_tokens, 4096) * (2 ** (attempt - 1)),
                    16384,
                )
                try:
                    self._usage["calls"] += 1
                    config_kwargs: dict[str, Any] = {
                        "system_instruction": system_instruction,
                        "max_output_tokens": attempt_max_tokens,
                    }

                    if model_name.startswith("gemini-3"):
                        config_kwargs["thinking_config"] = types.ThinkingConfig(
                            thinking_level=thinking_level,
                        )
                    else:
                        config_kwargs["thinking_config"] = types.ThinkingConfig(
                            thinking_budget=0,
                        )

                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(**config_kwargs),
                    )
                    self._track_usage(response)
                    raw = (response.text or "").strip()
                    if raw:
                        return raw
                    raise ValueError(
                        "Gemini trả về nội dung rỗng "
                        f"(finish_reason={self._finish_reason(response)})"
                    )

                except errors.APIError as exc:
                    last_error = exc
                    code = getattr(exc, "code", None)
                    if code == 404:
                        logger.warning(
                            "Model %s không khả dụng; thử model fallback.",
                            model_name,
                        )
                        break
                    if code in self.RETRYABLE_CODES and attempt < self._max_retries:
                        self._sleep_before_retry(model_name, attempt, exc)
                        continue
                    raise RuntimeError(
                        f"Gemini API error với model {model_name}: {exc}"
                    ) from exc

                except Exception as exc:
                    last_error = exc
                    if attempt < self._max_retries:
                        self._sleep_before_retry(model_name, attempt, exc)
                        continue
                    break

        raise RuntimeError(
            "Không nhận được text output hợp lệ từ Gemini. "
            f"Lỗi cuối: {last_error}"
        )

    @staticmethod
    def _finish_reason(response: Any) -> str:
        """Return the first candidate finish reason for diagnostics."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "UNKNOWN"
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return "UNKNOWN"
        return str(getattr(reason, "name", reason))

    def list_model_names(self) -> list[str]:
        names: list[str] = []
        for model in self._client.models.list():
            name = getattr(model, "name", None)
            if name:
                names.append(name.removeprefix("models/"))
        return sorted(set(names))

    def usage(self) -> dict[str, int]:
        return dict(self._usage)

    def reset_usage(self) -> None:
        self._usage = {"input": 0, "output": 0, "total": 0, "calls": 0}

    def close(self) -> None:
        self._client.close()

    def _model_candidates(self, requested: str) -> list[str]:
        result: list[str] = []
        for name in (requested, *self._fallback_models):
            if name and name not in result:
                result.append(name)
        return result

    def _sleep_before_retry(
        self,
        model_name: str,
        attempt: int,
        exc: Exception,
    ) -> None:
        wait = self._base_backoff_sec * (2 ** (attempt - 1))
        logger.warning(
            "Gemini attempt %s/%s failed for %s: %s. Retry in %.1fs",
            attempt,
            self._max_retries,
            model_name,
            exc,
            wait,
        )
        time.sleep(wait)

    def _track_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return

        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0)

        self._usage["input"] += input_tokens
        self._usage["output"] += output_tokens
        self._usage["total"] += total_tokens or (input_tokens + output_tokens)


# =============================================================================
# T1 - deterministic preprocessing
# =============================================================================


_CONTROL_CHARS = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_REPEAT_PUNCT = re.compile(r"([.,!?])\1+")


def preprocess(raw: str, lowercase: bool = False) -> str:
    """Normalize Unicode and whitespace without deleting semantic details."""
    if not isinstance(raw, str):
        raise TypeError("raw phải là chuỗi")

    text = unicodedata.normalize("NFC", raw)
    text = _CONTROL_CHARS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _REPEAT_PUNCT.sub(r"\1", text)

    if lowercase:
        text = text.lower()
    return text


# =============================================================================
# T2 - query planning
# =============================================================================


SYSTEM_INSTRUCTION_T2 = """
Bạn là Query Planner cho hệ thống video retrieval AIC.

Nhiệm vụ:
1. Phân loại task: KIS, AVS, VQA, KISC, TRAKE.
2. Chuẩn hóa và dịch query sang tiếng Anh tự nhiên.
3. Trích entity, relation, hard/soft/negative constraint.
4. Phát hiện temporal logic: single moment, ordered events, before/current/after.
5. Nếu TRAKE, trích ra danh sách events theo đúng thứ tự (id, order, label, description).
6. Chỉ tạo OCR/ASR/caption query khi thật sự cần.
7. Không bịa thêm chi tiết không có trong query.
8. Trả về đúng JSON schema, không markdown.
""".strip()


def analyze_query(
    query: str,
    client: GeminiClient,
    history: list[str] | None = None,
    model: str = "gemini-3.5-flash",
) -> T2Analysis:
    history_payload = history[-3:] if history else []

    prompt = f"""
QUERY_JSON: {json.dumps(query, ensure_ascii=False)}
HISTORY_JSON: {json.dumps(history_payload, ensure_ascii=False)}

Quy tắc task, áp dụng theo thứ tự:
1. VQA: query là câu hỏi cần trả lời từ video.
2. TRAKE: query mô tả một CHUỖI nhiều khoảnh khắc/sự kiện có thứ tự bên trong
   cùng một video (ví dụ các bước của một động tác thể thao, một quy trình
   nhiều giai đoạn) và yêu cầu xác định một khung hình riêng cho MỖI bước —
   khác AVS (không phải tìm nhiều cảnh/video riêng biệt) và khác KIS thông
   thường (không chỉ 1 khoảnh khắc). Tín hiệu: liệt kê đánh số (1)(2)(3),
   "giai đoạn", "bước", "chuỗi khoảnh khắc", "trình tự", hoặc nhiều mốc nối
   bằng "rồi", "sau đó", "tiếp theo", "cuối cùng".
3. KISC: chỉ khi query phải dùng history để hiểu các từ như
   "người đó", "cái đó", "nó", "that one".
4. AVS: chỉ khi người dùng yêu cầu nhiều hoặc toàn bộ cảnh, với tín hiệu rõ
   như "tất cả", "mọi", "các cảnh", "những cảnh", "liệt kê",
   "all scenes" hoặc "every occurrence".
5. KIS: mặc định cho một mô tả cảnh/khoảnh khắc cụ thể.
- Một người đang walking/running/riding vẫn là KIS, không phải AVS.

Ví dụ:
- "người đàn ông mặc áo đỏ đi xe đạp vào ban đêm" -> KIS
- "tìm tất cả cảnh người đàn ông đi xe đạp" -> AVS
- "người đàn ông đang làm gì?" -> VQA
- "tìm 4 khoảnh khắc chính khi vận động viên nhảy cao: (1) giậm nhảy,
   (2) bay qua xà, (3) tiếp đất, (4) đứng dậy" -> TRAKE, events =
   [{id:"e1",order:1,label:"giậm nhảy",description:"athlete's take-off foot leaves the ground"},
    {id:"e2",order:2,label:"bay qua xà",description:"athlete's hip is at its highest point over the bar"},
    {id:"e3",order:3,label:"tiếp đất",description:"athlete's back first touches the landing mat"},
    {id:"e4",order:4,label:"đứng dậy",description:"athlete stands up after landing"}]

Quy tắc events (chỉ điền khi task = TRAKE, để rỗng ở mọi task khác):
- Mỗi phần tử: id (e1, e2, ...), order (1, 2, 3, ... theo đúng thứ tự xảy ra),
  label (tên ngắn tiếng Việt của bước, nếu query có nêu), description (mô tả
  tiếng Anh ngắn, giàu tín hiệu thị giác riêng cho đúng khoảnh khắc đó — không
  lặp lại mô tả của bước khác).
- Phải có ít nhất 2 events khi task = TRAKE.

Quy tắc query và language:
- query.original và query.normalized phải giữ đúng QUERY_JSON.
- query.translated là bản dịch tiếng Anh tự nhiên, giữ số lượng, phủ định,
  quan hệ không gian và thứ tự thời gian.
- language.source là vi, en hoặc mixed; target là en.

Quy tắc entity graph:
- Mỗi entity có id duy nhất như e1, e2; type; canonical; attributes.
- Không đưa "red shirt" thành object riêng nếu nó là thuộc tính quần áo của person.
- Relation như person riding bicycle phải tham chiếu entity id.
- Hard constraint chỉ dùng cho điều kiện cốt lõi.
- Màu sắc, giới tính, thời gian trong ngày thường là soft vì visual model
  có thể nhận nhầm, trừ khi query nhấn mạnh đó là điều kiện bắt buộc.
- Phủ định phải đưa vào constraints.negative.

Quy tắc OCR/ASR/caption:
- Chỉ tạo ocr_queries khi query yêu cầu chữ trên hình, biển hiệu, logo,
  số hoặc tên riêng. ID lần lượt o0, o1, ...
- Chỉ tạo asr_queries khi query yêu cầu lời nói, âm thanh, câu thoại hoặc
  người nói. ID lần lượt a0, a1, ...
- caption_queries chỉ dùng khi mô tả ngữ nghĩa dạng caption giúp bổ sung
  visual retrieval. ID lần lượt c0, c1, ...
- Không đẩy toàn bộ visual query sang OCR/ASR.

Quy tắc temporal:
- Nếu có trước/sau/rồi/sau đó/before/after: type=ordered_events,
  has_order=true và điền slots.
- Nếu là hành động chuyển động như riding/running/walking:
  motion_required=true.
- Nếu không có thứ tự: has_order=false.
- clip_window_sec luôn phải lớn hơn 0, không bao giờ trả 0.
- single_moment không chuyển động: 4.0 giây.
- single_moment có chuyển động: 6.0 giây.
- ordered_events: 10.0 giây.
- range: dùng thời lượng query nêu; nếu không có thì 12.0 giây.

Nếu query thiếu thông tin để retrieval đáng tin cậy, đặt ambiguity.ambiguous=true
và viết clarification_needed bằng tiếng Việt.
""".strip()

    result = client.generate_structured(
        prompt=prompt,
        response_model=T2Analysis,
        model=model,
        system_instruction=SYSTEM_INSTRUCTION_T2,
        temperature=0.0,
        max_output_tokens=2048,
    )
    assert isinstance(result, T2Analysis)

    # Deterministic corrections: the model must not rewrite these two fields.
    result.query.original = query
    result.query.normalized = query
    if not result.query.translated.strip():
        result.query.translated = query

    result.language.needs_translation = (
        result.language.source.casefold() != result.language.target.casefold()
    )
    result.ocr_queries = _normalize_query_ids(result.ocr_queries, "o")
    result.asr_queries = _normalize_query_ids(result.asr_queries, "a")
    result.caption_queries = _normalize_query_ids(result.caption_queries, "c")
    result = _normalize_task_type(query=query, analysis=result, history=history)
    return result


_AVS_PATTERNS = (
    r"\btất cả\b",
    r"\bmọi\b",
    r"\bcác cảnh\b",
    r"\bnhững cảnh\b",
    r"\bcác đoạn\b",
    r"\bnhững đoạn\b",
    r"\bliệt kê\b",
    r"\btoàn bộ\b",
    r"\ball scenes\b",
    r"\bevery scene\b",
    r"\bevery occurrence\b",
    r"\ball occurrences\b",
)

_VQA_PATTERNS = (
    r"\bbao nhiêu\b",
    r"\bai\b",
    r"\bcái gì\b",
    r"\blàm gì\b",
    r"\bở đâu\b",
    r"\bkhi nào\b",
    r"\btại sao\b",
    r"\bhow many\b",
    r"\bwho\b",
    r"\bwhat\b",
    r"\bwhere\b",
    r"\bwhen\b",
    r"\bwhy\b",
)

_CONTEXT_PATTERNS = (
    r"\bngười đó\b",
    r"\bcái đó\b",
    r"\bcảnh đó\b",
    r"\bnó\b",
    r"\bthat one\b",
    r"\bthe previous scene\b",
)

_TRAKE_PATTERNS = (
    r"\(\d+\)",
    r"\bbước \d+\b",
    r"\bgiai đoạn \d+\b",
    r"\bkhoảnh khắc \d+\b",
    r"\bevent \d+\b",
    r"\bchuỗi (các )?(khoảnh khắc|sự kiện|hành động)\b",
    r"\btrình tự\b",
    r"\bcác giai đoạn\b",
)

_TRAKE_CONNECTORS = (
    r"\brồi\b",
    r"\bsau đó\b",
    r"\btiếp theo\b",
    r"\bcuối cùng\b",
)


def _normalize_task_type(
    query: str,
    analysis: T2Analysis,
    history: list[str] | None = None,
) -> T2Analysis:
    """Correct clear task classes after the probabilistic LLM decision."""

    normalized = query.casefold().strip()
    has_history = bool(history)

    is_vqa = normalized.endswith("?") or any(
        re.search(pattern, normalized) for pattern in _VQA_PATTERNS
    )
    is_avs = any(re.search(pattern, normalized) for pattern in _AVS_PATTERNS)
    needs_context = any(
        re.search(pattern, normalized) for pattern in _CONTEXT_PATTERNS
    )
    connector_hits = sum(
        1 for pattern in _TRAKE_CONNECTORS if re.search(pattern, normalized)
    )
    is_trake = len(analysis.events) >= 2 or any(
        re.search(pattern, normalized) for pattern in _TRAKE_PATTERNS
    ) or connector_hits >= 2

    if is_vqa:
        expected = TaskType.VQA
        reason = "Query là câu hỏi cần trả lời từ video."
    elif is_trake:
        expected = TaskType.TRAKE
        reason = "Query mô tả chuỗi nhiều khoảnh khắc/sự kiện có thứ tự."
    elif needs_context and has_history:
        expected = TaskType.KISC
        reason = "Query chứa tham chiếu cần lịch sử hội thoại."
    elif is_avs:
        expected = TaskType.AVS
        reason = "Query yêu cầu tìm nhiều hoặc toàn bộ cảnh."
    else:
        expected = TaskType.KIS
        reason = "Query mô tả một cảnh hoặc khoảnh khắc cụ thể."

    if analysis.task.type != expected:
        logger.info(
            "Task type corrected by deterministic rule: %s -> %s",
            analysis.task.type.value,
            expected.value,
        )
        analysis.task.type = expected
        analysis.task.reason = reason
        analysis.task.confidence = max(0.90, analysis.task.confidence)

    if expected == TaskType.TRAKE and len(analysis.events) < 2:
        # Gemini classified/matched TRAKE but didn't extract a usable event
        # sequence -- retrieve_trake() requires >=2 events, so fail safe by
        # downgrading to KIS rather than crashing downstream in T5.
        logger.warning(
            "TRAKE task nhưng chỉ có %d events hợp lệ -- downgrade về KIS.",
            len(analysis.events),
        )
        analysis.task.type = TaskType.KIS
        analysis.task.reason = (
            "TRAKE thiếu events hợp lệ (Gemini không trích được đủ 2 bước), "
            "fallback về KIS."
        )

    return analysis


def _normalize_query_ids(
    queries: list[SearchQuery],
    prefix: str,
) -> list[SearchQuery]:
    output: list[SearchQuery] = []
    seen: set[str] = set()

    for item in queries:
        key = item.value.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(
            item.model_copy(
                update={
                    "id": f"{prefix}{len(output)}",
                    "text": item.value,
                    "query": item.value,
                }
            )
        )
    return output


# =============================================================================
# T3 - semantic visual-query expansion
# =============================================================================


SYSTEM_INSTRUCTION_T3 = """
Bạn là Query Expansion Agent cho video retrieval dùng CLIP/SigLIP.

Tạo query tiếng Anh ngắn, giàu tín hiệu visual, tăng recall nhưng không thêm
điều kiện mới. Không tự thêm camera angle như close-up/wide-shot nếu query gốc
không yêu cầu. Không tự thêm số lượng, địa điểm, thương hiệu hoặc màu sắc mới.
Trả về đúng JSON schema, không markdown.
""".strip()


_FOCUS_WEIGHTS = {
    "original": 1.0,
    "full_scene": 1.0,
    "subject_action": 0.9,
    "visual_attributes": 0.85,
    "scene_context": 0.7,
    "temporal_context": 0.8,
}


def _fallback_t3_expansions(analysis: T2Analysis) -> list[T3ExpandedItem]:
    """Build safe local expansions when Gemini returns malformed/truncated JSON.

    The fallback only reuses entities, relations, attributes and the translated
    query already produced by T2; it does not invent new scene details.
    """
    entity_by_id = {entity.id: entity for entity in analysis.entities}
    candidates: list[tuple[str, str]] = []

    relation_phrases: list[str] = []
    for relation in analysis.relations:
        subject = entity_by_id.get(relation.subject)
        object_ = entity_by_id.get(relation.object)
        subject_text = subject.canonical if subject else relation.subject
        object_text = object_.canonical if object_ else relation.object
        phrase = " ".join(
            part.strip()
            for part in (
                subject_text,
                relation.predicate.replace("_", " "),
                object_text,
            )
            if part and part.strip()
        )
        if phrase:
            relation_phrases.append(phrase)

    if relation_phrases:
        candidates.append(("subject_action", "; ".join(relation_phrases)))

    visual_terms: list[str] = []
    for entity in analysis.entities:
        visual_terms.append(entity.canonical)
        for value in entity.attributes.model_dump(mode="python").values():
            if isinstance(value, list):
                visual_terms.extend(str(item) for item in value if str(item).strip())
            elif value not in (None, ""):
                visual_terms.append(str(value))

    if visual_terms:
        candidates.append(("visual_attributes", " ".join(visual_terms)))

    translated = analysis.query.translated.strip()
    if translated:
        candidates.append(("scene_context", translated))

    output: list[T3ExpandedItem] = []
    seen: set[str] = set()
    original_key = translated.casefold()

    for focus, candidate_text in candidates:
        normalized = " ".join(candidate_text.split()).strip(" ,;.")
        key = normalized.casefold()
        if not normalized or key in seen or key == original_key:
            continue
        seen.add(key)
        output.append(
            T3ExpandedItem(
                text=normalized[:180],
                focus=focus,
                weight=_FOCUS_WEIGHTS[focus],
            )
        )

    return output[:3]


def expand_query(
    analysis: T2Analysis,
    client: GeminiClient,
    model: str = "gemini-3.5-flash",
) -> list[T3ExpandedItem]:
    """Generate T3 expansions with Gemini 3.5 Flash without JSON parsing.

    Gemini 3.5 Flash is still the model doing the expansion. We request exactly
    three plain-text lines and assign focus/weight deterministically in Python.
    This removes the EOF JSON failure seen with structured output.
    """
    prompt = f"""
TRANSLATED_QUERY: {json.dumps(analysis.query.translated, ensure_ascii=False)}
ENTITIES_JSON: {json.dumps([e.model_dump(mode='json') for e in analysis.entities], ensure_ascii=False)}
RELATIONS_JSON: {json.dumps([r.model_dump(mode='json') for r in analysis.relations], ensure_ascii=False)}
HARD_CONSTRAINTS_JSON: {json.dumps(analysis.constraints.hard, ensure_ascii=False)}
SOFT_CONSTRAINTS_JSON: {json.dumps(analysis.constraints.soft, ensure_ascii=False)}
TEMPORAL_JSON: {analysis.temporal.model_dump_json()}

Write exactly 3 English visual-search queries.
Output format is exactly three lines, with no JSON, bullets, numbering, labels,
quotes, markdown, commentary, or blank lines.

Rules:
- Line 1 emphasizes subject and action.
- Line 2 emphasizes visible entities and attributes.
- Line 3 emphasizes scene or temporal context already present in the input.
- Preserve every hard constraint.
- Do not invent objects, colors, brands, places, counts, or camera angles.
- Each line must contain at most 22 words.
- Do not repeat the translated query verbatim.
""".strip()

    try:
        raw = client.generate_text(
            prompt=prompt,
            model=model,
            system_instruction=SYSTEM_INSTRUCTION_T3,
            max_output_tokens=4096,
            thinking_level="minimal",
        )
    except Exception as exc:
        logger.warning(
            "T3 Gemini 3.5 Flash failed; using deterministic fallback: %s",
            exc,
        )
        return _fallback_t3_expansions(analysis)

    lines: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[-*•\s]+", "", line)
        line = re.sub(r"^\d+[.)\-:]\s*", "", line)
        line = line.strip(' "\'`')
        line = " ".join(line.split())
        if line:
            lines.append(line)

    # Some responses may put all items on one line separated by semicolons.
    if len(lines) < 3 and ";" in raw:
        lines = [
            " ".join(part.strip(' "\'`').split())
            for part in raw.split(";")
            if part.strip()
        ]

    focus_order = (
        "subject_action",
        "visual_attributes",
        "scene_context",
    )
    original_key = analysis.query.translated.casefold().strip()
    seen: set[str] = {original_key}
    output: list[T3ExpandedItem] = []

    for focus, line in zip(focus_order, lines[:3]):
        normalized = line[:180].strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        output.append(
            T3ExpandedItem(
                text=normalized,
                focus=focus,
                weight=_FOCUS_WEIGHTS[focus],
            )
        )

    if len(output) < 3:
        fallback = _fallback_t3_expansions(analysis)
        for item in fallback:
            key = item.text.casefold().strip()
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
            if len(output) == 3:
                break

    return output[:3]


# =============================================================================
# T4 - build ProcessedQuery v3
# =============================================================================


def _make_query_id() -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"q_{date_str}_{uuid.uuid4().hex[:8]}"


def _build_retrieval_plan(
    analysis: T2Analysis,
    visual_queries: list[SearchQuery],
    final_top_k: int = 50,
) -> RetrievalPlan:
    visual_enabled = bool(visual_queries)
    ocr_enabled = bool(analysis.ocr_queries)
    asr_enabled = bool(analysis.asr_queries)
    caption_enabled = bool(analysis.caption_queries)

    # Recap/caption text usually carries the specific detail (counts, names,
    # dialogue context) that a pure image embedding can't see, so it should
    # out-rank the visual channel rather than be a weak 0.4x fallback --
    # image search stays the secondary/confirming signal. VQA leans on this
    # even harder since the answer itself typically lives in that detail.
    if analysis.task_type == TaskType.VQA:
        visual_weight, caption_weight = 0.6, 1.2
    else:
        visual_weight, caption_weight = 0.8, 1.0

    channels = {
        "visual": ChannelPlan(
            enabled=visual_enabled,
            query_refs=[q.id for q in visual_queries],
            index="siglip",
            top_k_per_query=100,
            top_k=100,
            weight=visual_weight,
        ),
        "ocr": ChannelPlan(
            enabled=ocr_enabled,
            query_refs=[q.id for q in analysis.ocr_queries],
            index="elasticsearch_ocr",
            top_k_per_query=50,
            top_k=50 if ocr_enabled else 0,
            weight=1.2 if ocr_enabled else 0.0,
        ),
        "asr": ChannelPlan(
            enabled=asr_enabled,
            query_refs=[q.id for q in analysis.asr_queries],
            index="elasticsearch_asr",
            top_k_per_query=50,
            top_k=50 if asr_enabled else 0,
            weight=1.2 if asr_enabled else 0.0,
        ),
        "caption": ChannelPlan(
            enabled=caption_enabled,
            query_refs=[q.id for q in analysis.caption_queries],
            index="elasticsearch_caption",
            top_k_per_query=50,
            top_k=50 if caption_enabled else 0,
            weight=caption_weight if caption_enabled else 0.0,
        ),
    }

    return RetrievalPlan(
        channels=channels,
        fusion=FusionPlan(
            method="weighted_rrf",
            rrf_k=60,
            use_query_weights=True,
            final_top_k=final_top_k,
        ),
        rerank=RerankPlan(
            enabled=True,
            method="metadata_vlm_or_llm",
            use_fields=["caption", "recap", "ocr", "asr", "objects", "timestamp"],
            top_k=20,
        ),
    )


def build_processed_query(
    *,
    original_query: str,
    cleaned_query: str,
    analysis: T2Analysis,
    expansions: list[T3ExpandedItem],
    query_id: str | None = None,
    analyze_model: str = "gemini-3.5-flash",
    expand_model: str = "gemini-3.5-flash",
    total_llm_calls: int = 0,
    total_tokens: int = 0,
    latency_ms: int = 0,
    phase_latency_ms: dict[str, int] | None = None,
    final_top_k: int = 50,
) -> ProcessedQuery:
    visual_queries: list[SearchQuery] = [
        SearchQuery(
            id="v0",
            text=analysis.query.translated,
            weight=1.0,
            focus="original",
        )
    ]

    seen = {analysis.query.translated.casefold().strip()}
    for item in expansions:
        key = item.text.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        visual_queries.append(
            SearchQuery(
                id=f"v{len(visual_queries)}",
                text=item.text,
                weight=item.weight,
                focus=item.focus,
            )
        )

    visual_queries = visual_queries[:5]
    search_queries = SearchQueries(
        visual=visual_queries,
        ocr=analysis.ocr_queries,
        asr=analysis.asr_queries,
        caption=analysis.caption_queries,
    )

    retrieval_plan = _build_retrieval_plan(
        analysis=analysis,
        visual_queries=visual_queries,
        final_top_k=final_top_k,
    )

    return ProcessedQuery(
        schema_version="3.0",
        query_id=query_id or _make_query_id(),
        task=analysis.task,
        language=analysis.language,
        query=QueryText(
            original=original_query,
            normalized=cleaned_query,
            translated=analysis.query.translated,
        ),
        search_queries=search_queries,
        entities=analysis.entities,
        relations=analysis.relations,
        constraints=analysis.constraints,
        temporal=analysis.temporal,
        events=analysis.events,
        retrieval_plan=retrieval_plan,
        ambiguity=analysis.ambiguity,
        meta=Meta(
            planner_version="v3.0",
            models_used=[
                ModelUse(stage="T2_query_planning", model=analyze_model),
                ModelUse(stage="T3_query_expansion", model=expand_model),
            ],
            total_llm_calls=total_llm_calls,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            phase_latency_ms=phase_latency_ms or {},
        ),
    )


# =============================================================================
# T5 - Team 2 retrieval callbacks + multi-channel fusion
# =============================================================================


Candidate = dict[str, Any]
RawCandidate = dict[str, Any] | tuple[str, float] | list[Any]
SearchFn = Callable[[str, int], list[RawCandidate]]


class RetrievalCallbacks(BaseModel):
    """Documentation-only container is avoided because callables are arbitrary.

    Use ``retrieve_and_fuse`` keyword arguments directly. This class exists only
    as a named marker for architecture documentation and is not instantiated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


_CHANNEL_QUERY_ATTR = {
    "visual": "visual",
    "ocr": "ocr",
    "asr": "asr",
    "caption": "caption",
}


def retrieve_and_fuse(
    processed: ProcessedQuery,
    visual_search_fn: SearchFn,
    *,
    ocr_search_fn: SearchFn | None = None,
    asr_search_fn: SearchFn | None = None,
    caption_search_fn: SearchFn | None = None,
    final_top_k: int | None = None,
) -> list[Candidate]:
    """Execute T5 and fuse all available ranked lists.

    Team 2 only needs to provide callbacks with this contract::

        search_fn(query_text: str, top_k: int) -> list

    Each result may be either ``{"frame_id": ..., "score": ...}`` or
    ``(frame_id, score)``. Extra metadata is preserved.
    """

    callbacks: dict[str, SearchFn | None] = {
        "visual": visual_search_fn,
        "ocr": ocr_search_fn,
        "asr": asr_search_fn,
        "caption": caption_search_fn,
    }

    ranked_lists: list[
        tuple[str, ChannelPlan, SearchQuery, list[Candidate]]
    ] = []

    for channel_name, attr_name in _CHANNEL_QUERY_ATTR.items():
        plan = processed.retrieval_plan.channels.get(channel_name)
        if plan is None or not plan.enabled:
            continue

        callback = callbacks[channel_name]
        if callback is None:
            logger.warning(
                "Channel %s được bật nhưng chưa có search callback; bỏ qua.",
                channel_name,
            )
            continue

        channel_queries: list[SearchQuery] = getattr(
            processed.search_queries,
            attr_name,
        )
        query_by_id = {item.id: item for item in channel_queries}
        request_top_k = max(plan.top_k_per_query, plan.top_k, 1)

        for query_ref in plan.query_refs:
            query_obj = query_by_id.get(query_ref)
            if query_obj is None:
                logger.warning(
                    "Không tìm thấy query_ref=%s trong channel=%s.",
                    query_ref,
                    channel_name,
                )
                continue

            raw_results = callback(query_obj.value, request_top_k)
            results = _normalize_candidates(raw_results)
            _validate_candidates(results)
            ranked_lists.append((channel_name, plan, query_obj, results))

    fusion = processed.retrieval_plan.fusion
    output_top_k = final_top_k or fusion.final_top_k

    if fusion.method == "weighted_sum":
        return _weighted_sum_fusion(
            ranked_lists,
            use_query_weights=fusion.use_query_weights,
            final_top_k=output_top_k,
        )

    return _rrf_fusion(
        ranked_lists,
        rrf_k=fusion.rrf_k,
        weighted=fusion.method == "weighted_rrf",
        use_query_weights=fusion.use_query_weights,
        final_top_k=output_top_k,
    )


def _normalize_candidates(candidates: list[RawCandidate]) -> list[Candidate]:
    output: list[Candidate] = []

    for index, item in enumerate(candidates):
        if isinstance(item, dict):
            frame_id = item.get("frame_id", item.get("id"))
            score = item.get("score", item.get("_score"))
            if frame_id is None or score is None:
                raise ValueError(
                    f"Candidate #{index} phải có frame_id/id và score/_score"
                )
            output.append(
                {
                    **item,
                    "frame_id": str(frame_id),
                    "score": float(score),
                }
            )
            continue

        if isinstance(item, (tuple, list)) and len(item) >= 2:
            output.append(
                {
                    "frame_id": str(item[0]),
                    "score": float(item[1]),
                }
            )
            continue

        raise TypeError(
            f"Candidate #{index} phải là dict hoặc tuple(frame_id, score)"
        )

    return output


def _validate_candidates(candidates: list[Candidate]) -> None:
    for index, item in enumerate(candidates):
        if "frame_id" not in item:
            raise ValueError(f"Candidate #{index} thiếu frame_id")
        if "score" not in item:
            raise ValueError(f"Candidate #{index} thiếu score")


def retrieve_trake(
    processed: ProcessedQuery,
    visual_search_fn: SearchFn,
    *,
    caption_search_fn: SearchFn | None = None,
    top_k_per_event: int = 50,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """T5 variant for TaskType.TRAKE: locate the one target video, then align
    exactly one semantic keyframe per event inside that video.

    Unlike retrieve_and_fuse() (single flat ranked list of frame_ids across
    channels/queries), TRAKE needs N separate frame_ids that are all inside
    the SAME video and in event order -- so each event is searched
    independently first, then:

      Stage 1 (video selection): every event "votes" for the videos its own
      top candidates belong to (summing each candidate's per-event RRF
      contribution); the video with the highest total wins.
      Stage 2 (alignment): for each event, keep its own best-scoring
      candidate whose video_id matches the stage-1 winner.

    IMPORTANT: visual_search_fn/caption_search_fn passed here must return
    candidates that include "video_id" (e.g. src/inference.py's search()/
    search_captions(), unlike es_store.py's raw (frame_id, score) BM25
    tuples) -- candidates missing video_id can't vote in stage 1 and can
    never be selected in stage 2.
    """
    if len(processed.events) < 2:
        raise ValueError(
            "retrieve_trake() cần ProcessedQuery.events có >= 2 phần tử "
            f"(hiện có {len(processed.events)})"
        )

    channels: list[tuple[str, SearchFn, float]] = [("visual", visual_search_fn, 1.0)]
    if caption_search_fn is not None:
        channels.append(("caption", caption_search_fn, 0.6))

    events_by_order = sorted(processed.events, key=lambda e: e.order)
    per_event_hits: dict[str, list[Candidate]] = {}

    for event in events_by_order:
        merged: dict[str, Candidate] = {}
        for channel_name, search_fn, weight in channels:
            raw_results = search_fn(event.description, top_k_per_event)
            hits = _normalize_candidates(raw_results)
            _validate_candidates(hits)
            for rank, candidate in enumerate(hits, start=1):
                frame_id = str(candidate["frame_id"])
                record = merged.setdefault(
                    frame_id, {**candidate, "frame_id": frame_id, "fusion_score": 0.0}
                )
                record["fusion_score"] += weight / (rrf_k + rank)
        per_event_hits[event.id] = sorted(
            merged.values(), key=lambda item: item["fusion_score"], reverse=True
        )

    # Stage 1: each event casts one vote per distinct video_id it saw, weighted
    # by that candidate's fusion_score -- avoids one event's deep candidate
    # list drowning out another event's shallow-but-confident list.
    video_scores: dict[str, float] = {}
    for event in events_by_order:
        seen_videos: set[str] = set()
        for candidate in per_event_hits[event.id]:
            video_id = candidate.get("video_id")
            if not video_id or video_id in seen_videos:
                continue
            seen_videos.add(video_id)
            video_scores[video_id] = video_scores.get(video_id, 0.0) + candidate["fusion_score"]

    if not video_scores:
        return {
            "video_id": None,
            "video_score": 0.0,
            "events": [
                {
                    "event_id": event.id,
                    "order": event.order,
                    "label": event.label,
                    "frame_id": None,
                    "score": 0.0,
                    "matched": False,
                }
                for event in events_by_order
            ],
        }

    target_video, video_score = max(video_scores.items(), key=lambda item: item[1])

    # Stage 2: best in-video candidate per event, in event order.
    event_results = []
    for event in events_by_order:
        best = next(
            (c for c in per_event_hits[event.id] if c.get("video_id") == target_video),
            None,
        )
        event_results.append(
            {
                "event_id": event.id,
                "order": event.order,
                "label": event.label,
                "frame_id": best["frame_id"] if best else None,
                "score": best["fusion_score"] if best else 0.0,
                "matched": best is not None,
            }
        )

    return {
        "video_id": target_video,
        "video_score": video_score,
        "events": event_results,
    }


def _rrf_fusion(
    ranked_lists: list[tuple[str, ChannelPlan, SearchQuery, list[Candidate]]],
    *,
    rrf_k: int,
    weighted: bool,
    use_query_weights: bool,
    final_top_k: int,
) -> list[Candidate]:
    merged: dict[str, Candidate] = {}

    for channel_name, channel_plan, query_obj, candidates in ranked_lists:
        channel_weight = channel_plan.weight if weighted else 1.0
        query_weight = query_obj.weight if use_query_weights and weighted else 1.0

        for rank, candidate in enumerate(candidates, start=1):
            frame_id = str(candidate["frame_id"])
            contribution = channel_weight * query_weight / (rrf_k + rank)

            record = merged.setdefault(
                frame_id,
                {
                    **candidate,
                    "frame_id": frame_id,
                    "fusion_score": 0.0,
                    "matched_queries": [],
                    "retrieval_sources": [],
                },
            )
            record["fusion_score"] += contribution
            if channel_name not in record["retrieval_sources"]:
                record["retrieval_sources"].append(channel_name)
            record["matched_queries"].append(
                {
                    "channel": channel_name,
                    "query_id": query_obj.id,
                    "query": query_obj.value,
                    "focus": query_obj.focus,
                    "rank": rank,
                    "raw_score": float(candidate["score"]),
                    "channel_weight": channel_weight,
                    "query_weight": query_weight,
                    "contribution": contribution,
                }
            )

    fused = sorted(
        merged.values(),
        key=lambda item: float(item["fusion_score"]),
        reverse=True,
    )
    for item in fused:
        item["score"] = float(item["fusion_score"])
    return fused[:final_top_k]


def _weighted_sum_fusion(
    ranked_lists: list[tuple[str, ChannelPlan, SearchQuery, list[Candidate]]],
    *,
    use_query_weights: bool,
    final_top_k: int,
) -> list[Candidate]:
    merged: dict[str, Candidate] = {}

    for channel_name, channel_plan, query_obj, candidates in ranked_lists:
        raw_scores = [float(item["score"]) for item in candidates]
        if not raw_scores:
            continue

        low, high = min(raw_scores), max(raw_scores)
        query_weight = query_obj.weight if use_query_weights else 1.0

        for rank, candidate in enumerate(candidates, start=1):
            raw_score = float(candidate["score"])
            normalized = 1.0 if high == low else (raw_score - low) / (high - low)
            contribution = channel_plan.weight * query_weight * normalized
            frame_id = str(candidate["frame_id"])

            record = merged.setdefault(
                frame_id,
                {
                    **candidate,
                    "frame_id": frame_id,
                    "fusion_score": 0.0,
                    "matched_queries": [],
                    "retrieval_sources": [],
                },
            )
            record["fusion_score"] += contribution
            if channel_name not in record["retrieval_sources"]:
                record["retrieval_sources"].append(channel_name)
            record["matched_queries"].append(
                {
                    "channel": channel_name,
                    "query_id": query_obj.id,
                    "query": query_obj.value,
                    "focus": query_obj.focus,
                    "rank": rank,
                    "raw_score": raw_score,
                    "normalized_score": normalized,
                    "channel_weight": channel_plan.weight,
                    "query_weight": query_weight,
                    "contribution": contribution,
                }
            )

    fused = sorted(
        merged.values(),
        key=lambda item: float(item["fusion_score"]),
        reverse=True,
    )
    for item in fused:
        item["score"] = float(item["fusion_score"])
    return fused[:final_top_k]


# =============================================================================
# T6 - optional metadata reranking
# =============================================================================


SYSTEM_INSTRUCTION_T6 = """
Bạn là semantic judge cho video retrieval AIC 2026.
Chấm candidate dựa trên caption, recap, OCR, ASR, detected objects và
timestamp được cung cấp. recap thường chứa nhiều chi tiết/ngữ cảnh hơn caption
(bao gồm cả continuity giữa các frame) nên ưu tiên recap khi 2 field mâu
thuẫn nhau. Không giả vờ đã nhìn thấy frame nếu input không chứa ảnh.
Trả về đúng JSON schema, không markdown.
""".strip()


def rerank_candidates(
    processed: ProcessedQuery,
    candidates: list[Candidate],
    client: GeminiClient,
    *,
    model: str = "gemini-2.5-flash",
    top_k: int | None = None,
) -> list[Candidate]:
    if not candidates:
        return []

    rerank_plan = processed.retrieval_plan.rerank
    if not rerank_plan.enabled or rerank_plan.method == "none":
        return candidates

    rerank_top_k = top_k or rerank_plan.top_k
    selected = candidates[: max(1, min(rerank_top_k, len(candidates)))]
    compact_candidates = [
        _compact_candidate(item, rerank_plan.use_fields) for item in selected
    ]

    requirements = {
        "task": processed.task.model_dump(mode="json"),
        "entities": [item.model_dump(mode="json") for item in processed.entities],
        "relations": [
            item.model_dump(mode="json") for item in processed.relations
        ],
        "constraints": processed.constraints.model_dump(mode="json"),
        "temporal": processed.temporal.model_dump(mode="json"),
    }

    prompt = f"""
QUERY_EN: {json.dumps(processed.query.translated, ensure_ascii=False)}
REQUIREMENTS_JSON: {json.dumps(requirements, ensure_ascii=False)}
CANDIDATES_JSON: {json.dumps(compact_candidates, ensure_ascii=False)}

Xếp hạng lại candidate:
- Thiếu hard constraint hoặc relation cốt lõi: giảm điểm mạnh.
- Vi phạm constraints.negative: điểm rất thấp.
- Soft constraint chỉ dùng để phân biệt khi bằng điểm.
- Ưu tiên OCR, ASR, location và temporal khi query yêu cầu.
- Chỉ dùng metadata đã cung cấp; không suy đoán rằng bạn đã xem ảnh gốc.
- Mỗi frame_id chỉ xuất hiện một lần.
""".strip()

    result = client.generate_structured(
        prompt=prompt,
        response_model=T6RerankOutput,
        model=model,
        system_instruction=SYSTEM_INSTRUCTION_T6,
        temperature=0.0,
        max_output_tokens=2048,
    )
    assert isinstance(result, T6RerankOutput)

    source_by_id = {str(item["frame_id"]): item for item in selected}
    output: list[Candidate] = []
    seen: set[str] = set()

    for item in sorted(result.reranked, key=lambda value: value.score, reverse=True):
        frame_id = str(item.frame_id)
        if frame_id in seen or frame_id not in source_by_id:
            continue
        seen.add(frame_id)
        output.append(
            {
                **source_by_id[frame_id],
                "rerank_score": float(item.score),
                "rerank_reason": item.reason,
            }
        )

    for source in selected:
        frame_id = str(source["frame_id"])
        if frame_id not in seen:
            output.append(
                {
                    **source,
                    "rerank_score": 0.0,
                    "rerank_reason": "Model did not return this candidate.",
                }
            )
    return output


def _compact_candidate(
    candidate: Candidate,
    use_fields: list[str],
) -> Candidate:
    compact: Candidate = {
        "frame_id": str(candidate["frame_id"]),
    }

    aliases: dict[str, tuple[str, ...]] = {
        "timestamp": ("timestamp", "timestamp_sec", "timestamp_seconds"),
        "caption": ("caption",),
        "recap": ("recap",),
        "ocr": ("ocr", "ocr_text"),
        "asr": ("asr", "asr_text"),
        "objects": ("objects",),
    }

    for field_name in use_fields:
        for alias in aliases.get(field_name, (field_name,)):
            if alias in candidate and candidate[alias] not in (None, "", []):
                compact[field_name] = candidate[alias]
                break

    for score_field in ("fusion_score", "score"):
        if score_field in candidate:
            compact[score_field] = candidate[score_field]
    return compact


# =============================================================================
# Orchestrator
# =============================================================================


class QueryPipeline:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        analyze_model: str | None = None,
        expand_model: str | None = None,
        rerank_model: str | None = None,
        fallback_models: Sequence[str] = (
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
        ),
    ) -> None:
        self.client = GeminiClient(
            api_key=api_key,
            fallback_models=fallback_models,
        )
        self.analyze_model = analyze_model or os.getenv(
            "GEMINI_ANALYZE_MODEL",
            "gemini-3.5-flash",
        )
        self.expand_model = expand_model or os.getenv(
            "GEMINI_EXPAND_MODEL",
            "gemini-3.5-flash",
        )
        self.rerank_model = rerank_model or os.getenv(
            "GEMINI_RERANK_MODEL",
            "gemini-3.5-flash",
        )

    def run(
        self,
        raw_query: str,
        *,
        history: list[str] | None = None,
        skip_expansion: bool = False,
        final_top_k: int = 50,
    ) -> ProcessedQuery:
        started = time.perf_counter()
        phase_latency: dict[str, int] = {}
        self.client.reset_usage()

        phase_started = time.perf_counter()
        cleaned = preprocess(raw_query)
        phase_latency["T1_preprocess"] = _elapsed_ms(phase_started)
        if not cleaned:
            raise ValueError("Query rỗng sau khi preprocess")

        phase_started = time.perf_counter()
        analysis = analyze_query(
            query=cleaned,
            client=self.client,
            history=history,
            model=self.analyze_model,
        )
        phase_latency["T2_analyze"] = _elapsed_ms(phase_started)

        expansions: list[T3ExpandedItem] = []
        should_expand = (
            not skip_expansion
            and analysis.task_type != TaskType.TRAKE
            and not (
                analysis.ambiguity.ambiguous
                and analysis.ambiguity.clarification_needed
            )
        )

        if should_expand:
            phase_started = time.perf_counter()
            try:
                expansions = expand_query(
                    analysis=analysis,
                    client=self.client,
                    model=self.expand_model,
                )
            except Exception as exc:
                logger.warning(
                    "T3 failed; tiếp tục với query gốc: %s",
                    exc,
                )
            phase_latency["T3_expand"] = _elapsed_ms(phase_started)
        else:
            phase_latency["T3_expand"] = 0

        usage = self.client.usage()
        phase_started = time.perf_counter()
        processed = build_processed_query(
            original_query=raw_query,
            cleaned_query=cleaned,
            analysis=analysis,
            expansions=expansions,
            analyze_model=self.analyze_model,
            expand_model=self.expand_model,
            total_llm_calls=usage["calls"],
            total_tokens=usage["total"],
            latency_ms=_elapsed_ms(started),
            phase_latency_ms=phase_latency,
            final_top_k=final_top_k,
        )
        phase_latency["T4_build"] = _elapsed_ms(phase_started)
        processed.meta.phase_latency_ms = dict(phase_latency)
        processed.meta.latency_ms = _elapsed_ms(started)
        return processed

    def retrieve(
        self,
        processed: ProcessedQuery,
        visual_search_fn: SearchFn,
        *,
        ocr_search_fn: SearchFn | None = None,
        asr_search_fn: SearchFn | None = None,
        caption_search_fn: SearchFn | None = None,
        final_top_k: int | None = None,
    ) -> list[Candidate]:
        return retrieve_and_fuse(
            processed,
            visual_search_fn,
            ocr_search_fn=ocr_search_fn,
            asr_search_fn=asr_search_fn,
            caption_search_fn=caption_search_fn,
            final_top_k=final_top_k,
        )

    def retrieve_trake(
        self,
        processed: ProcessedQuery,
        visual_search_fn: SearchFn,
        *,
        caption_search_fn: SearchFn | None = None,
        top_k_per_event: int = 50,
    ) -> dict[str, Any]:
        return retrieve_trake(
            processed,
            visual_search_fn,
            caption_search_fn=caption_search_fn,
            top_k_per_event=top_k_per_event,
        )

    def rerank(
        self,
        processed: ProcessedQuery,
        candidates: list[Candidate],
        *,
        top_k: int | None = None,
    ) -> list[Candidate]:
        return rerank_candidates(
            processed,
            candidates,
            self.client,
            model=self.rerank_model,
            top_k=top_k,
        )

    def execute(
        self,
        raw_query: str,
        visual_search_fn: SearchFn,
        *,
        history: list[str] | None = None,
        ocr_search_fn: SearchFn | None = None,
        asr_search_fn: SearchFn | None = None,
        caption_search_fn: SearchFn | None = None,
        skip_expansion: bool = False,
        enable_rerank: bool = True,
        final_top_k: int = 50,
    ) -> dict[str, Any]:
        """Run T1-T6 while keeping Team 2 T5 callbacks injectable."""

        processed = self.run(
            raw_query,
            history=history,
            skip_expansion=skip_expansion,
            final_top_k=final_top_k,
        )

        if processed.ambiguity.ambiguous and processed.clarification_needed:
            return {
                "processed_query": processed,
                "clarification": processed.clarification_needed,
                "results": [],
                "trake_result": None,
            }

        if processed.task_type == TaskType.TRAKE:
            trake_result = self.retrieve_trake(
                processed,
                visual_search_fn,
                caption_search_fn=caption_search_fn,
                top_k_per_event=final_top_k,
            )
            return {
                "processed_query": processed,
                "clarification": None,
                "results": [],
                "trake_result": trake_result,
            }

        candidates = self.retrieve(
            processed,
            visual_search_fn,
            ocr_search_fn=ocr_search_fn,
            asr_search_fn=asr_search_fn,
            caption_search_fn=caption_search_fn,
            final_top_k=final_top_k,
        )

        if enable_rerank and processed.retrieval_plan.rerank.enabled:
            rerank_top_k = min(
                processed.retrieval_plan.rerank.top_k,
                len(candidates),
            )
            reranked_head = self.rerank(
                processed,
                candidates,
                top_k=rerank_top_k,
            )
            reranked_ids = {str(item["frame_id"]) for item in reranked_head}
            tail = [
                item
                for item in candidates
                if str(item["frame_id"]) not in reranked_ids
            ]
            candidates = reranked_head + tail

        return {
            "processed_query": processed,
            "clarification": None,
            "results": candidates[:final_top_k],
            "trake_result": None,
        }

    def close(self) -> None:
        self.client.close()


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


__all__ = [
    "AmbiguityInfo",
    "ChannelPlan",
    "Constraints",
    "Entity",
    "EventStep",
    "FusionPlan",
    "GeminiClient",
    "LanguageInfo",
    "Meta",
    "ModelUse",
    "ProcessedQuery",
    "QueryPipeline",
    "QueryText",
    "Relation",
    "RerankPlan",
    "RetrievalPlan",
    "SearchQueries",
    "SearchQuery",
    "T2Analysis",
    "T3ExpandedItem",
    "T3Expansion",
    "TaskInfo",
    "TaskType",
    "TemporalInfo",
    "TemporalSlot",
    "VerificationLevel",
    "analyze_query",
    "build_processed_query",
    "expand_query",
    "preprocess",
    "rerank_candidates",
    "retrieve_and_fuse",
    "retrieve_trake",
]
