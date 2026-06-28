"""
Query complexity classifier — lightweight model that feeds the dispatcher.
Owner: Le Nguyen Khoi
"""

from dataclasses import dataclass
from enum import IntEnum


class QueryType(IntEnum):
    TEXT_ONLY = 0
    OCR = 1
    ASR = 2
    HYBRID = 3


class QueryComplexity(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class ClassifierOutput:
    query_type: QueryType
    complexity: QueryComplexity
    complexity_score: float   # scalar in [0, 1], used by dispatcher


def rule_based_classify(query: str) -> ClassifierOutput:
    """
    Simple keyword-based fallback — no model weights needed.
    Use during Sprint 1 until the trained classifier is ready.

    TODO (Le Nguyen Khoi):
    - Improve keyword lists based on EDA findings from 01_eda_queries.ipynb
    - Replace with trained QueryClassifier from model.py in Sprint 2
    """
    q = query.lower()

    has_ocr = any(w in q for w in ["text", "written", "sign", "caption", "label"])
    has_asr = any(w in q for w in ["say", "speak", "voice", "audio", "sound"])

    if has_ocr and has_asr:
        qtype = QueryType.HYBRID
    elif has_ocr:
        qtype = QueryType.OCR
    elif has_asr:
        qtype = QueryType.ASR
    else:
        qtype = QueryType.TEXT_ONLY

    n_tokens = len(query.split())
    if n_tokens <= 5:
        complexity, score = QueryComplexity.LOW, 0.1
    elif n_tokens <= 15:
        complexity, score = QueryComplexity.MEDIUM, 0.5
    else:
        complexity, score = QueryComplexity.HIGH, 0.9

    return ClassifierOutput(qtype, complexity, score)
