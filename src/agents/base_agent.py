"""
Abstract base class for all specialized agents.
Owner: Truong Hoang Thong
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentResult:
    agent_type: str
    output: Any
    latency_ms: float
    success: bool
    error: str | None = None


class BaseAgent(ABC):
    """
    All agents (OCR, ASR, Visual) must inherit this class and implement _run().
    The process() method wraps _run() with timing and error handling.

    TODO (Truong Hoang Thong):
    - Add asyncio.Semaphore for concurrency control (max_concurrent)
    - Track latency stats (used by dispatcher for M/M/c service rate μ)
    """

    def __init__(self, name: str, max_concurrent: int = 4):
        self.name = name
        self.max_concurrent = max_concurrent

    async def process(self, payload: Any) -> AgentResult:
        raise NotImplementedError

    @abstractmethod
    async def _run(self, payload: Any) -> Any:
        ...
