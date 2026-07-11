"""
Abstract base class for all specialized agents.
Owner: Truong Hoang Thong

Contract (docs/IMPLEMENTATION_PLAN.md §2.3): process() always returns an
AgentResult, even on failure — callers never need to catch exceptions from
agents directly, they check `.success`.
"""

import asyncio
import time
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
    process() wraps _run() with timing, a concurrency limit (max_concurrent),
    and error handling so one failed request never crashes the caller.
    """

    def __init__(self, name: str, max_concurrent: int = 4):
        self.name = name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def process(self, payload: Any) -> AgentResult:
        start = time.perf_counter()
        async with self._semaphore:
            try:
                output = await self._run(payload)
                return AgentResult(
                    agent_type=self.name,
                    output=output,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    success=True,
                )
            except Exception as exc:  # noqa: BLE001 — agents must never raise past this point
                return AgentResult(
                    agent_type=self.name,
                    output=None,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    success=False,
                    error=str(exc),
                )

    @abstractmethod
    async def _run(self, payload: Any) -> Any:
        ...
