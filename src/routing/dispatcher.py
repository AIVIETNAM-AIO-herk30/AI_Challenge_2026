"""
Dynamic dispatcher — Phase 1: direct, synchronous fan-out (no queuing math).
Owner: Le Nguyen Khoi

Phase 1 (docs/IMPLEMENTATION_PLAN.md §6): the preliminary round grades
accuracy, not latency, so this dispatch() simply runs every agent the
query's classified type needs, concurrently, and returns their results.
No wait-time estimation, no SLA enforcement — that's Phase 2.

Phase 2 TODO (Le Nguyen Khoi) — the actual M/M/c contribution
(docs/ARCHITECTURE.md §6):
  Each agent pool is modeled as an M/M/c queue:
    λ = measured arrival rate (queries/sec), from tracked call timestamps
    μ = 1 / avg_latency_sec, from AgentResult.latency_ms history
    c = max_concurrent (already in config.yaml per agent)
  For each candidate pool:
    ρ = λ / (c·μ)                         — must stay < 1 for a stable queue
    C(c, a) = Erlang-C probability of waiting (Kleinrock 1975)
    E[W] = C(c, a) / (c·μ − λ)            — expected wait, seconds
  If E[W] * 1000 > self._sla_ms: log a warning and prefer an alternate
  pool when the query is HYBRID (multiple pools can satisfy it).
"""

import asyncio

from .classifier import ClassifierOutput, QueryType


class DynamicDispatcher:
    # Maps QueryType -> which agent names to invoke
    AGENT_MAP = {
        QueryType.TEXT_ONLY: ["visual"],
        QueryType.OCR:       ["ocr", "visual"],
        QueryType.ASR:       ["asr", "visual"],
        QueryType.HYBRID:    ["ocr", "asr", "visual"],
    }

    def __init__(self, agents: dict, sla_ms: float = 500.0):
        self._agents = agents
        self._sla_ms = sla_ms

    async def dispatch(self, payload, clf_output: ClassifierOutput) -> list:
        agent_names = self.AGENT_MAP[clf_output.query_type]
        tasks = [
            self._agents[name].process(payload)
            for name in agent_names
            if name in self._agents
        ]
        if not tasks:
            return []
        return await asyncio.gather(*tasks)
