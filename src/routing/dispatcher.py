"""
Dynamic dispatcher: direct, synchronous fan-out.
Owner: Le Nguyen Khoi

This dispatch() simply runs every agent the query's classified type needs, concurrently, and returns their results.
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
