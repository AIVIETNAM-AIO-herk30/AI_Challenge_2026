"""
Dynamic dispatcher based on M/M/c queuing theory.
Owner: Le Nguyen Khoi

Key concept:
  Each agent pool is modeled as an M/M/c queue:
    λ  = measured arrival rate (queries/sec)
    μ  = agent service rate (1 / avg_latency_sec)
    c  = number of parallel workers (max_concurrent)
  Goal: route query to the pool with minimum expected waiting time E[W].
"""

from .classifier import ClassifierOutput, QueryType


class DynamicDispatcher:
    """
    TODO (Le Nguyen Khoi):
    - Implement erlang_c_wait(lambda_, mu, c) → expected wait in ms
      (Erlang-C formula, see Kleinrock 1975)
    - Track arrival timestamps per agent to estimate λ
    - In dispatch(): select agents for the given QueryType,
      compute E[W] per pool, log warning if E[W] > sla_ms
    - Run selected agents concurrently with asyncio.gather()

    Reference formula (Erlang-C):
      ρ = λ / (c·μ)          must be < 1 for stable queue
      C(c, a) = Erlang-C probability (prob of waiting)
      E[W] = C(c, a) / (c·μ - λ)   [seconds]
    """

    # Maps QueryType → which agent names to invoke
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
        raise NotImplementedError
