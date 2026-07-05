"""
RAMFold Pareto Search — find Pareto-optimal memory policies.

A policy is Pareto-optimal if no other policy is both better in quality
and lower in memory. This module finds the Pareto frontier across
observed policy outcomes.
"""

from __future__ import annotations

from typing import Optional

from .policy_schema import MemoryPolicy


class ParetoFrontier:
    """Track Pareto-optimal policies by (quality, memory) pairs."""

    def __init__(self):
        self._entries: list[tuple[float, float, MemoryPolicy]] = []

    def add(self, policy: MemoryPolicy):
        """Add a policy observation to the frontier."""
        if policy.pulls == 0:
            return
        quality = policy.avg_quality
        memory = policy.avg_memory
        self._entries.append((quality, memory, policy))
        self._rebuild()

    def _rebuild(self):
        """Remove dominated entries."""
        kept = []
        for i, (q_i, m_i, p_i) in enumerate(self._entries):
            dominated = False
            for j, (q_j, m_j, _) in enumerate(self._entries):
                if i == j:
                    continue
                if q_j >= q_i and m_j <= m_i and (q_j > q_i or m_j < m_i):
                    dominated = True
                    break
            if not dominated:
                kept.append((q_i, m_i, p_i))
        self._entries = kept

    def frontier(self) -> list[dict]:
        """Return Pareto-optimal policies sorted by quality."""
        return sorted(
            [
                {
                    "name": p.name,
                    "quality": round(q, 4),
                    "memory": round(m, 4),
                    "pulls": p.pulls,
                    "reward": round(p.avg_reward, 4),
                }
                for q, m, p in self._entries
            ],
            key=lambda x: x["quality"],
            reverse=True,
        )

    def best_quality_per_gb(self) -> Optional[dict]:
        """Return the policy with highest quality/memory ratio."""
        if not self._entries:
            return None
        best = max(self._entries, key=lambda x: x[0] / max(x[1], 0.001))
        q, m, p = best
        return {
            "name": p.name,
            "quality": round(q, 4),
            "memory": round(m, 4),
            "qpg": round(q / max(m, 0.001), 4),
            "pulls": p.pulls,
        }
