"""
RAMFold Bandit Controller — UCB policy search over memory policies.

Each policy arm is a memory configuration.
Reward = quality - memory_penalty - swap_penalty - latency_penalty

Select next arm using UCB:
  arm_t = argmax_i score_i + c * sqrt(log(t) / pulls_i)
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Optional

from .policy_schema import MemoryPolicy, generate_policy_grid


class BanditController:
    """UCB bandit over memory policies."""

    def __init__(
        self,
        arms: Optional[list[MemoryPolicy]] = None,
        c: float = 1.41,
        target_pressure: float = 0.70,
        memory_weight: float = 2.0,
        swap_weight: float = 3.0,
        latency_weight: float = 0.3,
    ):
        self.arms = arms or generate_policy_grid()
        self.c = c
        self.target_pressure = target_pressure
        self.memory_weight = memory_weight
        self.swap_weight = swap_weight
        self.latency_weight = latency_weight
        self.total_pulls = 0
        self.history: list[dict] = []

    def compute_reward(
        self,
        quality: float,
        pressure: float,
        swap_delta: float,
        latency_s: float,
    ) -> float:
        """
        r = quality - memory_penalty - swap_penalty - latency_penalty
        """
        memory_penalty = max(0.0, pressure - self.target_pressure) * self.memory_weight
        swap_penalty = max(0.0, swap_delta) * self.swap_weight
        latency_penalty = max(0.0, 1.0 - (1.0 / max(latency_s, 0.001))) * self.latency_weight
        return quality - memory_penalty - swap_penalty - latency_penalty

    def ucb_score(self, arm: MemoryPolicy) -> float:
        if arm.pulls == 0:
            return float("inf")
        exploitation = arm.avg_reward
        exploration = self.c * math.sqrt(math.log(max(self.total_pulls, 1)) / arm.pulls)
        return exploitation + exploration

    def select(self) -> MemoryPolicy:
        """Select next arm using UCB."""
        return max(self.arms, key=self.ucb_score)

    def update(
        self,
        arm: MemoryPolicy,
        quality: float,
        pressure: float,
        swap_delta: float,
        latency_s: float,
    ):
        """Update arm statistics after a pull."""
        reward = self.compute_reward(quality, pressure, swap_delta, latency_s)
        arm.pulls += 1
        arm.total_reward += reward
        arm.total_quality += quality
        arm.total_memory += pressure
        arm.total_swap += max(0.0, swap_delta)
        arm.total_latency += latency_s
        self.total_pulls += 1

        self.history.append({
            "pull": self.total_pulls,
            "arm": arm.name,
            "reward": round(reward, 4),
            "quality": round(quality, 4),
            "pressure": round(pressure, 4),
            "swap_delta": round(swap_delta, 4),
            "latency_s": round(latency_s, 4),
            "ts": time.time(),
        })

    def best_arm(self) -> Optional[MemoryPolicy]:
        pulled = [a for a in self.arms if a.pulls > 0]
        if not pulled:
            return None
        return max(pulled, key=lambda a: a.avg_reward)

    def top_arms(self, n: int = 10) -> list[MemoryPolicy]:
        pulled = [a for a in self.arms if a.pulls > 0]
        return sorted(pulled, key=lambda a: a.avg_reward, reverse=True)[:n]

    def summary(self) -> dict:
        best = self.best_arm()
        return {
            "total_pulls": self.total_pulls,
            "arms_total": len(self.arms),
            "arms_pulled": sum(1 for a in self.arms if a.pulls > 0),
            "best_arm": best.name if best else None,
            "best_reward": round(best.avg_reward, 4) if best else None,
            "best_quality": round(best.avg_quality, 4) if best else None,
            "best_memory": round(best.avg_memory, 4) if best else None,
            "best_swap": round(best.avg_swap, 4) if best else None,
            "top_5": [
                {
                    "name": a.name,
                    "reward": round(a.avg_reward, 4),
                    "quality": round(a.avg_quality, 4),
                    "pulls": a.pulls,
                }
                for a in self.top_arms(5)
            ],
        }


class SwapWeightedGovernor:
    """
    RAMFold v2 swap-weighted attribution governor.

    Decision logic:
    - Hold when model share is small.
    - Hold when pressure rises but swap is flat and loss is improving.
    - Compress when swap increases meaningfully.
    - Compress when model-attributed memory rises sharply.
    - Compress when pressure jumps hard AND throughput collapses AND loss not improving.
    - Floor-proceed when already at minimum settings.
    """

    def __init__(
        self,
        budget_gb: float = 18.0,
        model_share_threshold: float = 0.01,
        swap_delta_threshold: float = 0.3,
        swap_absolute_threshold: float = 2.0,
        pressure_jump_threshold: float = 0.10,
        pressure_floor: float = 0.85,
        tps_collapse_ratio: float = 0.3,
    ):
        self.budget_gb = budget_gb
        self.budget_mb = budget_gb * 1024
        self.model_share_threshold = model_share_threshold
        self.swap_delta_threshold = swap_delta_threshold
        self.swap_absolute_threshold = swap_absolute_threshold
        self.pressure_jump_threshold = pressure_jump_threshold
        self.pressure_floor = pressure_floor
        self.tps_collapse_ratio = tps_collapse_ratio

    def decide(
        self,
        policy: MemoryPolicy,
        model_mem_mb: float,
        pressure: float,
        pressure_delta: float,
        swap_delta: float,
        swap_growing: bool,
        loss_improving: bool,
        tps_collapsing: bool,
        at_floor: bool,
        swap_absolute: float = 0.0,
    ) -> tuple[str, MemoryPolicy]:
        """Return (action_name, new_policy)."""
        model_share = model_mem_mb / self.budget_mb

        # Gate 1: Model footprint gate
        if model_share < self.model_share_threshold:
            return "hold_low_model_share", policy

        # Gate 2: Swap-weighted danger gate (delta-based)
        if swap_delta > self.swap_delta_threshold or swap_growing:
            new = policy.shrink(0.80)
            return f"v2_swap_danger_delta={swap_delta:.3f}", new

        # Gate 2b: Absolute swap level gate — swap already elevated from competing workloads
        if swap_absolute > self.swap_absolute_threshold and not at_floor:
            new = policy.shrink(0.90)
            return f"v2_swap_absolute={swap_absolute:.3f}", new

        # Gate 3: Pressure + throughput collapse + no loss improvement
        if (pressure_delta > self.pressure_jump_threshold and
            pressure > self.pressure_floor and
            tps_collapsing and not loss_improving and not at_floor):
            new = policy.shrink(0.85)
            return f"v2_pressure_tps_collapse_p={pressure_delta:.3f}", new

        # Relax: pressure dropping and swap stable
        if pressure_delta < -0.05 and swap_delta <= 0.05:
            new = policy.relax(1.15)
            return "v2_relax", new

        # Default: hold. Pressure rising but swap flat = macOS compressing successfully.
        return "hold_pressure_compression_safe", policy
