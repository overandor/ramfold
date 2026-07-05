"""
RAMFold Verification Layer — prevents fake progress.

A memory policy only wins if quality survives.

Metrics:
  QPG = Verified Quality / Peak GB
  SFR = Swap-Free Run Rate
  MER = Memory Elasticity Ratio = quality_retained / memory_reduced
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Quality metrics for a single trial."""
    final_loss: float
    min_loss: float
    avg_loss: float
    tokens_per_sec: float
    peak_pressure: float
    peak_swap_gb: float
    model_mem_mb: float
    budget_gb: float
    crashed: bool
    policy_changes: int
    total_steps: int
    completed_steps: int

    @property
    def quality_score(self) -> float:
        """Normalized quality: 1.0 = perfect, 0.0 = useless."""
        if self.crashed:
            return 0.0
        return max(0.0, 1.0 - (self.final_loss / 5.0))

    @property
    def verification_score(self) -> float:
        """1.0 if verified useful, 0.5 if marginal, 0.0 if failed."""
        if self.crashed:
            return 0.0
        if self.final_loss < 1.0:
            return 1.0
        elif self.final_loss < 2.0:
            return 0.75
        elif self.final_loss < 3.0:
            return 0.5
        return 0.25

    @property
    def peak_gb(self) -> float:
        return self.peak_pressure * self.budget_gb

    @property
    def qpg(self) -> float:
        """Quality Per GB = verified quality / peak GB used."""
        if self.peak_gb <= 0:
            return 0.0
        return round(self.quality_score * self.verification_score / self.peak_gb, 4)

    @property
    def sfr(self) -> float:
        """Swap-Free Run Rate = fraction of steps with zero swap growth."""
        if self.peak_swap_gb <= 0.01:
            return 1.0
        return round(max(0.0, 1.0 - self.peak_swap_gb / 10.0), 4)

    @property
    def mer(self) -> float:
        """Memory Elasticity Ratio = quality_retained / memory_reduced.
        
        Requires a baseline to compare against.
        Returns 1.0 if no baseline (no reduction).
        """
        return 1.0  # computed in comparison with baseline

    def to_dict(self) -> dict:
        return {
            "final_loss": self.final_loss,
            "min_loss": self.min_loss,
            "avg_loss": self.avg_loss,
            "quality_score": round(self.quality_score, 4),
            "verification_score": round(self.verification_score, 4),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "peak_pressure": round(self.peak_pressure, 4),
            "peak_swap_gb": round(self.peak_swap_gb, 4),
            "model_mem_mb": round(self.model_mem_mb, 2),
            "peak_gb": round(self.peak_gb, 4),
            "qpg": self.qpg,
            "sfr": self.sfr,
            "crashed": self.crashed,
            "policy_changes": self.policy_changes,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
        }


@dataclass
class PolicyObjective:
    """
    J(θ, μ) = VerifiedQuality(θ, C_μ(D))
              - λ1·PeakMemory
              - λ2·Swap
              - λ3·Latency
              - λ4·Recompute
              - λ5·CloudCost

    This is the formal objective. The controller searches for μ* = argmax J.
    """
    lambda_memory: float = 0.5
    lambda_swap: float = 2.0
    lambda_latency: float = 0.01
    lambda_recompute: float = 0.1
    lambda_cloud: float = 1.0

    def evaluate(
        self,
        metrics: QualityMetrics,
        recompute_cost: float = 0.0,
        cloud_cost: float = 0.0,
        latency_s: float = 0.0,
    ) -> float:
        """Compute J(θ, μ) for a trial result."""
        verified_quality = metrics.quality_score * metrics.verification_score
        peak_memory = metrics.peak_gb
        swap = metrics.peak_swap_gb
        return round(
            verified_quality
            - self.lambda_memory * peak_memory
            - self.lambda_swap * swap
            - self.lambda_latency * latency_s
            - self.lambda_recompute * recompute_cost
            - self.lambda_cloud * cloud_cost,
            6,
        )

    def to_dict(self) -> dict:
        return {
            "lambda_memory": self.lambda_memory,
            "lambda_swap": self.lambda_swap,
            "lambda_latency": self.lambda_latency,
            "lambda_recompute": self.lambda_recompute,
            "lambda_cloud": self.lambda_cloud,
        }


def compute_mer(baseline: QualityMetrics, adaptive: QualityMetrics):
    """Memory Elasticity Ratio: quality_retained / memory_reduced.
    
    Returns None when memory reduction is negligible (<2%),
    because the ratio becomes unstable and misleading.
    """
    quality_retained = adaptive.quality_score / max(baseline.quality_score, 0.001)
    memory_baseline = baseline.peak_gb
    memory_adaptive = adaptive.peak_gb
    if memory_baseline <= 0:
        return None
    memory_reduced = (memory_baseline - memory_adaptive) / memory_baseline
    if memory_reduced < 0.02:
        return None  # negligible reduction — MER would be misleading
    return round(quality_retained / memory_reduced, 4)


def compare_trials(baseline: QualityMetrics, adaptive: QualityMetrics) -> dict:
    """Compare baseline vs adaptive trial."""
    return {
        "baseline": baseline.to_dict(),
        "adaptive": adaptive.to_dict(),
        "quality_delta": round(adaptive.quality_score - baseline.quality_score, 4),
        "memory_delta_gb": round(adaptive.peak_gb - baseline.peak_gb, 4),
        "swap_delta_gb": round(adaptive.peak_swap_gb - baseline.peak_swap_gb, 4),
        "qpg_baseline": baseline.qpg,
        "qpg_adaptive": adaptive.qpg,
        "qpg_delta": round(adaptive.qpg - baseline.qpg, 4),
        "qpg_improvement": round(adaptive.qpg - baseline.qpg, 4),
        "mer": compute_mer(baseline, adaptive),
        "sfr_baseline": baseline.sfr,
        "sfr_adaptive": adaptive.sfr,
        "verdict": _verdict(baseline, adaptive),
    }


def _verdict(baseline: QualityMetrics, adaptive: QualityMetrics) -> str:
    q_delta = adaptive.quality_score - baseline.quality_score
    m_delta = adaptive.peak_gb - baseline.peak_gb
    s_delta = adaptive.peak_swap_gb - baseline.peak_swap_gb
    # Use percentage-based threshold: 2% of budget is meaningful reduction
    meaningful_threshold = baseline.budget_gb * 0.02
    m_reduced = m_delta < -meaningful_threshold
    m_stable = abs(m_delta) < meaningful_threshold

    if adaptive.crashed:
        return "FAIL: adaptive crashed"
    # Meaningful memory reduction (>=2% of budget)
    if abs(q_delta) < 0.03 and m_reduced and s_delta <= 0:
        return "WIN: same quality, less memory, no swap"
    if abs(q_delta) < 0.03 and m_reduced:
        return "WIN: same quality, reduced memory"
    if q_delta > 0 and m_reduced:
        return "WIN: better quality, reduced memory"
    if q_delta < -0.05 and m_reduced:
        return "TRADE: lost quality but saved memory"
    # No meaningful memory reduction — non-interference check
    if abs(q_delta) < 0.03 and m_stable and s_delta <= 0:
        return "HOLD: same quality, stable memory, no unnecessary compression"
    if q_delta > 0 and m_stable:
        return "HOLD: better quality, stable memory, no unnecessary compression"
    if q_delta < -0.05:
        return "FAIL: lost quality without memory savings"
    return "NEUTRAL: no significant difference"
