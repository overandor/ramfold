#!/usr/bin/env python3
"""
Experiment 04: Five-Arm Memory Policy Comparison

The definitive experiment proving the RAMFold thesis:

  μ* = argmaxμ [VerifiedQuality(θ, C_μ(D))
                - λ1·PeakMemory - λ2·Swap - λ3·Latency
                - λ4·Recompute - λ5·CloudCost]

Five arms:
  A. Full fixed policy (max context, no compression)
  B. Static compressed policy (short seq, 50% compression, 8-bit KV)
  C. Static checkpointed policy (checkpoint=2, 8-bit KV, 8-bit emb)
  D. RAMFold adaptive governor (swap-weighted, reacts to memory state)
  E. RAMFold bandit/Pareto policy search (UCB over policy grid)

The breakthrough claim:
  "The optimal local LLM policy is not the largest context that fits.
   It is the smallest verified context that preserves task outcome."

If D or E achieves same verified quality as A with lower memory/swap,
the thesis is proven. If B or C loses quality, static compression is
shown to be insufficient. If A crashes under pressure, full context
is shown to be fragile.

Each arm produces receipts. The Pareto frontier is plotted from all arms.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ramfold.trainers import (
    CharDataset, TrialConfig, run_trial, result_to_metrics, get_dataset,
)
from ramfold.controller import MemoryPolicy, BanditController, ParetoFrontier
from ramfold.receipts import ReceiptLedger
from ramfold.verification import QualityMetrics, PolicyObjective, compare_trials


TIERS = {
    "medium": {
        "embed_dim": 256, "num_heads": 8, "num_layers": 4, "ff_mult": 4,
        "batch_size": 8, "seq_len": 256,
    },
    "large": {
        "embed_dim": 512, "num_heads": 8, "num_layers": 6, "ff_mult": 4,
        "batch_size": 4, "seq_len": 512,
    },
    "stress": {
        "embed_dim": 512, "num_heads": 8, "num_layers": 8, "ff_mult": 4,
        "batch_size": 8, "seq_len": 768,
    },
}


def make_arm_a_full(tier_cfg: dict) -> MemoryPolicy:
    """Arm A: Full fixed policy — max context, no compression, full precision."""
    return MemoryPolicy(
        seq_len=tier_cfg["seq_len"],
        batch_size=tier_cfg["batch_size"],
        grad_accum=1,
        checkpoint_level=0,
        kv_bits=32,
        embedding_bits=32,
        context_compression_ratio=1.0,
        name="A_full_fixed",
    )


def make_arm_b_static_compressed(tier_cfg: dict) -> MemoryPolicy:
    """Arm B: Static compressed — short seq, 50% compression, 8-bit KV."""
    return MemoryPolicy(
        seq_len=tier_cfg["seq_len"] // 2,
        batch_size=tier_cfg["batch_size"],
        grad_accum=2,
        checkpoint_level=0,
        kv_bits=8,
        embedding_bits=32,
        context_compression_ratio=0.5,
        name="B_static_compressed",
    )


def make_arm_c_static_checkpointed(tier_cfg: dict) -> MemoryPolicy:
    """Arm C: Static checkpointed — full checkpointing, 8-bit KV, 8-bit emb."""
    return MemoryPolicy(
        seq_len=tier_cfg["seq_len"],
        batch_size=tier_cfg["batch_size"],
        grad_accum=2,
        checkpoint_level=2,
        kv_bits=8,
        embedding_bits=8,
        context_compression_ratio=1.0,
        name="C_static_checkpointed",
    )


def make_arm_d_adaptive(tier_cfg: dict) -> MemoryPolicy:
    """Arm D: RAMFold adaptive governor — starts full, governor adjusts."""
    return MemoryPolicy(
        seq_len=tier_cfg["seq_len"],
        batch_size=tier_cfg["batch_size"],
        grad_accum=1,
        checkpoint_level=0,
        kv_bits=32,
        embedding_bits=32,
        context_compression_ratio=1.0,
        name="D_adaptive_governor",
    )


def main():
    ap = argparse.ArgumentParser(
        description="RAMFold Experiment 04: Five-Arm Memory Policy Comparison"
    )
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--tier", type=str, default="large", choices=list(TIERS.keys()))
    ap.add_argument("--memory-budget-gb", type=float, default=18.0)
    ap.add_argument("--bandit-trials", type=int, default=5)
    ap.add_argument("--pressure-gb", type=float, default=0.0,
                    help="Pre-allocate N GB to force memory pressure")
    ap.add_argument("--output", default=None)
    ap.add_argument("--receipts", default=None)
    args = ap.parse_args()

    tier_cfg = TIERS[args.tier]
    ds = get_dataset()

    output_path = args.output or f"results/exp04_{args.tier}_{args.steps}steps.json"
    receipts_path = args.receipts or f"results/exp04_{args.tier}_receipts.jsonl"
    ledger = ReceiptLedger(receipts_path)

    run_id_base = f"exp04_{args.tier}_{int(time.time())}"

    # Optional: pre-allocate memory to force pressure
    pressure_blocks = []
    if args.pressure_gb > 0:
        import numpy as np
        block_size = 256 * 1024 * 1024
        num_blocks = int(args.pressure_gb * 1024**3 / block_size)
        print(f"\n  Pre-allocating {args.pressure_gb:.1f} GB to force memory pressure...")
        for i in range(num_blocks):
            arr = np.ones(block_size // 4, dtype=np.float32)
            arr[::4096] = float(i)
            pressure_blocks.append(arr)
        print(f"  Allocated {len(pressure_blocks) * block_size / 1024**3:.1f} GB")

    # Define all arms
    arms = {
        "A_full": {
            "policy": make_arm_a_full(tier_cfg),
            "mode": "fixed",
            "label": "A: Full fixed (max context, full precision)",
        },
        "B_static_compressed": {
            "policy": make_arm_b_static_compressed(tier_cfg),
            "mode": "fixed",
            "label": "B: Static compressed (short seq, 50% cr, 8-bit KV)",
        },
        "C_static_checkpointed": {
            "policy": make_arm_c_static_checkpointed(tier_cfg),
            "mode": "fixed",
            "label": "C: Static checkpointed (ckpt=2, 8-bit KV, 8-bit emb)",
        },
        "D_adaptive": {
            "policy": make_arm_d_adaptive(tier_cfg),
            "mode": "adaptive",
            "label": "D: RAMFold adaptive governor",
        },
    }

    # Formal objective
    objective = PolicyObjective()

    all_results = {}
    all_metrics = {}
    all_j_scores = {}

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 04: FIVE-ARM MEMORY POLICY COMPARISON")
    print(f"  Tier: {args.tier.upper()}  Steps: {args.steps}  Budget: {args.memory_budget_gb} GB")
    print(f"  Objective: J(θ,μ) = VerifiedQuality - λ1·Mem - λ2·Swap - λ3·Latency")
    print(f"{'='*70}")

    # Run arms A-D
    for arm_id, arm_cfg in arms.items():
        cfg = TrialConfig(
            name=arm_id,
            mode=arm_cfg["mode"],
            steps=args.steps,
            embed_dim=tier_cfg["embed_dim"],
            num_heads=tier_cfg["num_heads"],
            num_layers=tier_cfg["num_layers"],
            ff_mult=tier_cfg["ff_mult"],
            memory_budget_gb=args.memory_budget_gb,
        )
        result = run_trial(cfg, arm_cfg["policy"], ds, ledger,
                          run_id=f"{run_id_base}_{arm_id}")
        metrics = result_to_metrics(result, args.memory_budget_gb)
        recompute_cost = 0.1 * arm_cfg["policy"].checkpoint_level
        j_score = objective.evaluate(
            metrics,
            recompute_cost=recompute_cost,
            cloud_cost=0.0,
            latency_s=result.total_seconds,
        )
        all_results[arm_id] = result
        all_metrics[arm_id] = metrics
        all_j_scores[arm_id] = j_score
        print(f"  J({arm_id}) = {j_score:.6f}")

    # Arm E: Bandit/Pareto policy search
    print(f"\n{'='*60}")
    print(f"  ARM E: BANDIT/PARETO POLICY SEARCH ({args.bandit_trials} trials)")
    print(f"{'='*60}")

    bandit = BanditController()
    pareto = ParetoFrontier()
    bandit_results = []

    for bt in range(args.bandit_trials):
        arm = bandit.select()
        bandit_policy = MemoryPolicy(
            seq_len=arm.seq_len,
            batch_size=arm.batch_size,
            grad_accum=arm.grad_accum,
            checkpoint_level=arm.checkpoint_level,
            kv_bits=arm.kv_bits,
            embedding_bits=arm.embedding_bits,
            context_compression_ratio=arm.context_compression_ratio,
            name=f"E_bandit_{arm.name}",
        )
        bandit_cfg = TrialConfig(
            name=f"E_bandit_{args.tier}_{bt}",
            mode="fixed",
            steps=max(20, args.steps // 3),
            embed_dim=tier_cfg["embed_dim"],
            num_heads=tier_cfg["num_heads"],
            num_layers=tier_cfg["num_layers"],
            ff_mult=tier_cfg["ff_mult"],
            memory_budget_gb=args.memory_budget_gb,
        )
        bresult = run_trial(bandit_cfg, bandit_policy, ds, ledger,
                           run_id=f"{run_id_base}_E_bandit_{bt}")
        bmetrics = result_to_metrics(bresult, args.memory_budget_gb)
        bandit.update(
            arm=arm,
            quality=bmetrics.quality_score,
            pressure=bresult.peak_pressure,
            swap_delta=bresult.peak_swap_gb,
            latency_s=bresult.total_seconds,
        )
        pareto.add(arm)
        j_score = objective.evaluate(
            bmetrics,
            recompute_cost=0.1 * arm.checkpoint_level,
            cloud_cost=0.0,
            latency_s=bresult.total_seconds,
        )
        bandit_results.append({
            "arm": arm.name,
            "policy": bandit_policy.to_dict(),
            "final_loss": bresult.final_loss,
            "quality_score": bmetrics.quality_score,
            "verification_score": bmetrics.verification_score,
            "peak_gb": bmetrics.peak_gb,
            "peak_swap_gb": bresult.peak_swap_gb,
            "qpg": bmetrics.qpg,
            "j_score": j_score,
            "mlx_peak_mb": bresult.mlx_peak_mb,
            "policy_changes": bresult.policy_changes,
        })
        print(f"  bandit {bt}: arm={arm.name} J={j_score:.6f} "
              f"quality={bmetrics.quality_score:.4f} peak_gb={bmetrics.peak_gb:.4f} "
              f"swap={bresult.peak_swap_gb:.4f}")

    # Best bandit arm
    best_bandit = bandit.best_arm()
    best_bandit_name = best_bandit.name if best_bandit else "none"
    best_bandit_j = max(bandit_results, key=lambda x: x["j_score"]) if bandit_results else None

    # Pareto frontier from all arms
    pareto_frontier = pareto.frontier()
    pareto_best = pareto.best_quality_per_gb()

    # Add A-D to Pareto manually
    all_pareto_points = []
    for arm_id, metrics in all_metrics.items():
        all_pareto_points.append({
            "arm": arm_id,
            "quality": round(metrics.quality_score, 4),
            "peak_gb": round(metrics.peak_gb, 4),
            "qpg": metrics.qpg,
            "j_score": all_j_scores[arm_id],
            "swap": round(metrics.peak_swap_gb, 4),
            "verification": round(metrics.verification_score, 4),
        })
    for br in bandit_results:
        all_pareto_points.append({
            "arm": br["arm"],
            "quality": br["quality_score"],
            "peak_gb": br["peak_gb"],
            "qpg": br["qpg"],
            "j_score": br["j_score"],
            "swap": br["peak_swap_gb"],
            "verification": br["verification_score"],
        })

    # Sort by J score (the formal objective)
    all_pareto_points.sort(key=lambda x: x["j_score"], reverse=True)

    # Print results table
    print(f"\n{'='*70}")
    print(f"  RESULTS: FIVE-ARM COMPARISON")
    print(f"{'='*70}")
    print(f"{'Arm':<30} {'Loss':>8} {'Quality':>8} {'Verify':>8} "
          f"{'PeakGB':>8} {'SwapGB':>8} {'QPG':>8} {'J(θ,μ)':>10}")
    print(f"{'-'*98}")

    for arm_id in ["A_full", "B_static_compressed", "C_static_checkpointed", "D_adaptive"]:
        r = all_results[arm_id]
        m = all_metrics[arm_id]
        print(f"{arm_id:<30} {r.final_loss:>8.4f} {m.quality_score:>8.4f} "
              f"{m.verification_score:>8.4f} {m.peak_gb:>8.4f} "
              f"{r.peak_swap_gb:>8.4f} {m.qpg:>8.4f} "
              f"{all_j_scores[arm_id]:>10.6f}")

    if best_bandit_j:
        print(f"{'E_best_bandit':<30} {best_bandit_j['final_loss']:>8.4f} "
              f"{best_bandit_j['quality_score']:>8.4f} "
              f"{best_bandit_j['verification_score']:>8.4f} "
              f"{best_bandit_j['peak_gb']:>8.4f} "
              f"{best_bandit_j['peak_swap_gb']:>8.4f} "
              f"{best_bandit_j['qpg']:>8.4f} "
              f"{best_bandit_j['j_score']:>10.6f}")

    # Pareto frontier
    print(f"\n  PARETO FRONTIER (all arms, sorted by J):")
    for pt in all_pareto_points:
        print(f"    {pt['arm']:<30} J={pt['j_score']:>10.6f} "
              f"quality={pt['quality']:.4f} peak_gb={pt['peak_gb']:.4f} "
              f"swap={pt['swap']:.4f}")

    # Key comparisons
    print(f"\n  KEY COMPARISONS:")
    a_q = all_metrics["A_full"].quality_score
    a_m = all_metrics["A_full"].peak_gb
    a_s = all_results["A_full"].peak_swap_gb

    for arm_id in ["B_static_compressed", "C_static_checkpointed", "D_adaptive"]:
        m = all_metrics[arm_id]
        r = all_results[arm_id]
        q_delta = m.quality_score - a_q
        m_delta = m.peak_gb - a_m
        s_delta = r.peak_swap_gb - a_s
        verdict = "SAME" if abs(q_delta) < 0.03 else ("BETTER" if q_delta > 0 else "WORSE")
        mem_verdict = "LESS" if m_delta < -0.36 else ("SAME" if abs(m_delta) < 0.36 else "MORE")
        print(f"    {arm_id} vs A_full: quality={verdict}({q_delta:+.4f}) "
              f"memory={mem_verdict}({m_delta:+.4f}GB) swap_delta={s_delta:+.4f}")

    if best_bandit_j:
        q_delta = best_bandit_j["quality_score"] - a_q
        m_delta = best_bandit_j["peak_gb"] - a_m
        s_delta = best_bandit_j["peak_swap_gb"] - a_s
        verdict = "SAME" if abs(q_delta) < 0.03 else ("BETTER" if q_delta > 0 else "WORSE")
        mem_verdict = "LESS" if m_delta < -0.36 else ("SAME" if abs(m_delta) < 0.36 else "MORE")
        print(f"    E_best_bandit vs A_full: quality={verdict}({q_delta:+.4f}) "
              f"memory={mem_verdict}({m_delta:+.4f}GB) swap_delta={s_delta:+.4f}")

    # The breakthrough sentence test
    print(f"\n  BREAKTHROUGH TEST:")
    print(f"  'The optimal policy is not the largest context that fits.")
    print(f"   It is the smallest verified context that preserves task outcome.'")
    best_j_arm = max(all_j_scores, key=all_j_scores.get)
    print(f"  Best J(θ,μ): {best_j_arm} = {all_j_scores[best_j_arm]:.6f}")
    if best_bandit_j and best_bandit_j["j_score"] > all_j_scores.get("A_full", -999):
        print(f"  Bandit found better J than full: {best_bandit_j['j_score']:.6f} > {all_j_scores.get('A_full', 0):.6f}")
        print(f"  THESIS SUPPORTED: adaptive policy beats full context on J(θ,μ)")
    elif all_j_scores.get("D_adaptive", -999) > all_j_scores.get("A_full", -999):
        print(f"  Adaptive governor found better J than full: "
              f"{all_j_scores['D_adaptive']:.6f} > {all_j_scores['A_full']:.6f}")
        print(f"  THESIS SUPPORTED: adaptive policy beats full context on J(θ,μ)")
    else:
        print(f"  THESIS NOT YET PROVEN on this workload/tier — need pressure to trigger difference")
        print(f"  (This is honest: without memory pressure, full context is optimal)")

    # Save results
    output = {
        "experiment": "exp04_five_arm",
        "timestamp": time.time(),
        "tier": args.tier,
        "steps": args.steps,
        "memory_budget_gb": args.memory_budget_gb,
        "pressure_gb": args.pressure_gb,
        "objective": objective.to_dict(),
        "arms": {
            arm_id: {
                "label": arms[arm_id]["label"],
                "policy": arms[arm_id]["policy"].to_dict(),
                "mode": arms[arm_id]["mode"],
                "final_loss": all_results[arm_id].final_loss,
                "min_loss": all_results[arm_id].min_loss,
                "avg_loss": all_results[arm_id].avg_loss,
                "quality_score": all_metrics[arm_id].quality_score,
                "verification_score": all_metrics[arm_id].verification_score,
                "peak_gb": all_metrics[arm_id].peak_gb,
                "peak_swap_gb": all_results[arm_id].peak_swap_gb,
                "qpg": all_metrics[arm_id].qpg,
                "sfr": all_metrics[arm_id].sfr,
                "j_score": all_j_scores[arm_id],
                "mlx_peak_mb": all_results[arm_id].mlx_peak_mb,
                "policy_changes": all_results[arm_id].policy_changes,
                "crashed": all_results[arm_id].crashed,
                "total_seconds": all_results[arm_id].total_seconds,
            }
            for arm_id in arms
        },
        "bandit_trials": bandit_results,
        "best_bandit_arm": best_bandit_name,
        "pareto_frontier": all_pareto_points,
        "pareto_from_bandit": pareto_frontier,
        "receipts_path": str(receipts_path),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to: {output_path}")
    print(f"  Receipts saved to: {receipts_path} ({ledger.count} receipts)")


if __name__ == "__main__":
    main()
