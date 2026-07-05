#!/usr/bin/env python3
"""
Experiment 01: Fixed Policy vs Adaptive RAMFold

Hypothesis:
  Adaptive memory policy lowers peak unified-memory pressure without
  materially degrading validation loss.

Procedure:
  Train the same MLX transformer under two regimes:
  - Fixed: static batch/seq/checkpointing throughout
  - Adaptive: RAMFold governor adjusts policy based on memory state

Measure:
  peak memory pressure, swap, tokens/sec, loss, quality score, QPG

Win condition:
  RAMFold has lower peak memory and equal-or-better quality per GB.

Usage:
  python exp01_fixed_vs_adaptive.py --steps 200 --tier large
  python exp01_fixed_vs_adaptive.py --steps 80 --tier stress
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ramfold.trainers import CharDataset, TrialConfig, run_trial, result_to_metrics, get_dataset
from ramfold.controller import MemoryPolicy, BanditController, ParetoFrontier
from ramfold.receipts import ReceiptLedger
from ramfold.verification import compare_trials

TIERS = {
    "tiny": {
        "embed_dim": 64, "num_heads": 4, "num_layers": 2, "ff_mult": 4,
        "batch_size": 8, "seq_len": 128,
    },
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


def main():
    ap = argparse.ArgumentParser(description="RAMFold Experiment 01: Fixed vs Adaptive")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tier", type=str, default="large", choices=list(TIERS.keys()))
    ap.add_argument("--memory-budget-gb", type=float, default=18.0)
    ap.add_argument("--output", default=None, help="Output JSON path")
    ap.add_argument("--receipts", default=None, help="Receipt ledger JSONL path")
    ap.add_argument("--bandit-trials", type=int, default=5, help="Number of bandit policy-search trials")
    args = ap.parse_args()

    tier_cfg = TIERS[args.tier]
    ds = get_dataset()

    output_path = args.output or f"results/exp01_{args.tier}_{args.steps}steps.json"
    receipts_path = args.receipts or f"results/exp01_{args.tier}_receipts.jsonl"

    ledger = ReceiptLedger(receipts_path)

    run_id_base = f"exp01_{args.tier}_{int(time.time())}"

    # Fixed baseline policy
    fixed_policy = MemoryPolicy(
        seq_len=tier_cfg["seq_len"],
        batch_size=tier_cfg["batch_size"],
        name=f"{args.tier}_fixed",
    )

    # Adaptive starts with same policy, governor will adjust
    adaptive_policy = MemoryPolicy(
        seq_len=tier_cfg["seq_len"],
        batch_size=tier_cfg["batch_size"],
        name=f"{args.tier}_adaptive",
    )

    results = {}

    # Run fixed baseline
    fixed_cfg = TrialConfig(
        name=f"{args.tier}_fixed",
        mode="fixed",
        steps=args.steps,
        embed_dim=tier_cfg["embed_dim"],
        num_heads=tier_cfg["num_heads"],
        num_layers=tier_cfg["num_layers"],
        ff_mult=tier_cfg["ff_mult"],
        memory_budget_gb=args.memory_budget_gb,
    )
    fixed_result = run_trial(fixed_cfg, fixed_policy, ds, ledger, run_id=f"{run_id_base}_fixed")
    results["fixed"] = fixed_result

    # Run adaptive
    adaptive_cfg = TrialConfig(
        name=f"{args.tier}_adaptive",
        mode="adaptive",
        steps=args.steps,
        embed_dim=tier_cfg["embed_dim"],
        num_heads=tier_cfg["num_heads"],
        num_layers=tier_cfg["num_layers"],
        ff_mult=tier_cfg["ff_mult"],
        memory_budget_gb=args.memory_budget_gb,
    )
    adaptive_result = run_trial(adaptive_cfg, adaptive_policy, ds, ledger, run_id=f"{run_id_base}_adaptive")
    results["adaptive"] = adaptive_result

    # Bandit policy-search trials
    bandit = BanditController()
    pareto = ParetoFrontier()
    bandit_results = []

    print(f"\n{'='*60}")
    print(f"  BANDIT POLICY SEARCH ({args.bandit_trials} trials)")
    print(f"{'='*60}")

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
            name=f"bandit_{arm.name}",
        )
        bandit_cfg = TrialConfig(
            name=f"bandit_{args.tier}_{bt}",
            mode="fixed",
            steps=max(20, args.steps // 4),
            embed_dim=tier_cfg["embed_dim"],
            num_heads=tier_cfg["num_heads"],
            num_layers=tier_cfg["num_layers"],
            ff_mult=tier_cfg["ff_mult"],
            memory_budget_gb=args.memory_budget_gb,
        )
        bresult = run_trial(bandit_cfg, bandit_policy, ds, ledger, run_id=f"{run_id_base}_bandit_{bt}")
        bmetrics = result_to_metrics(bresult, args.memory_budget_gb)
        bandit.update(
            arm=arm,
            quality=bmetrics.quality_score,
            pressure=bresult.peak_pressure,
            swap_delta=bresult.peak_swap_gb,
            latency_s=bresult.total_seconds,
        )
        pareto.add(arm)
        bandit_results.append({
            "arm": arm.name,
            "policy": bandit_policy.to_dict(),
            "final_loss": bresult.final_loss,
            "quality_score": bmetrics.quality_score,
            "peak_gb": bmetrics.peak_gb,
            "qpg": bmetrics.qpg,
            "mlx_peak_mb": bresult.mlx_peak_mb,
            "reward": bandit.compute_reward(bmetrics.quality_score, bresult.peak_pressure, bresult.peak_swap_gb, bresult.total_seconds),
            "avg_reward": arm.avg_reward,
        })
        print(f"  bandit trial {bt}: arm={arm.name} loss={bresult.final_loss:.4f} "
              f"quality={bmetrics.quality_score:.4f} peak_gb={bmetrics.peak_gb:.4f} "
              f"qpg={bmetrics.qpg:.4f}")

    # Pareto frontier
    frontier = pareto.frontier()
    print(f"\n  PARETO FRONTIER ({len(frontier)} non-dominated policies):")
    for point in frontier:
        print(f"    {point['name']}: quality={point['quality']:.4f} peak_gb={point['memory']:.4f} pulls={point['pulls']}")

    results["bandit"] = bandit_results
    results["pareto"] = frontier

    # Compute metrics
    fixed_metrics = result_to_metrics(fixed_result, args.memory_budget_gb)
    adaptive_metrics = result_to_metrics(adaptive_result, args.memory_budget_gb)
    comparison = compare_trials(fixed_metrics, adaptive_metrics)

    # Print results table
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 01: Fixed vs Adaptive — Tier: {args.tier.upper()}")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Fixed':>12} {'Adaptive':>12} {'Delta':>12}")
    print(f"{'-'*61}")

    def row(label, getter, fmt="{:.4f}"):
        f_val = fmt.format(getter(fixed_result)) if fixed_result else "N/A"
        a_val = fmt.format(getter(adaptive_result)) if adaptive_result else "N/A"
        try:
            delta = fmt.format(getter(adaptive_result) - getter(fixed_result))
        except Exception:
            delta = "N/A"
        print(f"{label:<25} {f_val:>12} {a_val:>12} {delta:>12}")

    row("Final loss", lambda r: r.final_loss)
    row("Min loss", lambda r: r.min_loss)
    row("Avg loss", lambda r: r.avg_loss)
    row("Avg tok/s", lambda r: r.avg_tokens_per_sec, fmt="{:.0f}")
    row("Peak pressure", lambda r: r.peak_pressure, fmt="{:.4f}")
    row("Avg pressure", lambda r: r.avg_pressure, fmt="{:.4f}")
    row("Peak swap (GB)", lambda r: r.peak_swap_gb, fmt="{:.4f}")
    row("Model mem (MB)", lambda r: r.model_mem_mb, fmt="{:.1f}")
    row("MLX peak (MB)", lambda r: r.mlx_peak_mb, fmt="{:.1f}")
    row("Total time (s)", lambda r: r.total_seconds, fmt="{:.1f}")

    print(f"{'Crashed':<25} {'YES' if fixed_result.crashed else 'NO':>12} {'YES' if adaptive_result.crashed else 'NO':>12}")
    print(f"{'Policy changes':<25} {fixed_result.policy_changes:>12} {adaptive_result.policy_changes:>12}")
    print(f"{'Final seq/batch':<25} {f'{fixed_result.final_seq_len}/{fixed_result.final_batch_size}':>12} {f'{adaptive_result.final_seq_len}/{adaptive_result.final_batch_size}':>12}")
    print(f"{'Final grad_accum':<25} {fixed_result.final_grad_accum:>12} {adaptive_result.final_grad_accum:>12}")
    print(f"{'Final checkpoint':<25} {fixed_result.final_checkpoint_level:>12} {adaptive_result.final_checkpoint_level:>12}")
    print(f"{'Final kv_bits':<25} {fixed_result.final_kv_bits:>12} {adaptive_result.final_kv_bits:>12}")
    print(f"{'Final emb_bits':<25} {fixed_result.final_embedding_bits:>12} {adaptive_result.final_embedding_bits:>12}")
    print(f"{'Final compress_ratio':<25} {fixed_result.final_compression_ratio:>12.2f} {adaptive_result.final_compression_ratio:>12.2f}")

    print(f"\n  QUALITY METRICS:")
    print(f"  {'Metric':<25} {'Fixed':>12} {'Adaptive':>12}")
    print(f"  {'-'*49}")
    print(f"  {'Quality score':<25} {fixed_metrics.quality_score:>12.4f} {adaptive_metrics.quality_score:>12.4f}")
    print(f"  {'Verification score':<25} {fixed_metrics.verification_score:>12.4f} {adaptive_metrics.verification_score:>12.4f}")
    print(f"  {'QPG':<25} {fixed_metrics.qpg:>12.4f} {adaptive_metrics.qpg:>12.4f}")
    print(f"  {'SFR':<25} {fixed_metrics.sfr:>12.4f} {adaptive_metrics.sfr:>12.4f}")
    mer_str = f"{comparison['mer']:.4f}" if comparison['mer'] is not None else "N/A"
    print(f"  {'MER':<25} {'N/A':>12} {mer_str:>12}")

    print(f"\n  VERDICT: {comparison['verdict']}")

    # Save results
    output = {
        "experiment": "exp01_fixed_vs_adaptive",
        "timestamp": time.time(),
        "tier": args.tier,
        "steps": args.steps,
        "memory_budget_gb": args.memory_budget_gb,
        "fixed": {
            "final_loss": fixed_result.final_loss,
            "min_loss": fixed_result.min_loss,
            "avg_loss": fixed_result.avg_loss,
            "avg_tokens_per_sec": fixed_result.avg_tokens_per_sec,
            "peak_pressure": fixed_result.peak_pressure,
            "avg_pressure": fixed_result.avg_pressure,
            "peak_swap_gb": fixed_result.peak_swap_gb,
            "model_mem_mb": fixed_result.model_mem_mb,
            "mlx_peak_mb": fixed_result.mlx_peak_mb,
            "total_seconds": fixed_result.total_seconds,
            "crashed": fixed_result.crashed,
            "policy_changes": fixed_result.policy_changes,
            "final_seq_len": fixed_result.final_seq_len,
            "final_batch_size": fixed_result.final_batch_size,
            "quality_score": fixed_metrics.quality_score,
            "verification_score": fixed_metrics.verification_score,
            "qpg": fixed_metrics.qpg,
            "sfr": fixed_metrics.sfr,
        },
        "adaptive": {
            "final_loss": adaptive_result.final_loss,
            "min_loss": adaptive_result.min_loss,
            "avg_loss": adaptive_result.avg_loss,
            "avg_tokens_per_sec": adaptive_result.avg_tokens_per_sec,
            "peak_pressure": adaptive_result.peak_pressure,
            "avg_pressure": adaptive_result.avg_pressure,
            "peak_swap_gb": adaptive_result.peak_swap_gb,
            "model_mem_mb": adaptive_result.model_mem_mb,
            "mlx_peak_mb": adaptive_result.mlx_peak_mb,
            "total_seconds": adaptive_result.total_seconds,
            "crashed": adaptive_result.crashed,
            "policy_changes": adaptive_result.policy_changes,
            "final_seq_len": adaptive_result.final_seq_len,
            "final_batch_size": adaptive_result.final_batch_size,
            "quality_score": adaptive_metrics.quality_score,
            "verification_score": adaptive_metrics.verification_score,
            "qpg": adaptive_metrics.qpg,
            "sfr": adaptive_metrics.sfr,
        },
        "comparison": comparison,
        "bandit_trials": bandit_results,
        "pareto_frontier": frontier,
        "receipts_path": str(receipts_path),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to: {output_path}")
    print(f"  Receipts saved to: {receipts_path} ({ledger.count} receipts)")


if __name__ == "__main__":
    main()
