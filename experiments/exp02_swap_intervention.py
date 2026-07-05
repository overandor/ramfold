#!/usr/bin/env python3
"""
Experiment 02: Swap Intervention — the missing evidence.

The M5 Pro with 18 GB unified memory is too capable. It compresses
rather than swaps. To prove RAMFold v2 intervenes correctly under
real swap danger, we pre-allocate memory to force swap growth,
then train under fixed vs adaptive.

Hypothesis:
  Under artificial memory pressure that causes swap growth,
  RAMFold v2 will compress policy (shrink seq/batch) to reduce
  swap, while preserving as much quality as possible.
  Fixed policy will either crash or degrade under the same pressure.

Win condition:
  Adaptive has lower peak swap, fewer crashes, and QPG >= fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ramfold.trainers import CharDataset, TrialConfig, run_trial, result_to_metrics, get_dataset
from ramfold.controller import MemoryPolicy
from ramfold.receipts import ReceiptLedger
from ramfold.verification import compare_trials

TIERS = {
    "large": {
        "embed_dim": 512, "num_heads": 8, "num_layers": 6, "ff_mult": 4,
        "batch_size": 4, "seq_len": 512,
    },
    "stress": {
        "embed_dim": 512, "num_heads": 8, "num_layers": 8, "ff_mult": 4,
        "batch_size": 8, "seq_len": 768,
    },
}


def allocate_pressure_gb(gb: float):
    """Pre-allocate GB of memory to force swap growth."""
    import numpy as np
    blocks = []
    block_size = 256 * 1024 * 1024  # 256 MB blocks
    num_blocks = int(gb * 1024 * 1024 * 1024 / block_size)
    print(f"  ⚠ Pre-allocating {gb:.1f} GB to force memory pressure...")
    for i in range(num_blocks):
        arr = np.ones(block_size // 4, dtype=np.float32)
        # Touch every page to ensure real allocation
        arr[::4096] = float(i)
        blocks.append(arr)
    print(f"  ⚠ Allocated {len(blocks) * block_size / 1024**3:.1f} GB — swap should now grow during training")
    return blocks


def main():
    ap = argparse.ArgumentParser(description="RAMFold Experiment 02: Swap Intervention")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tier", type=str, default="large", choices=list(TIERS.keys()))
    ap.add_argument("--pressure-gb", type=float, default=10.0,
                    help="GB of memory to pre-allocate and hold during training")
    ap.add_argument("--memory-budget-gb", type=float, default=18.0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--receipts", default=None)
    args = ap.parse_args()

    tier_cfg = TIERS[args.tier]
    ds = get_dataset()

    output_path = args.output or f"results/exp02_{args.tier}_{args.steps}steps_p{int(args.pressure_gb)}gb.json"
    receipts_path = args.receipts or f"results/exp02_{args.tier}_{args.steps}steps_p{int(args.pressure_gb)}gb_receipts.jsonl"

    ledger = ReceiptLedger(receipts_path)
    run_id_base = f"exp02_{args.tier}_{int(time.time())}"

    # Pre-allocate memory pressure
    pressure_blocks = allocate_pressure_gb(args.pressure_gb)

    results = {}

    # Fixed baseline
    fixed_policy = MemoryPolicy(
        seq_len=tier_cfg["seq_len"],
        batch_size=tier_cfg["batch_size"],
        name=f"{args.tier}_fixed_pressure",
    )
    fixed_cfg = TrialConfig(
        name=f"{args.tier}_fixed_pressure",
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

    # Adaptive — should detect swap and compress
    adaptive_policy = MemoryPolicy(
        seq_len=tier_cfg["seq_len"],
        batch_size=tier_cfg["batch_size"],
        name=f"{args.tier}_adaptive_pressure",
    )
    adaptive_cfg = TrialConfig(
        name=f"{args.tier}_adaptive_pressure",
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

    # Release pressure
    del pressure_blocks
    print(f"\n  ◌ Released {args.pressure_gb:.1f} GB pressure allocation")

    # Compute metrics
    fixed_metrics = result_to_metrics(fixed_result, args.memory_budget_gb)
    adaptive_metrics = result_to_metrics(adaptive_result, args.memory_budget_gb)
    comparison = compare_trials(fixed_metrics, adaptive_metrics)

    # Print results
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 02: Swap Intervention — Tier: {args.tier.upper()} — Pressure: {args.pressure_gb} GB")
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

    # Show action distribution for adaptive
    if adaptive_result.actions_log:
        from collections import Counter
        actions = Counter(adaptive_result.actions_log)
        print(f"\n  ADAPTIVE ACTIONS:")
        for action, count in actions.most_common():
            print(f"    {action:<40} {count:>5}")

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
        "experiment": "exp02_swap_intervention",
        "timestamp": time.time(),
        "tier": args.tier,
        "steps": args.steps,
        "pressure_gb": args.pressure_gb,
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
        "receipts_path": str(receipts_path),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to: {output_path}")
    print(f"  Receipts saved to: {receipts_path} ({ledger.count} receipts)")


if __name__ == "__main__":
    main()
