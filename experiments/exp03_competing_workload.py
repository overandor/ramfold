#!/usr/bin/env python3
"""
Experiment 03: Competing Workload — real swap trigger.

Loads a large Ollama model into memory (consuming 4-5 GB of unified memory),
then runs the RAMFold experiment. The competing workload should push
the system past compression and into real swap growth.

This is the experiment that triggers v2's swap-weighted intervention.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import threading
import urllib.request
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


def load_ollama_model(model: str):
    """Load a model into Ollama memory and keep it warm."""
    print(f"  ⚠ Loading Ollama model: {model} ...")
    try:
        req = urllib.request.Request(
            f"http://localhost:11434/api/generate",
            data=json.dumps({
                "model": model,
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 1},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=120)
        _ = resp.read()
        print(f"  ⚠ {model} loaded into Ollama memory")
        return True
    except Exception as e:
        print(f"  ⚠ Failed to load {model}: {e}")
        return False


def keep_ollama_warm(model: str, stop_event: threading.Event):
    """Periodically ping Ollama to keep model in memory."""
    while not stop_event.is_set():
        try:
            req = urllib.request.Request(
                f"http://localhost:11434/api/generate",
                data=json.dumps({
                    "model": model,
                    "prompt": "ok",
                    "stream": False,
                    "options": {"num_predict": 1},
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            pass
        stop_event.wait(10)


def allocate_pressure_gb(gb: float):
    """Pre-allocate GB of memory to force swap growth."""
    import numpy as np
    blocks = []
    block_size = 256 * 1024 * 1024
    num_blocks = int(gb * 1024 * 1024 * 1024 / block_size)
    print(f"  ⚠ Pre-allocating {gb:.1f} GB ...")
    for i in range(num_blocks):
        arr = np.ones(block_size // 4, dtype=np.float32)
        arr[::4096] = float(i)
        blocks.append(arr)
    print(f"  ⚠ Allocated {len(blocks) * block_size / 1024**3:.1f} GB")
    return blocks


def main():
    ap = argparse.ArgumentParser(description="RAMFold Experiment 03: Competing Workload")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tier", type=str, default="stress", choices=list(TIERS.keys()))
    ap.add_argument("--ollama-model", type=str, default="llama3.1:8b",
                    help="Ollama model to load as competing workload")
    ap.add_argument("--extra-pressure-gb", type=float, default=8.0,
                    help="Additional artificial memory pressure")
    ap.add_argument("--memory-budget-gb", type=float, default=18.0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--receipts", default=None)
    args = ap.parse_args()

    tier_cfg = TIERS[args.tier]
    ds = get_dataset()

    output_path = args.output or f"results/exp03_{args.tier}_{args.steps}steps_{args.ollama_model.replace(':','_')}.json"
    receipts_path = args.receipts or f"results/exp03_{args.tier}_{args.steps}steps_receipts.jsonl"

    ledger = ReceiptLedger(receipts_path)
    run_id_base = f"exp03_{args.tier}_{int(time.time())}"

    # Step 1: Load competing Ollama model
    print(f"\n{'='*60}")
    print(f"  EXP03: Competing Workload + Artificial Pressure")
    print(f"{'='*60}")
    ollama_loaded = load_ollama_model(args.ollama_model)

    # Step 2: Start keep-warm thread
    stop_warm = threading.Event()
    if ollama_loaded:
        warm_thread = threading.Thread(target=keep_ollama_warm, args=(args.ollama_model, stop_warm), daemon=True)
        warm_thread.start()
        print(f"  ⚠ Keep-warm thread started for {args.ollama_model}")

    # Step 3: Pre-allocate additional pressure
    pressure_blocks = None
    if args.extra_pressure_gb > 0:
        pressure_blocks = allocate_pressure_gb(args.extra_pressure_gb)

    # Check memory state before training
    from ramfold.observer import MemoryObserver
    pre_observer = MemoryObserver(budget_gb=args.memory_budget_gb)
    pre_snap = pre_observer.snapshot()
    print(f"\n  Pre-training memory state:")
    print(f"    pressure={pre_snap.pressure:.3f} swap={pre_snap.swap_used_gb:.2f} GB")
    print(f"    active={pre_snap.active_gb:.2f} wired={pre_snap.wired_gb:.2f} compressed={pre_snap.compressed_gb:.2f}")
    print(f"    mlx_active={pre_snap.mlx_active_mb:.0f} MB")

    results = {}

    # Fixed baseline
    fixed_policy = MemoryPolicy(
        seq_len=tier_cfg["seq_len"],
        batch_size=tier_cfg["batch_size"],
        name=f"{args.tier}_fixed_competing",
    )
    fixed_cfg = TrialConfig(
        name=f"{args.tier}_fixed_competing",
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

    # Adaptive
    adaptive_policy = MemoryPolicy(
        seq_len=tier_cfg["seq_len"],
        batch_size=tier_cfg["batch_size"],
        name=f"{args.tier}_adaptive_competing",
    )
    adaptive_cfg = TrialConfig(
        name=f"{args.tier}_adaptive_competing",
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

    # Cleanup
    stop_warm.set()
    del pressure_blocks
    print(f"\n  ◌ Released pressure allocation and stopped keep-warm")

    # Compute metrics
    fixed_metrics = result_to_metrics(fixed_result, args.memory_budget_gb)
    adaptive_metrics = result_to_metrics(adaptive_result, args.memory_budget_gb)
    comparison = compare_trials(fixed_metrics, adaptive_metrics)

    # Print results
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 03: Competing Workload — Tier: {args.tier.upper()} — Model: {args.ollama_model} — Extra: {args.extra_pressure_gb} GB")
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

    if adaptive_result.actions_log:
        from collections import Counter
        actions = Counter(adaptive_result.actions_log)
        print(f"\n  ADAPTIVE ACTIONS:")
        for action, count in actions.most_common():
            print(f"    {action:<45} {count:>5}")

    print(f"\n  QUALITY METRICS:")
    print(f"  {'Metric':<25} {'Fixed':>12} {'Adaptive':>12}")
    print(f"  {'-'*49}")
    print(f"  {'Quality score':<25} {fixed_metrics.quality_score:>12.4f} {adaptive_metrics.quality_score:>12.4f}")
    print(f"  {'QPG':<25} {fixed_metrics.qpg:>12.4f} {adaptive_metrics.qpg:>12.4f}")
    print(f"  {'SFR':<25} {fixed_metrics.sfr:>12.4f} {adaptive_metrics.sfr:>12.4f}")
    mer_str = f"{comparison['mer']:.4f}" if comparison['mer'] is not None else "N/A"
    print(f"  {'MER':<25} {'N/A':>12} {mer_str:>12}")
    print(f"\n  VERDICT: {comparison['verdict']}")

    # Save
    output = {
        "experiment": "exp03_competing_workload",
        "timestamp": time.time(),
        "tier": args.tier,
        "steps": args.steps,
        "ollama_model": args.ollama_model,
        "extra_pressure_gb": args.extra_pressure_gb,
        "memory_budget_gb": args.memory_budget_gb,
        "pre_training": {
            "pressure": pre_snap.pressure,
            "swap_gb": pre_snap.swap_used_gb,
            "active_gb": pre_snap.active_gb,
            "wired_gb": pre_snap.wired_gb,
            "compressed_gb": pre_snap.compressed_gb,
        },
        "fixed": {k: getattr(fixed_result, k) for k in [
            "final_loss", "min_loss", "avg_loss", "avg_tokens_per_sec",
            "peak_pressure", "avg_pressure", "peak_swap_gb",
            "model_mem_mb", "mlx_peak_mb", "total_seconds",
            "crashed", "policy_changes", "final_seq_len", "final_batch_size",
        ]},
        "adaptive": {k: getattr(adaptive_result, k) for k in [
            "final_loss", "min_loss", "avg_loss", "avg_tokens_per_sec",
            "peak_pressure", "avg_pressure", "peak_swap_gb",
            "model_mem_mb", "mlx_peak_mb", "total_seconds",
            "crashed", "policy_changes", "final_seq_len", "final_batch_size",
        ]},
        "comparison": comparison,
        "receipts_path": str(receipts_path),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to: {output_path}")
    print(f"  Receipts saved to: {receipts_path} ({ledger.count} receipts)")


if __name__ == "__main__":
    main()
