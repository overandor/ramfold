#!/usr/bin/env python3
"""
RAMFold CLI — v0.1

Usage:
  ramfold probe              — sample unified memory state
  ramfold train --policy auto — train with adaptive RAMFold
  ramfold bench --compare baseline,ramfold — run experiment 01
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ramfold.observer import MemoryObserver
from ramfold.controller import MemoryPolicy, SwapWeightedGovernor


def cmd_probe(args):
    """Sample unified memory state."""
    observer = MemoryObserver(budget_gb=args.memory_budget_gb)
    snap = observer.snapshot()
    print(json.dumps({
        "timestamp": snap.timestamp,
        "pressure": snap.pressure,
        "free_gb": snap.free_gb,
        "active_gb": snap.active_gb,
        "wired_gb": snap.wired_gb,
        "compressed_gb": snap.compressed_gb,
        "swap_used_gb": snap.swap_used_gb,
        "mlx_active_mb": snap.mlx_active_mb,
        "mlx_peak_mb": snap.mlx_peak_mb,
        "mlx_cache_mb": snap.mlx_cache_mb,
        "thermal": snap.thermal_pressure,
        "budget_gb": args.memory_budget_gb,
    }, indent=2))


def cmd_train(args):
    """Train with RAMFold memory governor."""
    from ramfold.trainers import CharDataset, TrialConfig, run_trial, get_dataset
    from ramfold.receipts import ReceiptLedger

    ds = get_dataset()
    policy = MemoryPolicy(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        name="train_auto" if args.policy == "auto" else "train_fixed",
    )
    cfg = TrialConfig(
        name=f"cli_train_{args.policy}",
        mode="adaptive" if args.policy == "auto" else "fixed",
        steps=args.steps,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        memory_budget_gb=args.memory_budget_gb,
    )
    ledger = ReceiptLedger(args.receipts)
    result = run_trial(cfg, policy, ds, ledger, run_id=f"cli_{int(time.time())}")

    print(f"\nDone. {result.steps_completed} steps, loss={result.final_loss:.4f}, "
          f"policy_changes={result.policy_changes}, receipts={ledger.count}")


def cmd_bench(args):
    """Run experiment 01: fixed vs adaptive."""
    from experiments.exp01_fixed_vs_adaptive import main as exp01_main
    sys.argv = ["exp01", "--tier", args.tier, "--steps", str(args.steps)]
    if args.output:
        sys.argv += ["--output", args.output]
    exp01_main()


def main():
    ap = argparse.ArgumentParser(prog="ramfold", description="RAMFold: Memory-Elastic Policy for Apple Unified Memory")
    sub = ap.add_subparsers(dest="command")

    # probe
    p_probe = sub.add_parser("probe", help="Sample unified memory state")
    p_probe.add_argument("--memory-budget-gb", type=float, default=18.0)
    p_probe.set_defaults(func=cmd_probe)

    # train
    p_train = sub.add_parser("train", help="Train with RAMFold governor")
    p_train.add_argument("--policy", choices=["auto", "fixed"], default="auto")
    p_train.add_argument("--steps", type=int, default=300)
    p_train.add_argument("--seq-len", type=int, default=128)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--embed-dim", type=int, default=128)
    p_train.add_argument("--heads", type=int, default=4)
    p_train.add_argument("--layers", type=int, default=2)
    p_train.add_argument("--memory-budget-gb", type=float, default=18.0)
    p_train.add_argument("--receipts", default="results/train_receipts.jsonl")
    p_train.set_defaults(func=cmd_train)

    # bench
    p_bench = sub.add_parser("bench", help="Run benchmark experiment")
    p_bench.add_argument("--compare", default="baseline,ramfold")
    p_bench.add_argument("--tier", default="large")
    p_bench.add_argument("--steps", type=int, default=200)
    p_bench.add_argument("--output", default=None)
    p_bench.set_defaults(func=cmd_bench)

    args = ap.parse_args()
    if not args.command:
        ap.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
