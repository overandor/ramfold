#!/usr/bin/env python3
"""
RAMFold KV Daemon — HTTP service for live KV cache management.

Exposes:
  GET  /memory/snapshot    — live macOS unified memory state
  GET  /policy/best        — best policy from bandit history
  POST /kv/compress        — compress KV cache (reduce budget, quantize, evict)
  POST /kv/relax           — relax KV cache (increase budget, dequantize)
  POST /agent/notify       — notify about agent task dispatch/completion
  GET  /health             — health check

The daemon runs continuously, monitoring memory and adjusting
Ollama's KV cache budget in real-time. The C scheduler calls
this via HTTP socket (same pattern as ws_ollama_ask).

If this daemon is down, the C scheduler falls back to:
  python3 -m ramfold.controller --action compress

Usage:
  python3 ramfold_daemon.py --port 8801
  python3 ramfold_daemon.py --port 8801 --budget-gb 18.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import threading
import urllib.request
from dataclasses import asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ramfold.observer import MemoryObserver, snapshot_to_dict
from ramfold.controller import BanditController, SwapWeightedGovernor, MemoryPolicy, generate_policy_grid
from ramfold.compression import KVBudgeter, SemanticEviction, ContextCompressor, EmbeddingQuantizer
from ramfold.receipts import ReceiptLedger

# ─── Global state ───
observer: MemoryObserver = None
governor: SwapWeightedGovernor = None
bandit: BanditController = None
current_policy: MemoryPolicy = None
kv_budgeter: KVBudgeter = None
semantic_evictor: SemanticEviction = None
context_comp: ContextCompressor = None
embed_quantizer: EmbeddingQuantizer = None
ledger: ReceiptLedger = None
ollama_host = "127.0.0.1"
ollama_port = 11434
prev_receipt_hash = ""

# ─── Receipt ───
def write_receipt(action: str, status: str, detail: str, extra: dict = None):
    global prev_receipt_hash
    entry = {
        "timestamp": time.time(),
        "action": action,
        "status": status,
        "detail": detail,
        "prev_hash": prev_receipt_hash,
    }
    if extra:
        entry.update(extra)
    chain_input = json.dumps(entry, sort_keys=True)
    h = hashlib.sha256(chain_input.encode()).hexdigest()
    entry["hash"] = h
    prev_receipt_hash = h
    if ledger:
        ledger.write_dict(entry)
    return entry

# ─── Ollama KV management ───
def ollama_unload_model(model: str = "llama3.2"):
    """Unload model from Ollama to free KV cache memory."""
    try:
        req = urllib.request.Request(
            f"http://{ollama_host}:{ollama_port}/api/generate",
            data=json.dumps({
                "model": model,
                "keep_alive": 0,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10).read()
        return True
    except Exception as e:
        print(f"  [ramfold] ollama unload failed: {e}")
        return False

def ollama_set_keep_alive(model: str = "llama3.2", seconds: int = 300):
    """Set model keep-alive to control KV cache retention."""
    try:
        req = urllib.request.Request(
            f"http://{ollama_host}:{ollama_port}/api/generate",
            data=json.dumps({
                "model": model,
                "keep_alive": seconds,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10).read()
        return True
    except Exception as e:
        print(f"  [ramfold] ollama keep_alive failed: {e}")
        return False

# ─── Compression actions ───
def do_compress(aggressiveness: float = 0.8) -> dict:
    """Compress KV cache and context based on current memory pressure."""
    global current_policy, kv_budgeter, semantic_evictor, context_comp, embed_quantizer

    snap = observer.snapshot()
    pressure = snap.pressure
    swap_delta = observer.swap_delta(snap)
    swap_growing = observer.swap_growing()

    # Use governor to decide compression level
    action_name, new_policy = governor.decide(
        policy=current_policy,
        model_mem_mb=snap.mlx_active_mb,
        pressure=pressure,
        pressure_delta=observer.pressure_delta(snap),
        swap_delta=swap_delta,
        swap_growing=swap_growing,
        loss_improving=True,  # assume improving unless told otherwise
        tps_collapsing=False,
        at_floor=(current_policy.kv_bits <= 4 and current_policy.seq_len <= 64),
        swap_absolute=snap.swap_used_gb,
    )

    old_kv_bits = current_policy.kv_bits
    old_budget = current_policy.kv_budget_tokens
    current_policy = new_policy

    # Apply KV budget reduction
    kv_budgeter.budget_tokens = new_policy.kv_budget_tokens
    kv_budgeter.kv_bits = new_policy.kv_bits
    if new_policy.kv_eviction != "none":
        kv_budgeter.eviction = new_policy.kv_eviction
        evicted = kv_budgeter.evict()
    else:
        evicted = []

    # Apply context compression
    context_comp.compression_ratio = new_policy.context_compression_ratio
    context_comp.max_tokens = new_policy.kv_budget_tokens

    # Apply embedding quantization
    embed_quantizer.bits = new_policy.embedding_bits

    # If pressure is critical, unload Ollama model to free KV cache
    ollama_action = "none"
    if pressure > 0.92 and swap_delta > 0.5:
        ollama_unload_model()
        ollama_action = "unloaded"
    elif pressure > 0.85:
        ollama_set_keep_alive(seconds=60)  # short keep-alive
        ollama_action = "short_keepalive"
    else:
        ollama_set_keep_alive(seconds=300)  # normal
        ollama_action = "normal_keepalive"

    result = {
        "action": action_name,
        "pressure": round(pressure, 4),
        "swap_delta": round(swap_delta, 4),
        "swap_growing": swap_growing,
        "old_kv_bits": old_kv_bits,
        "new_kv_bits": new_policy.kv_bits,
        "old_budget": old_budget,
        "new_budget": new_policy.kv_budget_tokens,
        "evicted_tokens": len(evicted),
        "ollama_action": ollama_action,
        "compression_ratio": new_policy.context_compression_ratio,
    }

    write_receipt("compress", "success", action_name, result)
    return result

def do_relax(aggressiveness: float = 1.15) -> dict:
    """Relax KV cache budget after agent task completes."""
    global current_policy, kv_budgeter, context_comp, embed_quantizer

    snap = observer.snapshot()
    pressure = snap.pressure
    swap_delta = observer.swap_delta(snap)

    # Only relax if pressure is dropping and swap is stable
    if pressure < 0.70 and swap_delta < 0.05:
        new_policy = current_policy.relax(aggressiveness)
        old_kv_bits = current_policy.kv_bits
        old_budget = current_policy.kv_budget_tokens
        current_policy = new_policy

        kv_budgeter.budget_tokens = new_policy.kv_budget_tokens
        kv_budgeter.kv_bits = new_policy.kv_bits
        context_comp.compression_ratio = new_policy.context_compression_ratio
        context_comp.max_tokens = new_policy.kv_budget_tokens
        embed_quantizer.bits = new_policy.embedding_bits

        ollama_set_keep_alive(seconds=600)  # long keep-alive when relaxed

        result = {
            "action": "relax",
            "pressure": round(pressure, 4),
            "old_kv_bits": old_kv_bits,
            "new_kv_bits": new_policy.kv_bits,
            "old_budget": old_budget,
            "new_budget": new_policy.kv_budget_tokens,
            "ollama_action": "long_keepalive",
        }
        write_receipt("relax", "success", "pressure_low_relaxing", result)
        return result
    else:
        result = {
            "action": "hold",
            "pressure": round(pressure, 4),
            "swap_delta": round(swap_delta, 4),
            "reason": "pressure_still_high",
        }
        write_receipt("relax", "hold", "pressure_still_high", result)
        return result

def do_snapshot() -> dict:
    """Get current memory snapshot + policy state."""
    snap = observer.snapshot()
    return {
        "memory": snapshot_to_dict(snap),
        "policy": current_policy.to_dict(),
        "kv": {
            "budget_tokens": kv_budgeter.budget_tokens,
            "kv_bits": kv_budgeter.kv_bits,
            "eviction": kv_budgeter.eviction,
            "entries": len(kv_budgeter._entries),
        },
        "compression": {
            "ratio": context_comp.compression_ratio,
            "max_tokens": context_comp.max_tokens,
        },
        "embedding_bits": embed_quantizer.bits,
        "swap_delta": round(observer.swap_delta(snap), 4),
        "pressure_delta": round(observer.pressure_delta(snap), 4),
        "swap_growing": observer.swap_growing(),
        "pressure_trend": round(observer.pressure_trend(), 4),
    }

# ─── Background monitor thread ───
monitor_running = True
monitor_history = []

def monitor_loop(interval_s: float = 5.0):
    """Background thread that monitors memory and auto-compresses if needed."""
    global monitor_running, current_policy
    while monitor_running:
        try:
            snap = observer.snapshot()
            pressure = snap.pressure
            swap_delta = observer.swap_delta(snap)
            swap_growing = observer.swap_growing()

            monitor_history.append({
                "ts": time.time(),
                "pressure": pressure,
                "swap_gb": snap.swap_used_gb,
                "mlx_mb": snap.mlx_active_mb,
            })
            # Keep last 360 samples (30 min at 5s interval)
            if len(monitor_history) > 360:
                monitor_history.pop(0)

            # Auto-compress if pressure is high and swap is growing
            if pressure > 0.85 and (swap_growing or swap_delta > 0.3):
                print(f"  [ramfold] auto-compress: pressure={pressure:.3f} swap_delta={swap_delta:.3f}")
                do_compress()

        except Exception as e:
            print(f"  [ramfold] monitor error: {e}")
        time.sleep(interval_s)

# ─── HTTP handler ───
class RAMFoldHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default logging

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok", "service": "ramfold", "uptime_s": time.time() - start_time})
        elif self.path == "/memory/snapshot":
            self.send_json(do_snapshot())
        elif self.path == "/policy/best":
            best = bandit.best_arm()
            self.send_json({
                "best_arm": best.to_dict() if best else None,
                "summary": bandit.summary(),
            })
        elif self.path == "/monitor/history":
            self.send_json({"history": monitor_history[-60:]})
        else:
            self.send_error(404)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            data = json.loads(body)
        except:
            data = {}

        if self.path == "/kv/compress":
            aggressiveness = data.get("aggressiveness", 0.8)
            result = do_compress(aggressiveness)
            self.send_json(result)
        elif self.path == "/kv/relax":
            aggressiveness = data.get("aggressiveness", 1.15)
            result = do_relax(aggressiveness)
            self.send_json(result)
        elif self.path == "/agent/notify":
            event = data.get("event", "unknown")
            task_desc = data.get("description", "")
            task_lane = data.get("lane", "")
            if event == "dispatch":
                # Agent is dispatching a task — compress if it's heavy
                heavy_keywords = ["build", "train", "test", "notarize", "compile", "swift"]
                is_heavy = any(kw in task_desc.lower() for kw in heavy_keywords)
                if is_heavy:
                    print(f"  [ramfold] agent dispatching heavy task: {task_desc}")
                    result = do_compress(0.85)
                    self.send_json({"notified": True, "compressed": True, "result": result})
                else:
                    self.send_json({"notified": True, "compressed": False})
            elif event == "complete":
                # Agent task completed — try to relax
                result = do_relax()
                self.send_json({"notified": True, "relaxed": result["action"] == "relax", "result": result})
            elif event == "handback":
                # User returned — relax everything
                result = do_relax(1.2)
                self.send_json({"notified": True, "result": result})
            else:
                self.send_json({"notified": True, "event": event})
        else:
            self.send_error(404)

    def send_json(self, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

# ─── Main ───
start_time = time.time()

def main():
    global observer, governor, bandit, current_policy, kv_budgeter, semantic_evictor
    global context_comp, embed_quantizer, ledger, ollama_host, ollama_port

    ap = argparse.ArgumentParser(description="RAMFold KV Daemon")
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--budget-gb", type=float, default=18.0)
    ap.add_argument("--ollama-host", default="127.0.0.1")
    ap.add_argument("--ollama-port", type=int, default=11434)
    ap.add_argument("--receipts", default="receipts/ramfold_chain.jsonl")
    args = ap.parse_args()

    ollama_host = args.ollama_host
    ollama_port = args.ollama_port

    # Initialize components
    observer = MemoryObserver(budget_gb=args.budget_gb)
    governor = SwapWeightedGovernor(budget_gb=args.budget_gb)
    bandit = BanditController(arms=generate_policy_grid())
    current_policy = MemoryPolicy(
        seq_len=512, batch_size=4, kv_bits=32, kv_eviction="semantic",
        kv_budget_tokens=4096, context_compression_ratio=1.0,
        name="daemon_default",
    )
    kv_budgeter = KVBudgeter(budget_tokens=4096, kv_bits=32, eviction="semantic")
    semantic_evictor = SemanticEviction(budget=4096)
    context_comp = ContextCompressor(max_tokens=4096, compression_ratio=1.0)
    embed_quantizer = EmbeddingQuantizer(bits=32)

    Path(args.receipts).parent.mkdir(parents=True, exist_ok=True)
    ledger = ReceiptLedger(args.receipts)

    # Take baseline snapshot
    baseline = observer.snapshot()
    print(f"  [ramfold] baseline: pressure={baseline.pressure:.3f} swap={baseline.swap_used_gb:.2f}GB")
    print(f"  [ramfold] budget={args.budget_gb}GB ollama={ollama_host}:{ollama_port}")
    write_receipt("init", "success", f"ramfold daemon started, budget={args.budget_gb}GB")

    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_loop, args=(5.0,), daemon=True)
    monitor_thread.start()
    print(f"  [ramfold] monitor thread started (5s interval)")

    # Start HTTP server
    server = HTTPServer(("127.0.0.1", args.port), RAMFoldHandler)
    print(f"  [ramfold] HTTP daemon listening on 127.0.0.1:{args.port}")
    print(f"  [ramfold] endpoints: /health /memory/snapshot /kv/compress /kv/relax /agent/notify /policy/best")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  [ramfold] shutting down")
        monitor_running = False
        write_receipt("shutdown", "success", "daemon stopped")
        server.shutdown()

if __name__ == "__main__":
    main()
