"""
RAMFold Receipt Schema + Ledger — append-only JSONL receipts.

Every run logs policy, memory, loss, speed, sample quality, and verdict
the same way.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


RECEIPT_SCHEMA_VERSION = "1.0.0"


@dataclass
class Receipt:
    """Standard receipt for every RAMFold event."""
    schema_version: str = RECEIPT_SCHEMA_VERSION
    timestamp: float = field(default_factory=time.time)
    event: str = ""
    run_id: str = ""
    tier: str = ""
    mode: str = ""  # baseline, v0, v1, v2, adaptive
    step: int = 0
    loss: float = 0.0
    tokens_per_sec: float = 0.0
    pressure: float = 0.0
    swap_used_gb: float = 0.0
    swap_delta: float = 0.0
    pressure_delta: float = 0.0
    policy: dict = field(default_factory=dict)
    memory_snapshot: dict = field(default_factory=dict)
    quality_score: float = 0.0
    verification_score: float = 0.0
    qpg: float = 0.0
    action: str = ""
    crashed: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class ReceiptLedger:
    """Append-only JSONL ledger for RAMFold receipts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def write(self, receipt: Receipt):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(receipt.to_json() + "\n")
        self._count += 1

    def write_dict(self, data: dict):
        receipt = Receipt(
            event=data.get("event", ""),
            run_id=data.get("run_id", ""),
            tier=data.get("tier", ""),
            mode=data.get("mode", ""),
            step=data.get("step", 0),
            loss=data.get("loss", 0.0),
            tokens_per_sec=data.get("tokens_per_sec", 0.0),
            pressure=data.get("pressure", 0.0),
            swap_used_gb=data.get("swap_used_gb", 0.0),
            swap_delta=data.get("swap_delta", 0.0),
            pressure_delta=data.get("pressure_delta", 0.0),
            policy=data.get("policy", {}),
            memory_snapshot=data.get("memory_snapshot", {}),
            quality_score=data.get("quality_score", 0.0),
            verification_score=data.get("verification_score", 0.0),
            qpg=data.get("qpg", 0.0),
            action=data.get("action", ""),
            crashed=data.get("crashed", False),
            notes=data.get("notes", ""),
        )
        self.write(receipt)

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text().strip().splitlines()
            if line.strip()
        ]

    @property
    def count(self) -> int:
        return self._count


SCHEMA_JSON = {
    "schema_version": RECEIPT_SCHEMA_VERSION,
    "fields": {
        "event": "string — event type (step, trial_start, trial_end, benchmark)",
        "run_id": "string — unique run identifier",
        "tier": "string — model tier (tiny, medium, large, stress)",
        "mode": "string — policy mode (baseline, v0, v1, v2, adaptive)",
        "step": "int — training step number",
        "loss": "float — training loss",
        "tokens_per_sec": "float — throughput",
        "pressure": "float — memory pressure 0-1",
        "swap_used_gb": "float — swap usage in GB",
        "swap_delta": "float — swap change from baseline",
        "pressure_delta": "float — pressure change from baseline",
        "policy": "object — full memory policy",
        "memory_snapshot": "object — full memory observer snapshot",
        "quality_score": "float — normalized quality 0-1",
        "verification_score": "float — verification 0-1",
        "qpg": "float — quality per GB",
        "action": "string — governor action taken",
        "crashed": "bool — whether trial crashed",
        "notes": "string — freeform notes",
    },
}
