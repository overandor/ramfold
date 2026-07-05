"""
RAMFold Compression Plane — context, KV cache, and embedding compression.

Each memory type has its own compression policy, but the controller
sees them as one budget.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContextShard:
    id: str
    text: str
    priority: float
    hot: bool = False
    embedding: Optional[list[float]] = None

    def compressed_text(self, ratio: float) -> str:
        """Compress shard text by keeping first/last portions."""
        if ratio >= 1.0:
            return self.text
        keep = int(len(self.text) * ratio)
        if keep >= len(self.text):
            return self.text
        head = keep // 2
        tail = keep - head
        return self.text[:head] + "...[compressed]..." + self.text[-tail:]


class ContextCompressor:
    """Compress context shards by priority and budget."""

    def __init__(self, max_tokens: int = 4096, compression_ratio: float = 1.0):
        self.max_tokens = max_tokens
        self.compression_ratio = compression_ratio

    def compress(self, shards: list[ContextShard], top_k: int = 5) -> dict:
        """Compress context shards into kept/summarized/dropped."""
        ranked = sorted(shards, key=lambda s: (s.hot, s.priority), reverse=True)

        kept = []
        summarized = []
        dropped = []
        total_tokens = 0

        for i, shard in enumerate(ranked):
            compressed = shard.compressed_text(self.compression_ratio)
            token_est = len(compressed) // 4

            if i < top_k and total_tokens + token_est <= self.max_tokens:
                kept.append({
                    "id": shard.id,
                    "text": compressed,
                    "priority": shard.priority,
                    "tokens": token_est,
                })
                total_tokens += token_est
            elif i < top_k + 4:
                summary = shard.text[:120]
                summarized.append({
                    "id": shard.id,
                    "summary": summary,
                    "tokens": len(summary) // 4,
                })
            else:
                dropped.append(shard.id)

        return {
            "kept": kept,
            "summarized": summarized,
            "dropped": dropped,
            "total_tokens": total_tokens,
            "compression_ratio": self.compression_ratio,
        }


class KVBudgeter:
    """Manage KV cache memory with eviction policies."""

    def __init__(
        self,
        budget_tokens: int = 4096,
        kv_bits: int = 32,
        eviction: str = "none",
    ):
        self.budget_tokens = budget_tokens
        self.kv_bits = kv_bits
        self.eviction = eviction
        self._entries: list[dict] = []

    def estimate_kv_mb(self, layers: int, heads: int, head_dim: int) -> float:
        """Estimate KV cache memory in MB."""
        bytes_per_element = self.kv_bits / 8
        total = 2 * layers * self.budget_tokens * heads * head_dim * bytes_per_element
        return round(total / 1024**2, 2)

    def add_entry(self, token_id: int, importance: float = 0.0):
        self._entries.append({"token_id": token_id, "importance": importance})

    def evict(self) -> list[int]:
        """Evict tokens based on policy. Returns evicted token IDs."""
        if self.eviction == "none":
            return []
        elif self.eviction == "recency":
            if len(self._entries) <= self.budget_tokens:
                return []
            evict_count = len(self._entries) - self.budget_tokens
            evicted = [e["token_id"] for e in self._entries[:evict_count]]
            self._entries = self._entries[evict_count:]
            return evicted
        elif self.eviction == "semantic":
            self._entries.sort(key=lambda e: e["importance"], reverse=True)
            if len(self._entries) <= self.budget_tokens:
                return []
            evicted = [e["token_id"] for e in self._entries[self.budget_tokens:]]
            self._entries = self._entries[:self.budget_tokens]
            return evicted
        return []


class EmbeddingQuantizer:
    """Quantize embeddings to lower precision."""

    def __init__(self, bits: int = 32):
        self.bits = bits
        self.bytes_per_element = bits / 8

    def quantize(self, embedding: list[float]) -> list[float]:
        """Quantize a float embedding to target precision."""
        if self.bits >= 32:
            return embedding
        elif self.bits == 16:
            return [round(x, 3) for x in embedding]
        elif self.bits == 8:
            scale = max(abs(min(embedding)), abs(max(embedding)))
            return [round(x / scale * 127) / 127 * scale if scale > 0 else 0.0 for x in embedding]
        elif self.bits == 4:
            scale = max(abs(min(embedding)), abs(max(embedding)))
            return [round(x / scale * 7) / 7 * scale if scale > 0 else 0.0 for x in embedding]
        return embedding

    def memory_mb(self, dim: int, count: int) -> float:
        """Estimate memory for count embeddings of dim dimensions."""
        return round(count * dim * self.bytes_per_element / 1024**2, 2)

    def compression_ratio(self) -> float:
        """Compression ratio vs fp32."""
        return 32.0 / self.bits


class SemanticEviction:
    """Importance-ranked KV/context retention."""

    def __init__(self, budget: int):
        self.budget = budget
        self._store: dict[str, dict] = {}

    def add(self, key: str, value: str, importance: float):
        h = hashlib.sha1(key.encode()).hexdigest()
        self._store[h] = {
            "key": key,
            "value": value,
            "importance": importance,
            "hash": h,
        }

    def retain(self) -> list[dict]:
        """Return top-k by importance."""
        ranked = sorted(self._store.values(), key=lambda x: x["importance"], reverse=True)
        return ranked[: self.budget]

    def evict(self) -> list[str]:
        """Evict low-importance entries. Returns evicted hashes."""
        ranked = sorted(self._store.values(), key=lambda x: x["importance"], reverse=True)
        kept = ranked[: self.budget]
        evicted = [r["hash"] for r in ranked[self.budget:]]
        for h in evicted:
            del self._store[h]
        return evicted
