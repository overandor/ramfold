"""
RAMFold Policy Schema — all memory-saving mechanisms as one trainable policy.

μ = {
  sequence length,
  batch size,
  gradient accumulation,
  activation checkpointing level,
  adapter rank,
  optimizer precision,
  embedding precision,
  KV-cache precision,
  KV eviction policy,
  retrieval top-k,
  context compression ratio,
  tool-call budget,
  cloud escalation threshold
}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MemoryPolicy:
    """A single memory policy configuration (one arm in the bandit)."""

    # Training knobs
    seq_len: int = 256
    batch_size: int = 8
    grad_accum: int = 1
    checkpoint_level: int = 0  # 0=off, 1=partial, 2=full

    # Adapter knobs
    adapter_rank: int = 0  # 0=full fine-tune, >0=LoRA rank

    # Precision knobs
    optimizer_precision: str = "fp32"  # fp32, fp16, int8
    embedding_bits: int = 32  # 32, 16, 8, 4
    kv_bits: int = 32  # 32, 16, 8, 4

    # KV cache knobs
    kv_eviction: str = "none"  # none, recency, semantic, budgeted
    kv_budget_tokens: int = 4096

    # Retrieval knobs
    retrieval_top_k: int = 0  # 0=no retrieval
    context_compression_ratio: float = 1.0  # 1.0=full, 0.5=half, 0.25=quarter

    # Agent knobs
    tool_call_budget: int = 10
    cloud_escalation_threshold: float = 0.95  # pressure threshold for cloud fallback

    # Metadata
    name: str = "default"
    pulls: int = 0
    total_reward: float = 0.0
    total_quality: float = 0.0
    total_memory: float = 0.0
    total_swap: float = 0.0
    total_latency: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.pulls, 1)

    @property
    def avg_quality(self) -> float:
        return self.total_quality / max(self.pulls, 1)

    @property
    def avg_memory(self) -> float:
        return self.total_memory / max(self.pulls, 1)

    @property
    def avg_swap(self) -> float:
        return self.total_swap / max(self.pulls, 1)

    @property
    def avg_latency(self) -> float:
        return self.total_latency / max(self.pulls, 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def memory_footprint_estimate_mb(self, dim: int, layers: int, ff_mult: int,
                                     vocab_size: int) -> float:
        """Estimate hot memory footprint in MB for this policy."""
        seq = self.seq_len
        bs = self.batch_size

        # Parameter memory (scaled by precision)
        param_bytes = {
            "fp32": 4, "fp16": 2, "int8": 1
        }.get(self.optimizer_precision, 4)

        token_params = vocab_size * dim
        pos_params = seq * dim
        block_params = (4 * dim * dim) + (2 * dim * ff_mult * dim) + (4 * dim)
        total_params = token_params + pos_params + layers * block_params + dim * vocab_size

        if self.adapter_rank > 0:
            trainable_params = self.adapter_rank * (4 * dim * dim + 2 * dim * ff_mult * dim) * layers
            param_mb = (total_params * 2 + trainable_params * param_bytes) / 1024**2
        else:
            param_mb = total_params * param_bytes / 1024**2

        # KV cache (scaled by kv_bits)
        kv_bytes_per_element = self.kv_bits / 8
        kv_mb = 2 * layers * bs * seq * (dim // 4) * (dim // 4) * kv_bytes_per_element / 1024**2 if dim >= 4 else 0

        # Activations (reduced by checkpointing)
        act_multiplier = {0: 6, 1: 3, 2: 1}[self.checkpoint_level]
        act_mb = bs * seq * dim * layers * act_multiplier * 4 / 1024**2

        # Embedding memory (scaled by embedding_bits)
        emb_bytes = self.embedding_bits / 8
        emb_mb = vocab_size * dim * emb_bytes / 1024**2

        # Context compression reduces effective context memory
        context_mb = bs * seq * dim * 2 * 4 * self.context_compression_ratio / 1024**2

        return round(param_mb + kv_mb + act_mb + emb_mb + context_mb, 2)

    def shrink(self, factor: float = 0.8) -> "MemoryPolicy":
        """Create a compressed version of this policy."""
        return MemoryPolicy(
            seq_len=max(32, int(self.seq_len * factor)),
            batch_size=max(1, self.batch_size // 2),
            grad_accum=min(16, self.grad_accum * 2),
            checkpoint_level=min(2, self.checkpoint_level + 1),
            adapter_rank=self.adapter_rank,
            optimizer_precision=self.optimizer_precision,
            embedding_bits=max(4, min(self.embedding_bits, 8)),
            kv_bits=max(4, min(self.kv_bits, 8)),
            kv_eviction="recency" if self.kv_eviction == "none" else self.kv_eviction,
            kv_budget_tokens=max(512, int(self.kv_budget_tokens * factor)),
            retrieval_top_k=self.retrieval_top_k,
            context_compression_ratio=max(0.25, self.context_compression_ratio * factor),
            tool_call_budget=self.tool_call_budget,
            cloud_escalation_threshold=self.cloud_escalation_threshold,
            name=f"{self.name}_shrink_{factor:.2f}",
        )

    def relax(self, factor: float = 1.15) -> "MemoryPolicy":
        """Create a relaxed version of this policy."""
        return MemoryPolicy(
            seq_len=min(2048, int(self.seq_len * factor)),
            batch_size=min(32, self.batch_size + 1),
            grad_accum=max(1, self.grad_accum // 2),
            checkpoint_level=max(0, self.checkpoint_level - 1),
            adapter_rank=self.adapter_rank,
            optimizer_precision=self.optimizer_precision,
            embedding_bits=min(32, self.embedding_bits * 2),
            kv_bits=min(32, self.kv_bits * 2),
            kv_eviction="none" if self.kv_eviction == "recency" else self.kv_eviction,
            kv_budget_tokens=min(8192, int(self.kv_budget_tokens * factor)),
            retrieval_top_k=self.retrieval_top_k,
            context_compression_ratio=min(1.0, self.context_compression_ratio * factor),
            tool_call_budget=self.tool_call_budget,
            cloud_escalation_threshold=self.cloud_escalation_threshold,
            name=f"{self.name}_relax_{factor:.2f}",
        )


def generate_policy_grid() -> list[MemoryPolicy]:
    """Generate a grid of policy arms for the bandit to explore."""
    policies = []
    seq_lens = [64, 128, 256, 512]
    batch_sizes = [1, 2, 4, 8, 16]
    checkpoint_levels = [0, 1, 2]
    kv_bits_options = [32, 16, 8]
    compression_ratios = [1.0, 0.5, 0.25]

    for seq in seq_lens:
        for bs in batch_sizes:
            for ckpt in checkpoint_levels:
                for kv_b in kv_bits_options:
                    for cr in compression_ratios:
                        ga = max(1, 16 // bs)
                        name = f"s{seq}_b{bs}_c{ckpt}_kv{kv_b}_cr{int(cr*100)}"
                        policies.append(MemoryPolicy(
                            seq_len=seq,
                            batch_size=bs,
                            grad_accum=ga,
                            checkpoint_level=ckpt,
                            kv_bits=kv_b,
                            context_compression_ratio=cr,
                            name=name,
                        ))
    return policies
