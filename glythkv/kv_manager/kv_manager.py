"""
GlythKV KV Manager — parser-controlled KV cache eviction, folding, and cooling.

The KV cache does not grow forever like some cursed attic full of junk.
If the current grammar can no longer use a KV block, it gets evicted,
folded, or cooled.

This is the inference-time analog of RAMFold's training-time memory policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .glyth_parser import KVAction, GlythParser


@dataclass
class KVBlock:
    """A single KV cache block with metadata."""
    block_id: int
    token_range: tuple[int, int]  # (start, end) position in sequence
    size_mb: float
    precision_bits: int = 32  # 32, 16, 8, 4
    age: int = 0               # steps since last access
    access_count: int = 0
    last_access_step: int = 0
    block_type: str = "general"  # general, code, evidence, log, constraint, system
    alive: bool = True
    folded_into: Optional[int] = None  # block_id this was folded into
    cooled: bool = False


class KVManager:
    """
    Parser-controlled KV cache manager.

    The parser decides which KV blocks live and die.
    This manager executes those decisions.
    """

    def __init__(
        self,
        budget_blocks: int = 4096,
        budget_mb: float = 2048.0,
        min_precision_bits: int = 4,
    ):
        self.budget_blocks = budget_blocks
        self.budget_mb = budget_mb
        self.min_precision_bits = min_precision_bits

        self.blocks: dict[int, KVBlock] = {}
        self.next_block_id = 0
        self.total_evicted = 0
        self.total_folded = 0
        self.total_cooled = 0
        self.peak_blocks = 0
        self.peak_mb = 0.0

    def add_block(
        self,
        token_range: tuple[int, int],
        size_mb: float,
        block_type: str = "general",
    ) -> KVBlock:
        """Add a new KV block to the cache."""
        block_id = self.next_block_id
        self.next_block_id += 1

        block = KVBlock(
            block_id=block_id,
            token_range=token_range,
            size_mb=size_mb,
            block_type=block_type,
            last_access_step=0,
        )
        self.blocks[block_id] = block
        self.peak_blocks = max(self.peak_blocks, len(self.blocks))
        self.peak_mb = max(self.peak_mb, self.current_mb)
        return block

    @property
    def current_mb(self) -> float:
        """Total MB of alive blocks."""
        return sum(b.size_mb for b in self.blocks.values() if b.alive)

    @property
    def alive_count(self) -> int:
        """Number of alive blocks."""
        return sum(1 for b in self.blocks.values() if b.alive)

    def age_blocks(self):
        """Age all alive blocks by one step."""
        for block in self.blocks.values():
            if block.alive:
                block.age += 1

    def access_block(self, block_id: int, step: int):
        """Mark a block as accessed at this step."""
        if block_id in self.blocks:
            block = self.blocks[block_id]
            block.access_count += 1
            block.last_access_step = step
            block.age = 0

    def execute_actions(self, actions: dict[int, KVAction], step: int) -> dict:
        """
        Execute parser-decided KV actions.

        Returns a summary of what happened for receipts.
        """
        evicted = 0
        folded = 0
        cooled = 0
        kept = 0
        mb_freed = 0.0

        for block_id, action in actions.items():
            if block_id not in self.blocks:
                continue
            block = self.blocks[block_id]

            if action == KVAction.EVICT and block.alive:
                block.alive = False
                mb_freed += block.size_mb
                evicted += 1
                self.total_evicted += 1

            elif action == KVAction.FOLD and block.alive:
                # Fold: reduce precision to save space
                old_size = block.size_mb
                if block.precision_bits > self.min_precision_bits:
                    block.precision_bits = max(self.min_precision_bits, block.precision_bits // 2)
                    block.size_mb = old_size * (block.precision_bits / (block.precision_bits * 2))
                    mb_freed += old_size - block.size_mb
                block.cooled = True
                folded += 1
                self.total_folded += 1

            elif action == KVAction.COOL and block.alive:
                # Cool: reduce precision by one level
                old_size = block.size_mb
                if block.precision_bits > self.min_precision_bits:
                    block.precision_bits = max(self.min_precision_bits, block.precision_bits // 2)
                    block.size_mb = old_size * 0.5
                    mb_freed += old_size - block.size_mb
                block.cooled = True
                cooled += 1
                self.total_cooled += 1

            elif action == KVAction.KEEP:
                kept += 1

        return {
            "step": step,
            "evicted": evicted,
            "folded": folded,
            "cooled": cooled,
            "kept": kept,
            "mb_freed": round(mb_freed, 2),
            "alive_blocks": self.alive_count,
            "current_mb": round(self.current_mb, 2),
        }

    def summary(self) -> dict:
        """Summary for receipts."""
        return {
            "total_blocks_created": self.next_block_id,
            "alive_blocks": self.alive_count,
            "peak_blocks": self.peak_blocks,
            "current_mb": round(self.current_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "total_evicted": self.total_evicted,
            "total_folded": self.total_folded,
            "total_cooled": self.total_cooled,
            "budget_mb": self.budget_mb,
            "budget_utilization": round(self.current_mb / max(self.budget_mb, 1), 4),
        }
