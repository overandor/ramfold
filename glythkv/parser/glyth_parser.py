"""
GlythKV Parser — The parser becomes the memory manager.

At every generation step, the parser:
1. Determines valid next token shapes (syntax masking)
2. Decides which memory shards to fetch (context control)
3. Decides which KV blocks are alive vs dead (KV eviction)
4. Routes tensor work to Metal or CPU (operator placement)
5. Tracks proof obligations (verification receipts)

The parser is NOT parsing output after the fact.
It is fetching memory DURING generation.
It controls what the model can say, what context exists,
what KV survives, and what Metal executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class GrammarState(Enum):
    """Adaptive grammar states — the parser switches between these during generation."""
    INIT = auto()
    ANSWERING = auto()
    EVIDENCE_NEEDED = auto()
    CODE_REPAIR = auto()
    COMPRESSED_ANSWER = auto()  # triggered by RAM pressure
    VERIFYING = auto()
    DONE = auto()


class KVAction(Enum):
    """What happens to a KV block at this generation step."""
    KEEP = "keep"          # still useful, pin in memory
    EVICT = "evict"        # dead, remove from cache
    FOLD = "fold"          # merge with another block to save space
    COOL = "cool"          # move to lower precision / disk
    REFETCH = "refetch"    # was evicted, needed again


class FetchAction(Enum):
    """What context shards to fetch at this generation step."""
    NONE = "none"          # current context is sufficient
    SOURCE = "source"      # fetch source code shards
    LOGS = "logs"          # fetch log shards
    RECEIPTS = "receipts"  # fetch verification receipts
    CONSTRAINTS = "constraints"  # fetch system constraints
    COLD = "cold"          # fetch from cold storage (expensive)


class MetalOp(Enum):
    """Hot-path operators that should run on Metal."""
    ATTENTION = "attention"
    KV_COMPRESS = "kv_compress"
    SYNTAX_MASK = "syntax_mask"
    KV_FOLD = "kv_fold"
    EMBEDDING_LOOKUP = "embedding_lookup"


@dataclass
class ParserState:
    """The state of the parser at a single generation step."""
    step: int = 0
    grammar: GrammarState = GrammarState.INIT
    kv_actions: dict[int, KVAction] = field(default_factory=dict)  # block_id -> action
    fetch_actions: list[FetchAction] = field(default_factory=list)
    metal_ops: list[MetalOp] = field(default_factory=list)
    syntax_mask: Optional[list[bool]] = None  # True = allowed, False = masked
    proof_obligation: Optional[str] = None
    ram_pressure: float = 0.0
    kv_blocks_alive: int = 0
    kv_blocks_total: int = 0
    fetch_shards_used: int = 0
    fetch_shards_total: int = 0

    @property
    def kv_efficiency(self) -> float:
        """Fraction of KV blocks that are alive (useful)."""
        if self.kv_blocks_total == 0:
            return 1.0
        return self.kv_blocks_alive / self.kv_blocks_total

    @property
    def fetch_precision(self) -> float:
        """Fraction of fetched shards that were actually used."""
        if self.fetch_shards_total == 0:
            return 1.0
        return self.fetch_shards_used / self.fetch_shards_total

    @property
    def syntax_mask_ratio(self) -> float:
        """Fraction of vocabulary masked out by grammar."""
        if self.syntax_mask is None:
            return 0.0
        return 1.0 - (sum(self.syntax_mask) / len(self.syntax_mask))


@dataclass
class GlythDirective:
    """A single .glyth directive — one unit of memory policy."""
    target: str          # what this applies to (block id, shard name, op name)
    action: str          # pin, compress, lossy, kv_keep, kv_evict, metal_kernel, etc.
    priority: float = 1.0
    proof_required: bool = False
    metal_kernel: Optional[str] = None
    ram_budget_mb: Optional[float] = None


class GlythParser:
    """
    The adaptive parser that controls memory during LLM generation.

    This is NOT a post-hoc output parser. It runs AT EVERY GENERATION STEP
    and controls:
    - What the model is allowed to say (syntax masking)
    - What context exists (fetch control)
    - What KV survives (KV eviction)
    - What Metal executes (operator placement)
    - What needs proof (verification receipts)
    """

    def __init__(
        self,
        ram_budget_gb: float = 18.0,
        kv_budget_blocks: int = 4096,
        swap_absolute_threshold: float = 2.0,
        pressure_compress_threshold: float = 0.85,
    ):
        self.ram_budget_gb = ram_budget_gb
        self.ram_budget_mb = ram_budget_gb * 1024
        self.kv_budget_blocks = kv_budget_blocks
        self.swap_absolute_threshold = swap_absolute_threshold
        self.pressure_compress_threshold = pressure_compress_threshold

        self.state = ParserState()
        self.directives: list[GlythDirective] = []
        self.grammar_history: list[GrammarState] = []
        self.kv_block_meta: dict[int, dict] = {}  # block_id -> metadata
        self.fetch_history: list[FetchAction] = []

    def update_grammar(
        self,
        ram_pressure: float,
        swap_gb: float,
        has_evidence: bool,
        task_type: str,
        loss_improving: bool = True,
    ) -> GrammarState:
        """
        Adapt the grammar based on runtime conditions.

        The grammar CHANGES during inference:
        - Task becomes code repair → code-patch grammar
        - Model lacks evidence → evidence-repair grammar
        - RAM pressure rises → compressed-answer grammar
        - Proof obligation changes → grammar changes with it
        """
        prev_grammar = self.state.grammar

        # RAM pressure → compressed answer grammar
        if ram_pressure > self.pressure_compress_threshold or swap_gb > self.swap_absolute_threshold:
            self.state.grammar = GrammarState.COMPRESSED_ANSWER
        # No evidence → evidence repair grammar
        elif not has_evidence and self.state.grammar == GrammarState.ANSWERING:
            self.state.grammar = GrammarState.EVIDENCE_NEEDED
        # Code task → code repair grammar
        elif task_type == "code" and self.state.grammar == GrammarState.ANSWERING:
            self.state.grammar = GrammarState.CODE_REPAIR
        # Evidence found → back to answering
        elif self.state.grammar == GrammarState.EVIDENCE_NEEDED and has_evidence:
            self.state.grammar = GrammarState.ANSWERING
        # Default
        elif self.state.grammar == GrammarState.INIT:
            self.state.grammar = GrammarState.ANSWERING

        if prev_grammar != self.state.grammar:
            self.grammar_history.append(self.state.grammar)

        return self.state.grammar

    def decide_kv(
        self,
        block_ids: list[int],
        block_ages: list[int],
        block_access_counts: list[int],
        block_sizes_mb: list[float],
    ) -> dict[int, KVAction]:
        """
        Decide what happens to each KV block at this step.

        The parser controls KV — not the model. Blocks that the current
        grammar can no longer use get evicted, folded, or cooled.
        """
        actions = {}
        total_kv_mb = sum(block_sizes_mb)

        # Under RAM pressure: evict old, low-access blocks
        if self.state.grammar == GrammarState.COMPRESSED_ANSWER:
            for i, bid in enumerate(block_ids):
                age = block_ages[i] if i < len(block_ages) else 0
                accesses = block_access_counts[i] if i < len(block_access_counts) else 0
                size = block_sizes_mb[i] if i < len(block_sizes_mb) else 0

                if accesses == 0 and age > 10:
                    actions[bid] = KVAction.EVICT
                elif age > 20 and accesses < 3:
                    actions[bid] = KVAction.COOL
                elif size > 50 and age > 5:
                    actions[bid] = KVAction.FOLD
                else:
                    actions[bid] = KVAction.KEEP

        # Evidence needed: evict non-evidence blocks, fetch evidence
        elif self.state.grammar == GrammarState.EVIDENCE_NEEDED:
            for i, bid in enumerate(block_ids):
                meta = self.kv_block_meta.get(bid, {})
                if meta.get("type") != "evidence":
                    actions[bid] = KVAction.COOL
                else:
                    actions[bid] = KVAction.KEEP

        # Code repair: keep code blocks, evict logs
        elif self.state.grammar == GrammarState.CODE_REPAIR:
            for i, bid in enumerate(block_ids):
                meta = self.kv_block_meta.get(bid, {})
                if meta.get("type") == "log":
                    actions[bid] = KVAction.EVICT
                elif meta.get("type") == "code":
                    actions[bid] = KVAction.KEEP
                else:
                    actions[bid] = KVAction.COOL

        # Default: keep everything
        else:
            for bid in block_ids:
                actions[bid] = KVAction.KEEP

        # Update state
        alive = sum(1 for a in actions.values() if a in (KVAction.KEEP, KVAction.REFETCH))
        self.state.kv_actions = actions
        self.state.kv_blocks_alive = alive
        self.state.kv_blocks_total = len(block_ids)

        return actions

    def decide_fetch(
        self,
        available_shards: list[str],
        shard_types: list[str],
    ) -> list[FetchAction]:
        """
        Decide what context shards to fetch at this step.

        The parser fetches ONLY what the current answer state needs.
        It does not shove everything into the prompt.
        """
        fetches = []

        if self.state.grammar == GrammarState.EVIDENCE_NEEDED:
            for i, shard_type in enumerate(shard_types):
                if shard_type == "source":
                    fetches.append(FetchAction.SOURCE)
                elif shard_type == "receipt":
                    fetches.append(FetchAction.RECEIPTS)

        elif self.state.grammar == GrammarState.CODE_REPAIR:
            for i, shard_type in enumerate(shard_types):
                if shard_type == "source":
                    fetches.append(FetchAction.SOURCE)
                elif shard_type == "constraint":
                    fetches.append(FetchAction.CONSTRAINTS)

        elif self.state.grammar == GrammarState.COMPRESSED_ANSWER:
            # Under pressure: fetch minimal shards only
            for i, shard_type in enumerate(shard_types):
                if shard_type == "constraint":
                    fetches.append(FetchAction.CONSTRAINTS)
            # Don't fetch logs or cold data under pressure

        elif self.state.grammar == GrammarState.VERIFYING:
            fetches.append(FetchAction.RECEIPTS)

        # Default answering: fetch constraints if available
        elif self.state.grammar == GrammarState.ANSWERING:
            for i, shard_type in enumerate(shard_types):
                if shard_type == "constraint":
                    fetches.append(FetchAction.CONSTRAINTS)

        self.state.fetch_actions = fetches
        self.fetch_history.extend(fetches)
        return fetches

    def decide_metal_ops(self) -> list[MetalOp]:
        """
        Decide which hot-path operators should run on Metal.

        Metal^932 means the hot path gets shoved down into fused, tiled,
        low-level Metal work: attention, masks, KV compression, tensor tiles.
        """
        ops = [MetalOp.ATTENTION]

        if self.state.grammar == GrammarState.COMPRESSED_ANSWER:
            ops.append(MetalOp.KV_COMPRESS)
            ops.append(MetalOp.KV_FOLD)

        if self.state.syntax_mask is not None:
            ops.append(MetalOp.SYNTAX_MASK)

        ops.append(MetalOp.EMBEDDING_LOOKUP)

        self.state.metal_ops = ops
        return ops

    def apply_syntax_mask(
        self,
        vocab_size: int,
        allowed_tokens: Optional[set[int]] = None,
    ) -> list[bool]:
        """
        Apply grammar-based syntax masking.

        The model is NOT allowed to generate invalid structure.
        The grammar masks dumb token paths before they cost more tokens.
        """
        if allowed_tokens is None:
            # No restriction — all tokens allowed
            mask = [True] * vocab_size
        else:
            mask = [False] * vocab_size
            for t in allowed_tokens:
                if 0 <= t < vocab_size:
                    mask[t] = True

        self.state.syntax_mask = mask
        return mask

    def set_proof_obligation(self, obligation: Optional[str]):
        """Set the current proof obligation — what must be verified."""
        self.state.proof_obligation = obligation
        if obligation is not None:
            self.state.grammar = GrammarState.VERIFYING

    def step(self) -> ParserState:
        """Advance one generation step and return current state."""
        self.state.step += 1
        return self.state

    def summary(self) -> dict:
        """Summary of parser state for receipts."""
        return {
            "step": self.state.step,
            "grammar": self.state.grammar.name,
            "kv_efficiency": round(self.state.kv_efficiency, 4),
            "fetch_precision": round(self.state.fetch_precision, 4),
            "syntax_mask_ratio": round(self.state.syntax_mask_ratio, 4),
            "metal_ops": [op.value for op in self.state.metal_ops],
            "proof_obligation": self.state.proof_obligation,
            "grammar_history": [g.name for g in self.grammar_history],
        }
