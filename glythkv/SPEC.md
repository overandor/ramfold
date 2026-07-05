# GlythKV / Metal^932 — Parser-Driven Memory Management for LLM Inference

## Thesis

**The parser becomes the memory manager.**

Not the model. Not the prompt. Not the browser. The parser.

At every generation step, the parser determines:
- Valid next token shapes (syntax masking)
- Which memory shards to fetch (context control)
- Which KV blocks are alive vs dead (KV eviction)
- What tensor work goes to Metal (operator placement)
- What needs verification (proof obligations)

## The New Machine Unit

A token alone is dumb. The unit is:

```
Token + ParserState + KVState + FetchState + ProofState
```

## Four Wastes

1. **Syntax waste** — grammar masks invalid token paths before they cost compute
2. **Context waste** — parser fetches only what current answer state needs
3. **KV waste** — dead KV blocks get evicted, folded, or cooled based on grammar state
4. **Operator waste** — hot path (attention, masks, KV compression) goes to fused Metal kernels

## .glyth Grammar

A `.glyth` program specifies:

```
pin          — must stay in RAM
compress     — can be compressed
lossy        — can be lossy
proof        — needs verification receipt
cpu          — runs on CPU
gpu          — runs on GPU/Metal
ram          — stays in RAM
disk         — can move to disk
no_cloud     — forbidden from cloud
kv_keep      — KV block preserved
kv_evict     — KV block evicted
metal_kernel — specific Metal kernel to run
receipt      — proof that output is not nonsense
```

## Adaptive Grammar

The grammar changes during inference based on:

- **Task state** — code repair → code-patch grammar; evidence needed → evidence-repair grammar
- **RAM pressure** — high pressure → compressed-answer grammar
- **Proof obligation** — changes in verification requirements change the grammar

## Architecture

```
glythkv/
  parser/           — adaptive parser that controls memory during generation
  grammar/          — .glyth grammar definitions and state machine
  kv_manager/       — KV cache eviction, folding, cooling
  fetch_controller/ — context shard fetching based on parser state
  metal/            — Metal kernels for attention, KV compression, masking
  receipts/         — proof obligations and verification receipts
  metrics/          — Verified Intelligence Density, Memory Alpha
```

## Metrics

| Metric | Definition |
|--------|-----------|
| **VID** | Verified Intelligence Density = verified_task_score / peak_unified_memory |
| **Memory Alpha** | VID(GlythKV) - VID(baseline) |
| **KV Efficiency** | useful_kv_blocks / total_kv_blocks at each step |
| **Fetch Precision** | fetched_shards_used / fetched_shards_total |
| **Syntax Mask Ratio** | masked_tokens / total_vocab at each step |
| **Metal Op Savings** | GPU_time_saved vs CPU_baseline |

## Connection to RAMFold

RAMFold proved: memory policy is an optimization object for **training**.
GlythKV proves: parser-driven memory management is an optimization object for **inference**.

Together: one closed-loop system where memory policy is optimized across both training and inference on Apple unified memory.
