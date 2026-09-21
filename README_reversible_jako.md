# Reversible-integer Japanese → Korean translation

A tiny sequence-to-sequence translator trained end to end with exact reversible
integer propagation and no depth-dependent activation cache.

The question this answers is not "is this a good translator" — it is not. It is:

> Can a real JA→KO seq2seq model be trained with exact reversible integer
> propagation, without caching per-layer activations?

All five conditions hold simultaneously: real parallel language data, a real
sequence model, actual learning, actual generation, exact reconstruction, and a
cache-free backward.

## Files

| file | role |
|---|---|
| `reversible_jako_data.py` | OPUS corpus download, cleaning, byte-level BPE, encoding |
| `reversible_jako_mt.py` | model, integer attention/FFN, manual cache-free backward, train/eval/generate |
| `reversible_jako_verify.py` | Gates 2–5 |
| `train_jako.sh` | the exact commands used |
| `jako_results.json`, `jako_gate6_overfit.json`, `jako_gates.json`, `jako_memory.json` | measured results |

## Reused, not reimplemented

Every integer primitive comes from the modules already verified in this project:

- `reversible_integer_final`: `RunConfig`/`set_config`, `wrap_state`, `saturate`,
  `rshift0`, `pow2norm`, `norm_block`, `SharedGrad`, `align_shared`,
  `dense_linear_backward`, `BlockSGD` (integer error-feedback, per-layer and
  batched stepping)
- `reversible_arch`: `bmm_backward_shared`, `score_act`, `score_act_backward`

## Integer semantics

Names here are easy to misread, so they are stated exactly:

| knob | meaning |
|---|---|
| `weight_bits = 8` | **signed**. Weights clamped to `[-2^7, 2^7-1]`, initialised uniform in `[-16, 16]` (`±2^(bits-4)`) |
| `state_bits = 16` | **signed two's-complement wrap** width of the reversible state, values in `[-2^15, 2^15)`. Wrap is exactly invertible, so reversibility survives it |
| `accum_bits = 32` | **signed saturating** width of GEMM accumulators |
| `hidden_bits = 16` | **magnitude** bits of the backward adjoint mantissa, carried alongside a separate sign; the discarded scale lives in a per-tensor exponent. This is the "grad-bits" knob |

`hidden_bits` has both a floor and a **ceiling**. Too few bits and the adjoint
collapses to zeros; above roughly 24 bits `norm_block` left-shifts the mantissa
far enough that the adjoint GEMMs overflow int64. Measured usable window on this
model: 10–24 bits, cosine 1.0000 against the float oracle throughout. 16 is used.

## Architecture

State is `[a, b]` over `(B, T, 2·d)`. Each block is an additive coupling
`[a,b] → [a, b + F(a)]` followed by a half swap, so `F` may be attention or an
FFN without affecting invertibility.

- encoder: `attn, ffn, …` (`k_enc` blocks)
- decoder: `self-attn, cross-attn, ffn, …` (`k_dec` blocks)
- attention is single-head with a ReLU score map and power-of-two rescaling —
  no softmax, no exp/log/division anywhere in the strict integer path
- cross-attention reads the encoder memory, whose width is `2·d`, so its K/V
  projections take `2·d` inputs while Q takes `d`
- loss is a per-token multiclass margin (hinge); PAD positions are excluded.
  No cross-entropy is used in the integer path
- decoding is greedy argmax over logits, no softmax

Embedding tables and the output projection are **not** reversible and their
activations are stored. That cost is O(1) in reversible depth and is reported as
a separate line item; it is never folded into "activation memory".

## Data

Real JA–KO parallel text, not substituted from any other language pair.

| corpus | URL | licence | raw lines | clean pairs |
|---|---|---|---:|---:|
| OPUS Tatoeba v2023-04-12 | `https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/ja-ko.txt.zip` | CC-BY 2.0 FR | 664 | 655 |
| OPUS TED2020 v1 | `https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/ja-ko.txt.zip` | CC BY-NC-ND 4.0 (TED) | 358,636 | 297,504 |

SHA-256 of both archives is recorded in the results JSON. Cleaning drops empty
lines, length outliers, script mismatches (Korean side must contain Hangul, the
Japanese side kana or CJK) and exact duplicates. Tokeniser is a shared byte-level
BPE over both languages.

## Commands

See `train_jako.sh`. Single process throughout; threads pinned with
`--threads 4` plus `OMP_NUM_THREADS`/`MKL_NUM_THREADS`, because oversubscription
has previously cost far more than it bought on this machine.

## Known limitations

- Translation quality is poor. The model learns Korean output, Korean morphology
  and frequent vocabulary, but collapses onto frequent prefixes and repeats
  tokens. It is under-trained, not competitive with any real MT system.
- Greedy decoding re-runs the whole decoder per timestep; no KV cache.
- Single head, no layer norm, no positional encoding beyond what the coupling
  and permutation provide.
- The integer optimizer emits at most ±1 per weight per step, which bounds how
  fast the model can move.
- Gate 6 is a partial overfit, not a full one.
