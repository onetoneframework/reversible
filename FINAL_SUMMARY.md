# JA→KO reversible-integer translator — final summary

All experiments stopped 2026-09-06 22:20 at the user's request. This is the
complete record of what was found, what was fixed, what failed, and where the
work actually stands.

**Bottom line: the goal was not reached.** No model produced translates
held-out Japanese into Korean. Four defects were found and fixed along the way,
three of them real and one of them decisive, and the failure is now localised
to a specific place instead of being unexplained.

---

## 1. What was proven, in order

### 1.1 A causality bug in attention (fixed)

The per-row shift in `pow2norm` was computed **before** the mask was applied, so
the maximum that set the scale included masked-out entries — future target keys
in decoder self-attention, PAD source keys in cross-attention. Teacher forcing
therefore had a scale side-channel to information that free-running decoding
does not have.

Found by mutating target tokens after a cut and walking the decoder tensor by
tensor: first divergence at `attn-self.raw → shift → S → Aw`, propagating to the
logits before the cut.

Fixed with `masked_pow2norm`, whose reduction runs over valid entries only. Same
bit-length rule, same arithmetic shift, same clamp. Gates added
(`reversible_jako_causalgate.py`): future-target mutation at cut 1/3/6, padding
invariance, masked-memory garbage. The old code fails all three with 28–30
leaked tensors; the fixed code passes.

### 1.2 An instability in the integer optimizer (fixed by a documented condition)

`BlockSGD`'s error-feedback accumulator gains up to `2**param_bits − 1` per step
and drains `2**shift` when it fires. With `shift = (param_bits−1) + lr_bits`, the
accumulator is bounded only when

```
2**shift > 2**param_bits − 1      i.e.   lr_bits >= 1
```

At `lr_bits = 0` the clip to ±1 actually engages and the residual grows without
bound. Measured on a 32-way lookup at `param_bits = 12`:

| lr_bits | drain | observed \|acc\|max | 32-way accuracy |
|---:|---:|---:|---|
| 0 | 2,048 | 2,366–5,260 (unbounded) | 6–25%, never converges |
| 1 | 4,096 | exactly 4,095 | **100% by epoch 400** |
| 2 | 8,192 | exactly 8,191 | **100% by epoch 400** |
| 3 | 16,384 | ≤ 16,382 | **100% by epoch 400** |

Every earlier small run had used `lr_bits = 0`, which invalidates their
conclusions. The condition is now in `BlockSGD`'s docstring and exposed as
`self.stable`.

### 1.3 The encoder → cross-attention → decoder path is sound

With a stable optimizer, source-only tasks are learned perfectly:

| task | accuracy (real source) | accuracy (shuffled source) |
|---|---:|---:|
| 32 distinct source tokens → random target token, loss at position 0 only | **100%** | 0% |
| 2–4 token sources → single random label | **100%** | 0% |

Chance is 3.12%. The hypothesis "cross-attention wiring is broken" is rejected.

### 1.4 The decisive defect: attention needed softmax's **backward**, not its forward

A float mirror — identical reversible coupling, positional tables, hinge loss,
data — was used to bisect. Each rung changes one thing:

| rung | forward | backward | greedy reproduction (64 pairs) |
|---|---|---|---:|
| float + softmax | softmax | softmax | 64/64 |
| softmax + **every integer rescale** | softmax | softmax | **64/64** |
| softmax removed | leaky power-of-two | matching | 0/64 |
| **softmax forward, diagonal backward** | softmax | no −w_i·w_j terms | **0/64** |
| **integer base-2 forward, softmax backward** | power-of-two weights | softmax | **64/64 in 1000 steps** |

So: integer rescaling is innocent; the forward *shape* of softmax is not needed;
what generation requires is the backward's row coupling

```
∂w_i/∂s_j = w_i (δ_ij − w_j)
```

the term through which raising one score lowers every other weight in the row.

Six earlier hypotheses were tested and rejected before this: prefix corruption
(made it worse), scheduled sampling, row-mass L1 normalisation at four targets,
score resolution 7→20 bits, non-negative vs leaky score maps, and query
collapse.

### 1.5 The integer implementation of it

`exp2_attend` — base-2 attention, `w_i = 2**(((s_i − max_j s_j) >> k) + cap)`,
masked before the maximum. `exp2_attend_backward` with the coupling:

```
dL/ds_i = w_i (g_i − gbar),   gbar = floor( Σ_j w_j g_j / Σ_j w_j )
```

One int64 dot product and one floor division per row. No exponential, no float.
Gates: off-path bit-identical, cache-free == cached, encoder inverse exact,
per-layer == batched, causality A/B/C pass, and cosine against the float coupled
Jacobian on identical inputs = **0.99392** (against the diagonal Jacobian:
0.65094).

Result on the integer model, one change from the previous formulation:

| variant | TF content | pos0 | greedy /64 |
|---|---:|---:|---:|
| leaky (previous) | 27.7% | 1.6% | **0** |
| base-2, diagonal adjoint | 19.0% | 1.6% | **0** |
| **base-2, coupled adjoint** | **91.5%** | **92.2%** | **27**, rising |

### 1.6 Positional encoding was also required

Learned integer positional tables added (`--positional N`, appended after
`Wout` so old checkpoints still load). At 32 pairs, with everything else equal:

| variant | greedy /32 | pos0 |
|---|---:|---:|
| no positional encoding | 0/32 | 3.1% |
| prefix corruption 50% | 0/32 | 3.1% |
| scheduled sampling 25% | 1/32 | 15.6% |
| **positional encoding** | **32/32** | **100%** |

### 1.7 How far the fixed system gets

| pairs | greedy reproduction | TF content | source gap (real − shuffled) |
|---:|---|---:|---:|
| 32 | **32/32** | 100% | +85 |
| 64 | **53/64** | 97.5% | +88 |
| 128 (d96) | **111/128** | 98.6% | +85 |
| 128 (d64) | 0/128 | 23% | +19 — width mattered |
| 512 | **0/512** | ≤ 67% | ≈ 0 |
| 8,000 (held-out) | — | ≤ 0.7% | ≈ 0 |

Before these fixes the integer model reproduced **nothing** above 32 pairs.

---

## 2. Where it fails, and the one thing that would settle it

512 pairs is the wall. Above it, teacher-forced accuracy can still rise while
the source contributes nothing — the model falls back on a target-side language
model. The clearest evidence: integer momentum raised teacher-forced content to
67% at 512 while scoring **higher with the encoder memory zeroed** than with the
real source. Momentum averages over ~2^k steps, which keeps whatever is
consistent across batches, and what is consistent is the language model, not the
per-sentence source signal.

At 512 pairs, float + Adam reaches 503/512 where the float replica of the
integer optimizer's arithmetic reaches 78/512. So the optimizer is implicated —
but this was measured on *memorisation*, not translation.

**The experiment that was never run, and should be run first:** float + Adam on
the 8,000-pair set with a held-out split. Nothing in this session establishes
that *any* model — float or integer — can translate at this data scale. The
training data is 7,600 TED-talk subtitle pairs; the acceptance set is everyday
conversational Japanese. That is a data-scale problem and a domain mismatch on
top of it, and if the float model also scores ~0 on held-out data then the
entire integer investigation has been chasing a problem that is not
integer-specific.

I was building that float held-out harness when work was stopped
(`reversible_jako_floatdiag.py` has the split and metrics wired; one helper
function reference is still unresolved).

A further self-inflicted constraint: the corpus was capped at 8,000 pairs to
match an earlier run. **187,725 pairs are available** at the same length limit,
and the mixed corpus with OpenSubtitles adds more.

---

## 3. Code left behind

Production path (unchanged in its guarantees: integer forward, exact additive
reversible coupling, cache-free reconstructed backward, tensor-shared gradient
exponent, manual integer backward, integer optimizer, no autograd or softmax in
the production path, no activation tape):

| file | what changed |
|---|---|
| `reversible_jako_mt.py` | `masked_pow2norm`; base-2 attention wiring; positional tables; `corrupt_prefix`; `scheduled_sample`; chunked/sparse output head; source-gated checkpoint selection; `_last.pt` and epoch snapshots |
| `reversible_arch.py` | `exp2_attend`, `exp2_attend_backward` (coupled adjoint), `SCORE_EXP2*` |
| `reversible_integer_final.py` | `BlockSGD` stability condition documented, `self.stable`, `momentum_bits` (gated, off — it removes source usage) |
| `reversible_jako_data.py` | OpenSubtitles v2024 with a strict filter, corpus tags, acceptance blocking (6/30 sentences were leaking), alphabet cap |

Diagnostics written: `reversible_jako_causalgate.py`, `_leakhunt.py`,
`_breakdown.py`, `_crossdiag.py`, `_querydiag.py`, `_oracle.py`, `_overfit32.py`,
`_lookup.py`, `_metrics.py`, `_headgate.py`, `_probe.py`, `_bisect.py`,
`_floatdiag.py`.

Records: `overnight_translation_report.md` (every run, appended as it happened),
`jako_run_journal.md`, `WHAT_WENT_WRONG.md` (where I failed to follow the
instructions), per-run logs and JSON in this directory.

---

## 4. What I would do next, in order

1. **Float + Adam, 8k, held-out.** Bounds everything else. Cheap.
2. If that fails: the problem is data scale and domain, not the integer stack.
   Retrain on 100k+ pairs from the available 187k, or on the mixed corpus, and
   replace the everyday-conversation acceptance set with one drawn from the
   training domain.
3. If that succeeds: the integer gap is real. Chase it at 8k with the float run
   as a matched control at every step — never again measure an integer change
   without one.
4. Do not select checkpoints on accuracy alone. Require
   `content_acc(real) > content_acc(shuffled)` and `real > zeroed`; this is now
   in the trainer and it would have rejected the momentum models that looked
   best by accuracy.
