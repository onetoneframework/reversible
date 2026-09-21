# Overnight: driving the reversible-integer JA→KO system to real translation

Append-only. Every run, hypothesis, failure and code change goes in here.
Numbers are measured; anything not measured says so.

Working directory: `res/Platform/Python/reversible`
Machine: Core Ultra 5 135H, 18 logical cores, 15.9 GB RAM, Windows 11, no CUDA.
Python: `%LOCALAPPDATA%\Programs\Python\Python313\python.exe` (the `python` on
PATH is a broken MinGW stub).

---

## 1. Repository state at the start of this mandate (2026-09-06 ~06:25)

Production path unchanged in its guarantees: integer forward, exact additive
reversible coupling, cache-free reconstructed backward, tensor-shared gradient
exponent, manual integer backward, integer optimizer, no autograd or softmax in
the production path, no activation tape.

Code changes already landed tonight:

| change | file | why |
|---|---|---|
| `masked_pow2norm` — attention score shift taken over valid entries only | `reversible_jako_mt.py` | causal/PAD leakage through the per-row shift |
| `BlockSGD` stability condition documented, `self.stable` exposed | `reversible_integer_final.py` | `lr_bits=0` diverges |
| `corrupt_prefix` — integer UNK substitution on the decoder input | `reversible_jako_mt.py` | target-prefix shortcut |
| chunked/sparse output head (`chunked_head`, `sparse_head_backward`) | `reversible_jako_mt.py` | 41 MB logits + 41 MB gradient per step |
| per-corpus validation, content-based checkpoint selection, `_last.pt` | `reversible_jako_mt.py` | pooled accuracy is 83% EOS |
| acceptance blocking on every corpus, alphabet cap ≥20 | `reversible_jako_data.py` | 6/30 acceptance sentences were leaking |
| OpenSubtitles v2024 + strict filter, corpus tags | `reversible_jako_data.py` | 50k mixed corpus |

New diagnostics written tonight: `reversible_jako_causalgate.py`,
`reversible_jako_leakhunt.py`, `reversible_jako_breakdown.py`,
`reversible_jako_crossdiag.py`, `reversible_jako_querydiag.py`,
`reversible_jako_oracle.py`, `reversible_jako_overfit32.py`,
`reversible_jako_lookup.py`, `reversible_jako_metrics.py`,
`reversible_jako_headgate.py`, `reversible_jako_probe.py`.

## 2. Known bugs — status

| bug | status | evidence |
|---|---|---|
| attention shift computed over masked future/PAD scores | **FIXED** | causality gate A cut=1/3/6 PASS; old order fails with 28–30 leaked tensors |
| `BlockSGD` accumulator diverges at `lr_bits=0` | **FIXED by policy** (lr_bits ≥ 1) | `\|acc\|max` 5260 > Q=2048 at lr0; exactly Q−1 at lr1/2/3 |
| pooled token accuracy dominated by EOS | **FIXED** (selection on balanced content acc) | 83.2% of correct tokens were EOS |
| 6/30 acceptance sentences present in training data | **FIXED** | leakage 0/30 |
| no explicit positional encoding | **OPEN, not yet the intervention** | source-only lookups pass without it |
| decoder learns `p(y_t \| y_gold_<t)` and ignores the source | **OPEN — the current target** | prefix worth −58.43pp, source −2.25pp |

## 3. Hypotheses, resolved

**H1 cross-attention wiring is broken — REJECTED.**
Single-token source → single random target label, 32-way, loss at position 0
only: `acc_real 100%`, `acc_shuffled 0%`, `rank_real 1.0` by epoch 400 at
lr_bits 1/2/3. Multi-token source (2–4 tokens) → single label: same, 100% / 0%,
`rank_shuffled` 712. The encoder → cross-attention → decoder → output path
carries source identity and learns it quickly.

**H2 query collapse — REJECTED at this depth.**
Causal-fixed d96/dec4: Q row cosine 0.12–0.14, common mode 0.36–0.41, flat from
ep0 to ep7. The collapse seen in the buggy d256/dec8 run (Q cos 0.993, common
mode 0.98) was a depth effect and did not reproduce. BOS-anchor intervention is
implemented and kept on the shelf for deep models.

**H3 causal leakage explains the translation failure — REJECTED as sufficient.**
Fixed, gates pass, and run2 (causal fix, lr_bits 0) still reached only 1.26%
balanced content accuracy with source gap oscillating around zero.

**H4 optimizer instability explains it — REJECTED as sufficient.**
run3 lr_bits 1 and 2 are stable; at matched epochs the source gap stayed within
±0.22pp and lexical anchors stayed 0/8.

**H5 (current) the decoder takes the gold-prefix shortcut and never learns the
source.** Supported by: prefix ablation 2×2 (losing the prefix costs 58.43pp,
losing the source 2.25pp; the two wrong-prefix cells agree to 7.3034%), oracle
prefix sweep climbing 0/32 → 3 → 4 → 9 → 25 with k, and shuffled sources scoring
the same or better at every k.

## 4. Runs

### run1 — d256 e4/d8, 50k mixed — INVALID (causal leak), archived
`run1_d256_50k_INVALID/`. Stopped after ep7. Never resume from it.

### run2 — d96 causal fix, lr_bits 0 — completed, unstable optimizer
17 epochs, early stop on patience 6. Best balanced content 1.2621% (ep10).
Source gap oscillated ±0.44pp, anchors 0/8. Retrospectively invalid as a
learning-rate baseline because `lr_bits=0` violates the accumulator bound.

### run3 lr_bits=2 — STOPPED at epoch 23 (PID 14456)
Reason: stable, but consistently behind lr_bits=1 in this d96 setting
(content 0.295% vs 1.107% at ep18) and produced blank greedy outputs on 2 of 6
arbitrary probes. Kept: `jako_small_lr2_best.pt`, `_small_lr2.log`,
`jako_translations_lr2.txt`.

### run3 lr_bits=1 — RUNNING (PID 18004), matched clean control
Kept deliberately: the prefix-corruption run needs a clean-prefix control at
the same epochs, and this one was still improving at ep18.

### run4 — prefix corruption 50% — see section 5.

---

## 5. run4 — target prefix corruption 50%

Launched 06:23:02 (PID 20108), **restarted 06:29:05 (PID 32188)** at epoch 3 to
pick up the log fields this mandate requires (`train_prefix_corruption`,
`eval_prefix_corruption`, `optimizer_stable`, source conditions under all three
sources, blank-output count). Nothing scientific was lost: same seed, same data,
same architecture, cold start either way.

```
corpus both, 7,600 train / 400 valid   (ted2020 7,549 | tatoeba 51)
vocab 5000 char-BPE, alphabet cap >=20, acceptance leakage 0/30
d96 ffn192 enc2 "af" dec4 "ccaf"  params 2,178,048
lr_bits 1 (optimizer_stable true)  hidden_bits 16  score_leak 3
max_len 16  max_tok 14  batch 16  seed 0  patience 8
train_prefix_corruption 0.50   eval_prefix_corruption 0.00
query_anchor false   positional_encoding false
```

Matched control: run3 lr_bits=1, identical except corruption 0%.

### Code changes made for this run

| change | invariants checked |
|---|---|
| `corrupt_prefix` — UNK substitution on the decoder input, integer threshold | BOS preserved, PAD preserved, only UNK substituted |
| header logs the corruption policy and optimizer stability | — |
| `blank_outputs` counted in the free-running probe | — |
| one-line metric now shows content and pos0 under real/shuffled/zeroed | — |

### Prepared but NOT applied (kept off, gated)

| capability | flag | gates |
|---|---|---|
| integer scheduled sampling (CASE 5B) | `--scheduled-sampling PCT` | BOS/PAD preserved; **causality**: mutating gold tokens at/after a cut leaves substitutions before it unchanged |
| learned integer positional tables (CASE 6) | `--positional N` | `positional=0` bit-identical to the pre-option model (23 params, identical gradients); with tables: cache-free == cached, encoder inverse exact, per-layer == batched; causality gates A/B/C PASS with PE on; explicit error when a sequence exceeds the table |

The positional tables are appended after `Wout` so every existing optimizer
index keeps its meaning and older checkpoints still load.


---

## 6. Autonomous ladder (overnight_driver.py)

Started 2026-09-06 10:41:27. Gate: 32-pair greedy reproduction >= 30/32 with teacher-forced content >= 95%. Batches run in parallel; the driver advances or drops to the next rung without waiting to be woken.

---

## 6. Autonomous ladder (overnight_driver.py)

Started 2026-09-06 10:45:04. Gate: 32-pair greedy reproduction >= 30/32 with teacher-forced content >= 95%. Batches run in parallel; the driver advances or drops to the next rung without waiting to be woken.

---

## 6. Autonomous ladder (overnight_driver.py)

Started 2026-09-06 10:46:03. Gate: 32-pair greedy reproduction >= 30/32 with teacher-forced content >= 95%. Batches run in parallel; the driver advances or drops to the next rung without waiting to be woken.

### batch2 — combinations and stronger settings

| variant | args | best greedy /32 | best give1 /32 | best TF content | real/shuf/zero (last) |
|---|---|---:|---:|---:|---|
| pos16_ss25 | `--positional 16 --scheduled-sampling 25` | **0** | 0 | 0.000% | 0.0 / 0.0 / 0.0 |
| pos16_pc50 | `--positional 16 --prefix-corrupt 50` | **0** | 0 | 0.000% | 0.0 / 0.0 / 0.0 |
| ss50 | `--scheduled-sampling 50` | **0** | 0 | 0.000% | 0.0 / 0.0 / 0.0 |
| pc75 | `--prefix-corrupt 75` | **0** | 0 | 0.000% | 0.0 / 0.0 / 0.0 |

Rung failed. Best: **pos16_ss25**, greedy 0/32, TF content 0.000%. Dropping to the next rung.


### batch3 — capacity and precision on the plain task

| variant | args | best greedy /32 | best give1 /32 | best TF content | real/shuf/zero (last) |
|---|---|---:|---:|---:|---|
| hb20 | `--hidden-bits 20` | **0** | 0 | 0.000% | 0.0 / 0.0 / 0.0 |
| hb24 | `--hidden-bits 24` | **0** | 0 | 0.000% | 0.0 / 0.0 / 0.0 |
| lr2 | `--lr-bits 2` | **0** | 0 | 0.000% | 0.0 / 0.0 / 0.0 |
| pos16_lr2 | `--positional 16 --lr-bits 2` | **0** | 0 | 0.000% | 0.0 / 0.0 / 0.0 |

Rung failed. Best: **hb20**, greedy 0/32, TF content 0.000%. Dropping to the next rung.


### batch4 — longer budget on the two best plain settings

| variant | args | best greedy /32 | best give1 /32 | best TF content | real/shuf/zero (last) |
|---|---|---:|---:|---:|---|
| pos32_dec6 | `--positional 16 --k-dec 6 --dec-pattern ccafca` | **0** | 0 | 0.000% | 0.0 / 0.0 / 0.0 |
| base_long | `` | **-1** | -1 | -1.000% | None / None / None |
| pos16_long | `` | **-1** | -1 | -1.000% | None / None / None |
| pos16_ss25_long | `` | **-1** | -1 | -1.000% | None / None / None |

Rung failed. Best: **pos32_dec6**, greedy 0/32, TF content 0.000%. Dropping to the next rung.


Ladder exhausted without passing STAGE 3. Next: float/autograd diagnostic to separate the integer training algorithm from the architecture and objective (mandate section 7).


---

## 7. STAGE 3 PASSED — the cause was the missing positional encoding

Four variants, 32-pair translation overfit, cold start, stable optimizer
(`lr_bits=1`), causal fix on, one change each. Same seed, same 32 pairs, same
architecture otherwise.

| variant | change | TF content @ep3000 | pos0 | **greedy /32** | give1 | give3 |
|---|---|---:|---:|---:|---:|---:|
| base | none | 75.84% | 3.12% | **0/32** | 7 | 8 |
| pc50 | `--prefix-corrupt 50` | 27.53% | 3.12% | **0/32** | 0 | 0 |
| ss25 | `--scheduled-sampling 25` | 56.74% | 15.62% | 1/32 | 2 | 4 |
| **pos16** | `--positional 16` | **100.00%** | **100.00%** | **32/32** | 32 | 32 |

`pos16` stopped early at epoch 3000 having reproduced all 32 training sentences
exactly under greedy decoding, with source conditioning intact
(real 100.0 / shuffled 82.58 / zeroed 68.54). Checkpoint: `of32_pos16.pt`.

Trajectory: content 17.42% at ep250, 47.19% at ep500, 73.60% at ep1000,
97.19% at ep2250, 100% at ep3000; pos0 3.12 -> 21.88 -> 100.

### Hypotheses settled

**H5 gold-prefix shortcut is the cause — REJECTED as the cause.** Both
interventions aimed at it made things worse, not better: prefix corruption cut
teacher-forced content accuracy from 75.84% to 27.53% and left greedy at 0/32;
scheduled sampling reached 1/32. Weakening the prefix removed signal without
redirecting it to the source. The shortcut was real and measurable, but it was a
*symptom* of the decoder having no other way to order its output, not the cause.

**H6 missing positional encoding — ACCEPTED.** It is the only change that moved
greedy reproduction off zero, and it moved it to 32/32.

Why the earlier evidence pointed away from it: the single-token and multi-token
source lookups both passed without positional encoding, which correctly showed
the encoder → cross-attention → decoder path works. But those tasks have one
output position and no order to represent. Passing them says nothing about
whether a sequence decoder can order its output, and that is what was missing.

`base` is the clean statement of the old failure: 75.84% teacher-forced content
accuracy and 0/32 under greedy. Teacher forcing supplies the order; without
positional information the model has no way to supply it itself.

## 8. STAGE 4 — 8k with positional encoding

Launched 11:17:58, PID 32068. Identical to the run3 `lr_bits=1` control except
`--positional 16`.

```
8,000 pairs (7,600 train / 400 valid), d96 ffn192 enc2 "af" dec4 "ccaf"
causal fix on, lr_bits 1 (stable), score_leak 3, positional 16
60 epochs, patience 10, snapshots at ep 0/5/10/20/40
per-epoch probe translations -> jako_translations_stage4.txt
```

The control it replaces reached balanced content 1.88%, source gap oscillating
within ±0.67pp, and 0/8 lexical anchors across 36 epochs. Those are the numbers
to beat.

### batch1 — one intervention each, stable optimizer

| variant | args | best greedy /32 | best give1 /32 | best TF content | real/shuf/zero (last) |
|---|---|---:|---:|---:|---|
| pos16 | `(baseline)` | **32** | 32 | 100.000% | 100.0 / 82.5843 / 68.5393 |
| base | `(baseline)` | **3** | 10 | 76.966% | 73.5955 / 61.236 / 67.4157 |
| ss25 | `(baseline)` | **2** | 9 | 75.281% | 72.4719 / 65.7303 / 59.5506 |
| pc50 | `(baseline)` | **0** | 0 | 47.191% | 47.191 / 45.5056 / 58.427 |

**STAGE 3 PASSED** by: pos16


---

## 9. Attention formulation search (attn_driver.py)

Started 2026-09-06 16:18:12. 64 pairs, d64, quick evaluation; a rung takes a couple of minutes. Pass: greedy >= 48/64, which the plain formulation cannot reach.

---

## 10. Re-examination after base-2 attention failed (17:10)

Base-2 attention (`w = 2**(((s - rowmax) >> k) + cap)`, integer, gated) produced
competitive rows — the largest weight carries 65.6% of a row against 12.5% for
the linear map — and still reproduced 0/64 at k = 2/3/4 and cap = 8/12/16. Best
teacher-forced content 39.4% (k=4) against the leaky baseline's 27.7%.

What the earlier bisection actually proved, re-read strictly:

| rung | condition | greedy | reading |
|---|---|---:|---|
| 0 | float + softmax | 64/64 | reference |
| 5 | softmax + every pow2 rescale | 64/64 | integer rescaling is not the cost |
| 1 | softmax removed, no pow2 | 0/64 | loss 1.4e8 — numerical blow-up, not a shape comparison |
| 2 | softmax removed + pow2 | 1/64 | the informative failure |
| 3, 4 | + weight quantisation / sign updates | 0/64 | **invalid**: the quantiser used a 1/16 grid and killed learning outright (TF 0%) |

So the proven statement is narrower than "competition is missing": swapping the
whole of softmax — forward shape *and* backward — for leaky+pow2 costs
generation. Base-2 attention copied the forward's shape (row max subtracted,
geometric fall-off) and did not recover it. What it did not copy is softmax's
backward coupling, ∂w_i/∂s_j = w_i(δ_ij − w_j): the term through which raising
one score lowers every other weight in the row. The integer exp2 backward
treats each entry independently.

Two float rungs added to separate the two:

- **rung 6** softmax forward, normaliser detached in the backward (diagonal
  approximation, no −w_i w_j terms). If this fails, the coupling is what
  generation needs.
- **rung 7** the integer base-2 forward in float (power-of-two weights,
  normalised to sum 1), with softmax's full backward substituted through a
  straight-through surrogate. If this passes, the forward shape is fine and
  only the integer backward needs the coupling.

Launched 17:10:17, 64 pairs, 3000 Adam steps, 6 threads each.

Also noted while re-checking the integer run: `accum_saturations` = 14 per
step with base-2 attention at cap 12. The saturation is in the attention
backward (`bmm_backward_shared(gO, Aw, V)`: 2^16 · 2^12 · T_k reaches 2^32),
not the forward. cap 8 removes it and did *worse* (5.99%), so saturation is not
the binding failure either, but the cap will need a bound tied to hidden_bits
if base-2 attention is kept.

### Rungs 6 and 7 — the answer (17:13)

| rung | forward | backward | greedy | TF real / shuffled |
|---|---|---|---:|---|
| 6 | softmax | diagonal (normaliser detached, no −w_i w_j) | **0/64** | 21.0% / 20.3% |
| 7 | **integer base-2 shape** (power-of-two weights, row max subtracted) | softmax's full backward (straight-through) | **64/64 @ 1000 steps** | 100% / 38.4% |

Softmax's forward shape is not what generation needs; the base-2 integer
forward is sufficient. What it needs is the backward's row coupling,
∂w_i/∂s_j = w_i(δ_ij − w_j). Remove that from real softmax and it collapses
to the same real≈shuffled non-translation seen all night; add it to the
integer-shaped forward and it memorises 64/64.

### Integer coupled adjoint (`exp2_attend_backward`, `SCORE_EXP2_COUPLED`)

    dL/ds_i = w_i * (g_i − gbar),   gbar = floor( Σ_j w_j g_j / Σ_j w_j )

One int64 dot product and one floor division per row; no float, no exp. The
2^-k from the base-2 exponent and ln 2 fold into the existing right shift by
k; the row-mass scale is carried by the caller's shared exponent.

Gates with the coupled adjoint on (d48, positional 12, seed 0):

| gate | result |
|---|---|
| `SCORE_EXP2=None` path bit-identical to before | PASS |
| coupled ≠ diagonal (the change is live) | PASS |
| cache-free backward == cached | PASS |
| per-layer optimizer == batched (6 steps) | PASS |
| causality A cut=1/3/6, B padding, C masked-memory garbage | PASS |
| cos(integer adjoint, float coupled Jacobian) on identical S/mask/g | **0.99392** |
| cos(integer adjoint, float diagonal Jacobian) — control | 0.65094 |

Launched 17:15:09, 64 pairs, d64, 600 epochs, quick eval, one change each:
`leaky` (baseline), `k3_diag` (base-2, diagonal adjoint — the previous
failure), `k3_coupled`, `k4_coupled`, `k3c10_coupled`.

### Integer result with the coupled adjoint (17:24) — first generation from the integer model

64 pairs, d64, 600 epochs (1200 BlockSGD steps), one change each:

| variant | TF content | pos0 | real | shuffled | zeroed | greedy /64 |
|---|---:|---:|---:|---:|---:|---:|
| leaky (baseline) | 27.7% | 1.6% | 27.7 | 11.0 | 1.7 | 0 |
| base-2, diagonal adjoint | 19.0% | 1.6% | 19.0 | 19.2 | 1.0 | 0 |
| **base-2 k=3, coupled adjoint** | **91.5%** | **92.2%** | **91.5** | **2.2** | 5.2 | **27** |
| base-2 k=3 cap 10, coupled | 90.8% | 90.6% | 90.8 | 2.5 | 4.7 | 27 |
| base-2 k=4, coupled | 82.8% | 89.1% | 82.8 | 2.7 | 5.2 | 16 |

The coupled adjoint is the only change between rows 2 and 3. Source gap goes
from ≈0 to +89pp, position-0 accuracy from 1.6% to 92%, greedy reproduction
from 0 to 27/64 and still rising at the budget's end (7 → 9 → 27 over the last
three reports). This is the first time the integer model has generated any
training sentence under greedy decoding at a size above 32.

Not yet at the 48/64 line. Launched 17:25: 32 pairs (1500 ep), 64 pairs (2000
ep), 128 pairs (1500 ep), all base-2 k=3 coupled, 5 threads each.

`reversible_jako_mt.py` now records `score_exp2` and `exp2_coupled` in the run
header and in every checkpoint, and `load_ckpt` restores them, so an 8k run and
its acceptance evaluation use the same attention. Smoke-tested end to end.

### Coupled base-2 attention across sizes (17:47)

| pairs | epochs | greedy | TF content | pos0 | real / shuffled / zeroed |
|---:|---:|---:|---:|---:|---|
| 32 | 1350 (early stop) | **32/32** | 100% | 100% | 100.0 / 14.6 / 6.7 |
| 64 | 1800 of 2000 | **53/64** | 97.5% | 98.4% | 97.5 / 9.2 / 36.7 |
| 128 | 1500 | 6/128 | 64.3%, still rising | 53.9% | 60.9 / 2.9 / 37.7 |

64 pairs clears the ≥48/64 line that no integer variant had approached before
(previous best 0/64). 128 is undecided between budget and scale: its
teacher-forced content rose 45 → 64% over the last three reports.

Launched 17:48: 128 pairs d64 for 5000 epochs, 128 pairs d96 for 3000 epochs,
and STAGE 4 (8,000 pairs, d96, causal fix, lr_bits 1, positional 16, base-2
k=3 coupled attention; otherwise identical to the run3 control) — 16 threads
total. The mandate's condition for STAGE 4 — 32-pair greedy PASS — holds under
the new attention.

### Translation with the 64-pair coupled checkpoint (17:52, greedy only)

Own training sentences: **7/8 reproduced exactly** (the eighth differs by one
syllable, 올빼미악 for 올빼미를). Held-out probe (8) and arbitrary sentences
(3): Korean-shaped, no semantic correspondence — expected from a 64-sentence
memoriser and not a verdict on generalisation, which STAGE 4 (8k) is running to
answer. Recorded so the before/after is on file: no earlier integer checkpoint
reproduced a single training sentence above 32 pairs.

### 18:06 — 128 pairs was a capacity limit; STAGE 4 widened

| run | epochs | greedy | TF content | real / shuffled |
|---|---:|---:|---:|---|
| 128 pairs, d64, coupled | 500 of 5000 | 0/128 | 23.2% | 20.5 / 1.2 |
| 128 pairs, **d96**, coupled | 600 of 3000 | **28/128 and rising** | 87.2% | 87.1 / 1.2 |

Same attention, same data, same optimizer — width alone separates them. The
d64 run is stopped as superseded.

STAGE 4 (8k, d96, coupled) at ep7: content ≤ 0.18%, gap within ±0.22, anchors
0/8 — indistinguishable from every earlier 8k run at the same point. Per-
sentence exposure is ~7 passes against the 600 the 128-pair run needed, so it
is early, but the 128 result makes width the obvious parallel variable.
Launched 18:07: STAGE 4 at **d192 / ffn 384**, patience 25 (patience 10 on a
selection metric with 0.18% resolution would stop on noise), everything else
identical.

### 18:22 — 128 pairs at d96 with coupled attention

| epoch | greedy /128 | TF content | pos0 | real / shuffled / zeroed |
|---:|---:|---:|---:|---|
| 300 | 13 | 75.5% | 78.9% | 71.8 / 2.4 / 2.9 |
| 600 | 28 | 87.2% | 86.7% | 87.1 / 1.2 / 6.2 |
| 900 | 65 | 92.1% | 93.8% | 90.0 / 3.1 / 17.2 |
| 1200 | **86** | 96.3% | 93.8% | 96.7 / 5.7 / 32.0 |

Where the old formulation reached 0/128 at 6000 epochs with real≈shuffled,
this reproduces 86/128 by epoch 1200 with a 91-point source gap.

STAGE 4 8k: d96 at ep14 and d192 at ep2 both still show content ≤ 0.26%, gap
within ±0.33, anchors 0/8. Not a verdict yet at this exposure; the 8k regime
requires lexical generalisation rather than sentence identity, which no run so
far has been asked to do with a working attention. Next ladder rung (512
pairs, d96) is queued to start automatically when the 128-pair run finishes.

### 19:05 — 128 pairs PASS; 8k at d96 fails again; batch-size test launched

128 pairs, d96, coupled base-2: **111/128** greedy at ep2700 (TF content
98.6%, pos0 98.4%, real 98.3 / shuffled 13.8 / zeroed 66.3). STAGE 3 now holds
at 32, 64 and 128 with the same formulation. 512 pairs (d96) auto-started 19:04.

STAGE 4, 8k, d96, coupled: early-stopped at ep26, best balanced content 0.26%,
gap within ±0.11, anchors 0/8, and `zeroed > real` in every late epoch
(ep26: 0.998 vs 0.222) — the source path is net noise at this scale. d192 at
ep14 shows the same (content 0.44%, gap −0.11, zeroed 0.67 > real 0.22).

This is a discontinuity, not a slowdown: 128 pairs reached 75% content after
~9 passes per sentence; 8k has had 26 passes and sits at 0.3%. The variable
that separates the regimes is gradient consistency — a batch of 32 out of 128
sentences agrees with itself, a batch of 16 out of 7,600 does not, and with
±1 updates and per-tensor 12-bit renormalisation every step is the same size
whether it carried signal or noise. Single-variable test launched: 8k, d96,
**batch 64**, patience 25, otherwise identical. A 1,024-pair trainer run (with
held-out validation, probe and anchors) is queued behind the 512-pair overfit
to locate where generalisation begins between 128 and 8k.

### 19:55 — the wall is between 128 and 512; batch size and width ruled out at 8k

| run | result |
|---|---|
| 512 pairs, d96, coupled | ep450 (7,200 steps): **0/512**, TF content 0.49%, real 0.70 / shuffled 0.23 — flat. 128 pairs reached 75% by 1,200 steps. |
| 8k, d96, batch 64 | ep39: content ≤ 0.18%, gap ≈ 0, anchors 0. Batch size is not the 8k variable. |
| 8k, d192, batch 16 | ep27: content ≤ 0.66%, gap flipping ±0.66, zeroed 1.66 > real 0.11. Width is not the 8k variable. |

All three stopped as decided. With attention fixed, the integer model
memorises up to 128 sentences and not 512, while the float mirror with softmax
and Adam memorised 128 in 1,000 steps. The remaining difference is the
optimizer/gradient path at scale. The earlier bisection rung for the sign
update was invalid (its quantiser killed learning), so it is redone properly:

- `blocksgd_replica` (float, softmax + pow2): per-tensor renormalisation of the
  gradient to the 12-bit scale, error-feedback accumulator, ±1/16 quantum,
  clip 1, shift 12 — the integer optimizer's arithmetic on float weights.
- `adam` control at the same 512 pairs, d96.
- integer 512 with `lr_bits 2` and with `hidden_bits 24` (gradient resolution).

Launched 19:56, 4 × 4 threads.

### 20:00 — integer momentum implemented; supervisor started

`BlockSGD(momentum_bits=k)`: m ← m − (m ≫ k) + g_renorm, then the accumulator
takes renorm(m) instead of the raw block. Shift-only, per parameter. Gates:
k=0 bit-identical to before; per-layer == batched with k=3; accumulator stays
under 2^shift (|acc|max 4095, Q 4096); updates differ from k=0. Rationale: a
batch's renormalised gradient is a ±2^11 vote from that batch alone, and with
thousands of distinct sentences the votes cancel step to step, so ±1 weight
moves random-walk; averaging ~2^k steps before quantising lets consistent
directions accumulate.

Launched integer 512 with momentum 3 alongside the four optimizer-bisection
runs, and `optim_driver.py`, which reads the 512 results and chooses the next
rung itself (A: momentum sweep → 8k; B: gradient-resolution sweep; C: float at
d192 then integer d192), then launches STAGE 4 with the best setting.

---

## 11. optim_driver.py — supervising the 512-pair wall (2026-09-06 20:00:07)

512 pairs: float Adam 64/512, float BlockSGD-replica 78/512; integer lr2 None, hb24 None, mom3 None
Branch **C**.

### 20:23 — the optimizer is the wall (branch A)

512 pairs, d96, softmax + pow2 float mirror:

| optimizer | greedy /512 | note |
|---|---:|---|
| Adam | **503** (99.9% TF) at 7,000 steps | still running to 12,000 |
| BlockSGD replica (renorm to 2^11, error feedback, ±1/16 quantum, clip 1) | **78** peak, then NaN from ~9,000 steps | the integer optimizer's arithmetic on float weights |

Same model, same data, same attention: the ±1-quantum error-feedback update
reaches 78/512 where Adam reaches 503. (The NaN is a replica artefact —
unbounded float intermediates once the accumulator saturates — and weakens the
comparison only in the sense that 78 is a lower bound on what the replica can
do; it never approached the pass line.) Integer 512 runs with `lr_bits 2` and
`hidden_bits 24` sat at 0/512 at ep150 and lr2 was stopped as uninformative.

Supervisor bug: `optim_driver.py` read `bisect_r5.json` left by the earlier
64-pair rung-5 run (64/64 < 384) and took branch C, launching a d192 Adam
run. Stopped both; branch A is driven directly. Lesson recorded: bisect
outputs must carry the pair count in the file name.

Branch A launched 20:24: integer 512 with `momentum_bits` 5 and 7 (3 already
running), d96, everything else identical. Best of {3,5,7,hb24} goes to STAGE 4.

### 21:38 — momentum raises accuracy by *removing* source usage; gradient resolution is the live direction

512 pairs, d96, coupled base-2 attention:

| variant | TF content | real | shuffled | gap | zeroed |
|---|---:|---:|---:|---:|---:|
| momentum 3 (ep750) | 48.9% | 51.9 | 49.8 | +2.1 | **74.2** |
| momentum 5 (ep600) | 56.6% | 61.2 | 59.1 | +2.1 | **73.5** |
| momentum 7 (ep450) | 67.2% | 67.0 | 58.1 | +8.9 | **69.3** |
| **hidden_bits 24** (ep750) | 35.6% | 33.5 | **6.3** | **+27.2** | 20.9 |

Momentum works exactly as designed and that is the problem: averaging over 2^k
steps keeps whatever is consistent across batches, and what is consistent is
the target-side language model, not the source-specific signal that appears in
one batch out of N. All three momentum runs score *higher with the encoder
memory zeroed than with the real source*. Selecting on teacher-forced accuracy
would have picked the worst model — the earlier supervisor would have done
exactly that, so its selection rule was wrong too.

`hidden_bits 24` is the only setting with the right ordering (real > zeroed >
shuffled) and a real gap. More adjoint resolution lets the small
source-specific gradient survive quantisation instead of being rounded away.

Momentum runs stopped. Launched 21:38 at 1500 epochs: hidden_bits 28, 32, and
24 (longer budget), 4 threads each. `momentum_bits` stays in the codebase,
gated and off, with this result recorded next to it.

---

## 12. auto.py — autonomous ladder (2026-09-06 21:41:03)

Selection requires source usage: gap > 0 and real > zeroed, else score 0.
