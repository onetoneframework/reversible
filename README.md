# Reversible integer training

Neural networks trained end to end in **exact integer arithmetic**, with a
**bit-exact inverse** and a **cache-free backward pass**: activations are
reconstructed from the layer above instead of being stored, so activation memory
is constant in reversible depth. No float in the production path — no softmax, no
exponential, no division that is not a floor.

Three problems are covered: sklearn digits, Fashion-MNIST (convolution and
attention), and a Japanese → Korean sequence-to-sequence translator.

## What this is, and what it is not

The machinery works and is verified to the bit. The translator does not translate
well.

- **Verified:** exact invertibility, cache-free backward equal to the cached
  backward on every gradient mantissa *and* exponent, per-layer stepping equal to
  batched stepping, depth-independent activation memory, and an integer attention
  adjoint whose cosine against the float coupled Jacobian is 0.99392.
- **Not reached:** competitive translation. On the 67k-pair JA→KO split the float
  mirror tops out at held-out greedy chrF ≈ 11 and the best integer run reaches
  7.86 (checkpoint-averaged). The architecture and the data bound both; the
  remaining integer-specific gap is attributed below and in
  [`generalize/README.md`](generalize/README.md).

The value here is the verification harness and the attribution, not the
translation quality.

**Gates 2–5 were re-run from this checkout** on Python 3.13.7 with torch
2.11.0+cpu, numpy 2.3.3 and tokenizers 0.22.2, and all pass: encoder and decoder
inverse exact; 19 tensors compared with every mantissa *and* every exponent equal
between the cache-free and cached backward; 8 optimizer steps with no divergence
between per-layer and batched stepping; attention adjoint cosine 1.0000 at 12 and
20 adjoint bits and 0.9998 at 6.

```
python reversible_jako_verify.py --all --threads 4
```

## Requirements

| | |
|---|---|
| Python | 3.13 (3.13.7 is what these runs used) |
| Packages | `torch` (a CPU wheel is enough for the integer runs; a CUDA wheel in a separate venv for `--device cuda` float runs), `numpy`, `tokenizers` |
| Optional | `scikit-learn` for the digits experiments |
| Hardware | Every number below was measured on a laptop: Core Ultra 5 135H, 14C/18T, 16 GB, CPU only |

Integer training is single-process and thread-pinned. Oversubscription has cost
more than it bought on every machine tried here; the default is `THREADS=4`, with
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` set to match.

## Quick start

On Windows, `run.bat` is the single entry point. It locates a real Python, forces
UTF-8 so Japanese and Korean print, and pins threads.

```
run.bat help        what each command does
run.bat gates       all verification gates          ~2 min
run.bat memory      depth-independent memory        ~1 min
run.bat demo        short training, then translate  ~5 min
run.bat train       overfit 512 real Tatoeba pairs  ~16 min
run.bat train-ted   real training on TED2020        ~32 min
run.bat translate   translate with the saved model
run.bat translate "<japanese text>"
run.bat digits      digits gate matrix
run.bat fashion     Fashion-MNIST conv / attention
run.bat all         gates + memory + demo
```

Override the thread count with `set THREADS=8`. From bash, `train_jako.sh` runs
the same sequence. The corpora are downloaded on first use; only the trained
tokenizers are cached here.

The larger experiments launch through `generalize/run_jobs.ps1`, which detaches
each job and writes `runs/<job>/<job>.log|.json|.pt`:

```powershell
cd generalize
.\run_jobs.ps1 -List
.\run_jobs.ps1 -Job int-d192-split -Threads 10            # integer, CPU
.\run_jobs.ps1 -Job float-ceiling-d384-long -Device cuda  # float, GPU
```

Score any integer checkpoint on the held-out split with the float scripts' own
metric:

```powershell
python jako_eval_ckpt.py --ckpt generalize/runs/int-d192-split/int-d192-split_best.pt --max-pairs 0 --base-pairs 0 --n 400
```

## How it works

State is `[a, b]` over `(B, T, 2·d)`. Every block is an additive coupling

```
[a, b]  ->  [a, b + F(a)]
```

followed by a half swap. `F` may be attention, an FFN or a convolution without
affecting invertibility: for any deterministic integer `F`, subtracting `F(a)`
recovers `b` exactly. The backward pass inverts the coupling to reconstruct the
inputs it needs, so no activation tape is kept.

- encoder: `attn, ffn, …` (`k_enc` blocks)
- decoder: `self-attn, cross-attn, ffn, …` (`k_dec` blocks)
- attention is single-head with a power-of-two rescale; cross-attention reads the
  encoder memory, whose width is `2·d`, so its K/V projections take `2·d` inputs
  while Q takes `d`
- decoding is greedy argmax over logits

Embedding tables and the output projection are **not** reversible and their
activations are stored. That cost is O(1) in reversible depth and is reported as
a separate line item; it is never folded into "activation memory". Total RAM is
not O(1) — parameters, optimizer state, encoder memory and token buffers all
scale normally.

### Integer semantics

These names are easy to misread, so they are stated exactly:

| knob | meaning |
|---|---|
| `weight_bits = 8` | **signed.** Weights clamped to `[-2^7, 2^7-1]`, initialised uniform in `[-16, 16]` |
| `state_bits = 16` | **signed two's-complement wrap** width of the reversible state, values in `[-2^15, 2^15)`. Wrap is exactly invertible, so reversibility survives it |
| `accum_bits = 32` | **signed saturating** width of the GEMM accumulators |
| `hidden_bits = 16` | **magnitude** bits of the backward adjoint mantissa, carried alongside a separate sign; the discarded scale lives in a per-tensor exponent |

`hidden_bits` has a floor *and* a ceiling. Too few bits and the adjoint collapses
to zeros; above roughly 24 bits the mantissa is left-shifted far enough that the
adjoint GEMMs overflow int64. Measured usable window on this model: **10–24
bits**, cosine 1.0000 against a matched-forward float oracle throughout.

### The optimizer, and its stability condition

`BlockSGD` is an integer sign-step optimizer whose error-feedback accumulator
gains up to `2**param_bits − 1` per step and drains `2**shift` when it fires.
With `shift = (param_bits−1) + lr_bits` the accumulator is bounded only when

```
2**shift > 2**param_bits − 1      i.e.   lr_bits >= 1
```

At `lr_bits = 0` the clip engages and the residual grows without bound. Measured
on a 32-way lookup at `param_bits = 12`:

| lr_bits | drain | observed \|acc\|max | 32-way accuracy |
|---:|---:|---:|---|
| 0 | 2,048 | 2,366–5,260 (unbounded) | 6–25%, never converges |
| 1 | 4,096 | exactly 4,095 | **100% by epoch 400** |
| 2 | 8,192 | exactly 8,191 | **100% by epoch 400** |
| 3 | 16,384 | ≤ 16,382 | **100% by epoch 400** |

Every run made before this was found had used `lr_bits = 0`, which invalidates
their conclusions. The condition is in `BlockSGD`'s docstring and exposed as
`self.stable`.

## Established results

- Additive coupling `(a,b) -> (a, b+f(a))` is bit-exact invertible for any
  deterministic integer `f` — verified for MLP, convolution, attention and the
  full seq2seq encoder/decoder.
- Cache-free reconstructed backward equals cached backward **bit for bit**, on
  every parameter gradient mantissa and exponent, and on the input gradient.
- Per-layer optimizer stepping equals batched stepping exactly.
- Cache-free activation memory is constant in reversible depth; the cached tape
  grows linearly.
- Attention requires softmax's **backward** row coupling, not its forward shape.
  With `w_i = 2**(((s_i − max_j s_j) >> k) + cap)` as the forward and

  ```
  dL/ds_i = w_i (g_i − gbar),   gbar = floor( Σ_j w_j g_j / Σ_j w_j )
  ```

  as the adjoint — one int64 dot product and one floor division per row, no
  exponential — greedy reproduction went from 0/64 to 32/32, 53/64 and 111/128.
  Against the float coupled Jacobian the cosine is 0.99392; against the diagonal
  Jacobian, 0.65094.
- The per-row shift must be computed **after** masking. Computed before, the
  maximum that sets the scale includes masked-out entries — future target keys in
  decoder self-attention, PAD source keys in cross-attention — and teacher
  forcing gains a side channel that free-running decoding does not have.
  `masked_pow2norm` fixes it; three gates in `reversible_jako_causalgate.py` fail
  on the old code with 28–30 leaked tensors.
- Learned integer positional tables were also required: at 32 pairs, 0/32 without
  them and 32/32 with them, everything else equal.
- Memorisation scale: 472 of 512 pairs reproduced with `--momentum-bits 7`
  (`blocksgd_scale/`).

## Where the integer model loses to float, and why

Same architecture, same data, one constraint changed at a time. Float side
(`generalize/float_ablation.py`, held-out token accuracy on the 67k split, d192):

| objective / optimizer | ep0 | ep1 | ep3 | best (epoch) | greedy chrF |
|---|---|---|---|---|---|
| CE + softmax + Adam (ceiling) | 23.3 | 26.8 | 30.0 | 33.1 (10) | 11.3 |
| hinge + softmax + Adam | 15.5 | 16.6 | – | 16.6 | 5.3 |
| hinge + exp2 + Adam | 15.2 | 16.1 | 17.1 | 17.1 | 5.2 |
| Weston-Watkins hinge + softmax + Adam | 11.9 | 16.8 | – | 16.8 | 5.2 |
| **base-2 CE, exact Z, k=0 + exp2 + Adam** | 20.4 | 23.4 | 26.1 | **28.7 (9)** | **9.8** |
| base-2 CE, k=3 | 13.6 | 15.1 | – | 15.1 | 4.0 |
| base-2 CE, power-of-two Z | 0.06 | – | – | collapses | 0.1 |
| any + BlockSGD replica | NaN | – | – | float overflow | – |

The hinge objective costs about 13 points, the power-of-two attention costs
nothing, and a base-2 cross-entropy with an **exact** per-row normaliser recovers
most of it. The normaliser has to be exact: replacing `Z` with the power of two
above it leaves a residual push of up to −0.5 on every confidently-correct target
and training collapses.

Integer side, the same head ported, one change at a time (cells are all-token % /
content %):

| integer d192, BlockSGD split momentum unless noted | ep0 | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| hinge head (baseline) | 12.2 / 3.06 | 12.7 / 4.07 | 13.6 / 3.84 | 13.6 / 4.55 |
| base-2 CE head, exact Z | 12.3 / 3.55 | 13.8 / 3.95 | 13.7 / 4.52 | 12.7 / 3.72 |
| base-2 CE head, power-of-two Z | 10.5 / 0.49 | 12.5 / 0.00 | – | collapses |
| CE head + IntegerMasterSGD η 2⁻⁷ | 3.2 / 0.06 | 2.2 / 0.06 | – | – |
| CE head + 9-bit activations | 9.1 / 0.60 | 10.9 / 1.23 | – | – |
| **CE head + 10-bit weights** | 12.5 / 3.01 | 14.8 / 4.58 | 14.9 / 5.24 | 15.1 / 5.61 |
| CE head + 12-bit weights | 14.3 / 4.29 | 15.7 / 5.67 | 15.8 / 6.04 | 15.3 / 5.32 |
| CE head + 10-bit weights, uniform momentum | 12.3 / 2.89 | 11.4 / 3.41 | – | – |

The objective that moves the float mirror from 17 to 29 points moves the integer
model by about one point. **The binding constraint at this scale is the int8
weight range under the sign-step optimizer's ramp:** every int8 run logs
`max_weight: 128` from epoch 0 — the weights sit on the ±127 clamp and the
optimizer has nowhere left to move them. The 10-bit run reaches its ±511 clamp at
epoch 4 and keeps improving, reaching 26.0% all-token / 14.5% content by epoch
17. What remains between 10-bit integer and the float mirror with the same
objective (~29%) is the sign-step optimizer itself.

Averaging neighbouring epoch checkpoints removes the ±1 sawtooth for free
(`avg_ckpts.py`, integer mean, ties away from zero):

| checkpoint | held-out chrF | BLEU | token acc |
|---|---|---|---|
| ep17 alone | 7.08 | 6.83 | 26.0% |
| mean of ep16–18 | **7.86** | **7.66** | 27.3% |
| mean of ep14–18 | 7.39 | 7.21 | **27.8%** |
| mean of ep14–25 (spans a collapse) | 5.10 | 4.29 | 23.1% |

A local window around the best checkpoint helps; one that spans the epoch-19
collapse mixes basins and hurts.

## Negative results worth keeping

**Float oracles must reuse the integer forward's own decisions.** Comparing the
integer backward against a float mirror built on unquantised scores, or with
plain division where the integer path floors, produces a wrong oracle. That
mistake twice produced a false "formula error" verdict here. The mirror should
assert that it reproduces the integer forward exactly before its gradients are
trusted.

**Finite-difference sign agreement does not predict trainability** in this
discrete system. The MLP that trains to 89% scores only 0.54–0.69; the
convolution that trains to 49% scores 0.436; attention produces too few nonzero
finite differences to be informative. Use the matched-forward autograd cosine
instead.

**Do not select checkpoints on accuracy alone.** Integer momentum once raised
teacher-forced content accuracy to 67% at 512 pairs while scoring *higher with
the encoder memory zeroed* than with the real source — it had learned a
target-side language model. The trainer now requires
`content_acc(real) > content_acc(shuffled)` and `real > zeroed`.

**Lowering the step rate late loses source conditioning.** Continuing the best
run at `--lr-bits 2` dropped the source gap from +6.3 to 0.0 within one epoch and
kept falling; the source pathway runs without momentum and needs the full step
rate.

## Repository layout

**Core integer machinery** — the source of truth, reused by everything else

| file | role |
|---|---|
| `reversible_integer_final.py` | coupling, `wrap_state`, `saturate`, `norm_block`, `SharedGrad`, `dense_linear_backward`, `BlockSGD`, width and grouping options |
| `reversible_integer_memory_bench.py` | live-storage tracker, cached vs cache-free memory |
| `int8_reversible.py` | the original NumPy derivation, kept for auditability |
| `integer_master_sgd.py` | sub-grid master-weight optimizer variant |
| `grad_exactness.py` | gradient exactness checks |
| `split_equivalence_gate.py` | per-layer vs batched equivalence gate |
| `avg_ckpts.py` | integer checkpoint averaging |

**Japanese → Korean translation**

| file | role |
|---|---|
| `reversible_jako_mt.py` | model, integer attention/FFN, manual cache-free backward, train/translate |
| `reversible_jako_data.py` | OPUS corpus download, cleaning, byte-level BPE |
| `reversible_jako_verify.py` | gates 2–5 |
| `jako_train_split.py` | the 67k split trainer, `--head ce2` and the weight-width options |
| `jako_eval_ckpt.py` | score a checkpoint with the float scripts' held-out metric |
| `reversible_jako_*.py` | diagnostics: accept, bisect, breakdown, causalgate, crossdiag, decode, diag, floatdiag, headgate, leakhunt, lookup, metrics, oracle, overfit32, probe, querydiag, translate |
| `train_jako.sh`, `run.bat` | the exact run sequences |
| `README_reversible_jako.md` | architecture, integer semantics, dataset, limitations |

**Digits experiments**

| file | role |
|---|---|
| `reversible_integer_golden_capture.py` | 34-entry behavioural fingerprint proving the option defaults are bit-exact |
| `reversible_integer_gate_matrix.py` | equivalence gates across width and grouping options |
| `reversible_integer_structural.py` | grouped-GEMM equivalence, permutation dispersion, 1/g ratios |
| `reversible_integer_crosscheck.py` | NumPy vs PyTorch, 5 and 10 seeds |

**Fashion-MNIST: convolution and attention**

| file | role |
|---|---|
| `reversible_arch.py` | conv and attention coupling blocks, `exp2_attend`, `exp2_attend_backward`, verify/train/memory |
| `reversible_attn_diag.py` | stage-by-stage attention backward tracer |
| `reversible_fd_control.py` | finite-difference control (a negative result, see above) |
| `fashion_mnist_data.py`, `fashion_mnist_run.py`, `fashion_mnist_sweep.py`, `fashion_mnist_projection.py` | data, runs, sweeps, projection comparison |

**Directories**

| directory | contents |
|---|---|
| `generalize/` | the 67k-split stage: float ceiling, the float ablation ladder, the NPU matmul probe, `run_jobs.ps1`, and `runs_laptop/` with every log, JSON and decoded sample |
| `blocksgd_scale/` | the 512-pair memorisation study, split vs uniform momentum, weight-width sweeps, decoded translations |
| `float_adam_8k_control_20260906_results/` | the float + Adam held-out control on the 8k split — audit, split, history and `FINAL_REPORT.md` |
| `float_adam_8k_trainfit_20260906_results/` | the train-fit companion to that control |
| `integer_matched_140/` | the matched integer run at 140 dimensions, with gates and history |
| `paper/` | manuscript and results compendium (`paper_revised.html`, `compendium.html`), the figures as SVG, and the builders that generate every table directly from the run files |
| `_jako_data/` | cached byte-level BPE tokenizers — vocabulary and merges only, **no corpus text** |
| `_fashion_mnist/` | the Fashion-MNIST idx files |

**Records**

`FINAL_SUMMARY.md` is the narrative of the translation investigation up to
2026-09-06: the causality bug, the optimizer stability condition, the attention
adjoint bisection, and what was still failing. It predates the `generalize/`
stage, so where the two disagree, `generalize/README.md` is newer.
`overnight_translation_report.md` and `jako_run_journal.md` are the per-run logs
as they were written.

## Data and licences

Corpora are downloaded on first use; this repository caches only the trained
tokenizers.

| corpus | source | licence |
|---|---|---|
| OPUS Tatoeba v2023-04-12 ja-ko | `object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/ja-ko.txt.zip` | CC-BY 2.0 FR |
| OPUS TED2020 v1 ja-ko | `object.pouta.csc.fi/OPUS-TED2020/v1/moses/ja-ko.txt.zip` | CC BY-NC-ND 4.0 (TED) |
| OpenSubtitles v2024 ja-ko | OPUS | per OPUS terms |
| Fashion-MNIST | Zalando Research, included under `_fashion_mnist/` | MIT |

The SHA-256 of each downloaded archive is recorded in the results JSON. Cleaning
drops empty lines, length outliers, script mismatches — the Korean side must
contain Hangul, the Japanese side kana or CJK — and exact duplicates. Acceptance
sentences are blocked from training; an earlier leak of 6 of 30 was found and
closed.

TED2020 is CC BY-NC-ND. What is published here is research output and a
tokenizer vocabulary, not a redistribution of that corpus.

## Known limitations

- Translation quality is poor and not competitive with any real MT system. The
  model learns Korean output, Korean morphology and frequent vocabulary, then
  collapses onto frequent prefixes and repeats tokens.
- Greedy decoding re-runs the whole decoder per timestep; there is no KV cache.
- Single head, no layer norm.
- The integer optimizer emits at most ±1 per weight per step, which bounds how
  fast the model can move.
- Total RAM is not O(1); only reversible-depth activation memory is.
- The corpus was capped at 8,000 pairs for the historical runs to match an
  earlier configuration. 187,725 pairs are available at the same length limit and
  209,979 at full length — the cap was self-inflicted, and the later 67k runs
  lifted it only partway.
- `reversible_jako_floatdiag.py` has the held-out split and metrics wired, but one
  helper reference was left unresolved when work stopped.
- Recorded output paths in the run JSON files were rewritten to `<local>/…`
  before publication; measured values are untouched.

## What to do next

1. Raise the weight width past 10 bits **and** fix the optimizer ramp together.
   The two are entangled: 12-bit alone was slower per epoch without being better,
   because each ±1 step is four times smaller.
2. Train at the full 210k scale rather than the 67k short-sentence split. The
   float ceiling of chrF 11 is a property of this data slice, not of the method.
3. Never measure an integer change without a matched float control at the same
   epoch. Every conclusion here that lacked one had to be retracted.

## Licence

Research code, published as-is. The corpus licences in the table above are not
the same as the code's and govern the data independently.

Author: Jeong Seong Yeol. Extracted from the Onetone Framework at
`res/Platform/Python/app/reversible`.
