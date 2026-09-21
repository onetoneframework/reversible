# JA→KO reversible-integer translator — autonomous run journal

Kept by the overnight loop. One section per run, appended as things happen.
Numbers here are measured, never estimated; anything not measured says so.

---

## run1 — d256 / e4 "af" / d8 "ccafccaf" / 50k mixed corpus

> **INVALID FOR CAUSAL TRANSLATION EVALUATION.** Training used an attention
> normalisation whose per-row shift was computed from masked scores: future
> target keys in decoder self-attention and PAD source keys in cross-attention
> both entered `pow2norm`'s row maximum before the mask was applied. Teacher
> forcing therefore had a scale side-channel to information that free-running
> decoding does not have. Every accuracy number below was optimised against
> that leak. Stopped after epoch 7 rather than run to ep11: further epochs
> would only buy more of the same leak.
>
> Kept as a negative control, and as the before-side of the causality fix.
> Artifacts archived under `run1_d256_50k_INVALID/`. **Never resume training
> from `jako_50k_best.pt`.**


Launched 2026-09-06 00:54:24, PID 33604. `_50k.log`, `jako_50k_best.pt`,
translations in `jako_translations.txt`.

```
corpus  mixed 50,000 pairs  (ted2020 23,692 | opensubtitles 23,737 | tatoeba 71)
        valid 2,500        (opensubtitles 1,263 | ted2020 1,234 | tatoeba 3)
        vocab 5000 char-BPE, alphabet capped at chars seen >= 20 times
        acceptance leakage 0/30 (6 base pairs were leaking; blocked)
model   8,265,728 params, core 3.67M
train   batch 32, 14 threads, lr_bits 2, hidden_bits 16, score_leak 3
        epochs 12, eval every epoch, patience 3
```

### epochs

| ep | train hinge | train acc | valid acc | valid hinge | ted | subs | moved | best |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 48.34 | 1.164% | 1.374% | 22.89 | 1.178 | 1.457 | 33.7% | BEST |
| 1 | 26.08 | 1.798% | 1.374% | 32.50 | 1.335 | 1.169 | 37.2% | no-best 1 |
| 2 | 32.54 | 1.730% | 1.559% | 34.35 | 1.571 | 1.569 | 38.7% | BEST |
| 3 | 22.05 | 2.607% | 3.026% | 18.54 | 3.029 | 2.929 | 40.5% | BEST |
| 4 | 20.58 | 2.579% | 1.744% | 24.92 | 2.221 | 1.857 | 41.8% | no-best 1 |
| 5 | 23.04 | 2.303% | 3.026% | 20.83 | 3.007 | 2.993 | 42.9% | no-best 2 |

ep0/ep2/ep3 BEST, ep1/ep4/ep5 no-best. Patience is 3; ep5 is no-best 2.

**Correction to what I wrote at ep4.** I called ep3 "a spike, not a new level".
ep5 reaches valid acc 3.026% again -- the same figure to three decimals -- so
ep3's level is reachable and ep4 (1.744%) was the outlier, not ep3. valid hinge
also recovered, 18.54 -> 24.92 -> 20.83, and the two domains stay balanced
(ted 3.007, subs 2.993).

ep5 is not saved as best only because of the tie-break: equal accuracy with a
worse hinge (20.83 vs ep3's 18.54) loses. So the run is oscillating around ~3%
rather than climbing, and patience will fire at ep6 unless ep6 beats 3.026%.

**The consistent shape is underfitting, not overfitting**: valid accuracy is
above train accuracy in every epoch that matters (ep3 3.026 vs 2.607, ep5 3.026
vs 2.303). Capacity is not the binding constraint -- the optimiser is not moving
enough of the model. 57% of weights have still never changed at epoch 5.

**Prepared change for the next run** (apply when patience fires, one knob only):
`--lr-bits 1`, i.e. `BlockSGD.shift` 13 -> 12, doubling how often a weight takes
its +-1 step. Chose 1 rather than 0 because the run does oscillate and 0 would
be a 4x effective learning rate. Resume from `jako_50k_best.pt` (ep3) rather
than starting cold, and turn on `--probe-file` so every epoch's translations are
recorded, including the ones that do not improve.

### diagnostics taken while it runs

**Weight clamp is not the bottleneck.** Only 0.128% of the 8.27M weights sit at
the 8-bit bound (±128). Three decoder tensors are much hotter than the rest —
indices 30/31/32, which are cross-attn block4 Wv/Wo and block5 Wq — at 3.0-6.9%
clamped with mean |w| ~33-37 against ~8 everywhere else. Worth watching, not
worth acting on at 0.128% overall.

**61% of parameters have never moved.** `weights_moved_since_init_pct` is 38.7%
after three epochs. The update quantum explains it: `BlockSGD.shift` is
`(param_bits-1) + lr_bits` = 11 + 2 = 13, so a weight only changes by ±1 when its
error-feedback accumulator crosses 8192, and `renorm_tensor(gm, 12)` caps the
largest gradient element at ~2048. Four maximum-magnitude steps minimum for one
increment, and typical elements are far below the block max. Measured
`|acc|max` = 8191 on essentially every tensor, i.e. the accumulator is
saturated-but-draining exactly as designed — this is the design's granularity,
not a defect.

**Prepared change, not yet applied:** `--lr-bits 0` drops the shift 13→11, a 4×
finer update quantum. Held back because ep2 improved; applying it now would
confound the run. Apply when patience fires.


### ep6 offline diagnostics (run1 untouched)

**Where the 4.176% comes from** (`reversible_jako_breakdown.py`, 512 valid seqs):
83% of correct tokens are EOS (EOS acc 51.4%); content tokens 0.200% (10 of
4,992); pos0 0.195%, pos8+ 6.8%. EOS false-positive 19.0%, generated length
5.7 vs reference 15.3. Source ablation: real 4.176 / shuffled 4.216 / zeroed
2.207 -- source identity contributes nothing, source presence does.

**Cross-attention** (`reversible_jako_crossdiag.py`, `reversible_jako_querydiag.py`):
- rows are broad, not uniform: H/logT 0.76-0.87, max share 2-3x uniform,
  effective support ~9-10 of ~12 source positions. Hypothesis "uniform
  collapse" rejected in its strong form.
- rows are **query-independent**: concentration statistics identical across
  decoder positions to 3 decimals in dec1/dec4/dec5. Score-row cosine across
  positions 0.83 / 0.999 / 0.988; only dec0 varies (0.42).
- cause is upstream of the scores. The query-side state `a` grows a shared
  component with depth: row-cos across positions 0.31 -> 0.82 -> 0.87 -> 0.95,
  common-mode share 0.55 -> 0.98, mean |a| 8 -> 56. The 7-bit (TB+2) pow2norm
  of the Wq projection then removes what is left (dec4: a-cos 0.872 -> Q-cos
  0.993). Score-stage quantisation is innocent: raw cos == S cos everywhere.
- counterfactuals: replacing rows with uniform or permuted weights changes
  78-92% of argmaxes and rel. logits by 0.3-0.6, yet TF accuracy stays 3.0-4.4%
  (uniform on dec0 alone: 4.416% > real 4.185%). The pattern is a signal the
  decoder is sensitive to and gets nothing from.

Net: every decoder position reads the same per-sentence weighted average of
the source -- a fixed bag-of-source vector. It cannot align a target token to a
source token, which is why shuffling sources costs nothing and pos0 is dead.

**Direction for run2** (not launched; run1 continues to ep11):
target the query path, not the score map, optimiser or data.
  (ii) integer query centering: subtract the per-sequence mean row of `a`
       before Wq inside cross-attention. Exact, inside f(a), reversibility
       untouched. Attacks the measured cause (common-mode share 0.93-0.98).
  (i)  raise projection resolution TB+2 = 7 -> ~10 bits as the second knob.
Also change best-checkpoint selection: content-token accuracy (EOS excluded)
plus a free-running length/diversity check, since the current metric is 83%
EOS and prefers early-stopping models (ep3 -> ep6 regression in generation).
