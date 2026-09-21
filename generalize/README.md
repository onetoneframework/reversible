# generalize/ — integer BlockSGD vs its float ceiling on the full corpus

Goal of this stage: integer BlockSGD trained on the same data as a float mirror
should approach the float mirror's held-out greedy chrF. Everything here runs
from a plain checkout; no path in these scripts is absolute.

## What to copy to another machine

1. `res/Platform/Python/reversible/` (this directory's parent; a git checkout is fine).
2. `res/Platform/Python/reversible/_jako_data/` — the corpus zips
   (`ted2020_ja-ko.zip`, `opensubtitles_ja-ko.zip`, tatoeba) and the cached
   tokenizers (`tok_both_a20_5000_char.json` is the one every run here uses).
   ~61 MB. Without it the data module tries to download the OPUS zips.
3. Python 3.13 with `torch` (CPU wheel for the integer runs; a CUDA wheel in a
   separate venv for `--device cuda` float runs), `numpy`, `tokenizers`.

## Launching

```powershell
cd res\Platform\Python\reversible\generalize
.\run_jobs.ps1 -List
.\run_jobs.ps1 -Job int-d192-split -Threads 10            # integer, CPU
.\run_jobs.ps1 -Job float-ceiling-d384-long -Device cuda  # float, GPU
.\run_jobs.ps1 -Job float-hinge-exp2-adam -Threads 4
```

Each job writes `runs\<job>\<job>.log|.json|.pt` and detaches; `-Epochs N`
overrides the default budget, `-Python path\to\python.exe` picks the
interpreter. Per-epoch validation lines look like

```
  ep3   all 13.580% content  4.5520% ... | cont r/s/z  4.799/ 4.018/ 4.241 gap +0.7812 | pos0 ...   (integer trainer)
  ep3   train_ce 3.565 | valid tok 30.04% | greedy chrF 9.79 BLEU 10.26 exact 0/400 | 29.6m         (float scripts)
```

Score an integer checkpoint on the held-out split with the float scripts' metric:

```powershell
python ..\jako_eval_ckpt.py --ckpt runs\int-d192-split\int-d192-split_best.pt --max-pairs 0 --base-pairs 0 --n 400
```

## Numbers so far (laptop: Core Ultra 5 135H, 14C/18T, 16 GB, CPU only)

Data: `--max-pairs 0 --base-pairs 0 --max-tok 14 --max-len 16` -> 66,853 train /
3,518 validation pairs (the historical "8k" runs used `--base-pairs 8000`, i.e.
7,600 of them). Full length (`--max-tok 32 --max-len 40`) -> 209,979 / 11,051.

| run | epochs | held-out greedy chrF (400 rows) | valid token acc | notes |
|---|---|---|---|---|
| float Adam CE d96 | 10 | 10.4 | 31% | 5.4 min/epoch |
| float Adam CE d192 | 10 | 11.3 | 33% | train CE 2.85, overfitting starts |
| float Adam CE d192, full-length 210k | 0 | 6.7 | 23% | 49 min/epoch, stopped |
| float hinge softmax Adam d192 | 0 | 5.5 | 15% | hinge costs vs CE at ep0 |
| float hinge exp2 Adam d192 | 0 | 4.6 | 15% | exp2 attention ~= softmax at ep0 |
| float hinge softmax BlockSGD-replica d192 | 0 | NaN | - | float softmax overflow, as historical rung 8 |
| integer mom7 d96 | 8 | - | 11-12% (content 3.6% max) | plateau |
| integer split m7/s0 d96 | 4 | - | 12% (content 3.2% max) | plateau |
| integer split lr_bits 3 d96 | 3 | - | 11% (content 2.1%) | slower, not better |
| integer split m7/s0 d192 | 3 | 4.2 (ep2) | 13.6% (content 4.6% at ep3, rising) | 20 min/epoch |

Reading: with the same architecture and data, float Adam tops out near chrF
11 on this short-sentence split, so the architecture/data — not only the
integer optimizer — bounds generalisation here; the integer model sits at
roughly a third of the float mirror's token accuracy and is still rising at
d192. The float ablation ladder (hinge vs CE, exp2 vs softmax, BlockSGD
replica) is what attributes the remaining gap.

## Attribution of the integer gap (d192, 67k split, held-out token accuracy)

Float side, one constraint at a time (`float_ablation.py`, exp2 attention unless noted):

| objective / optimizer | ep0 | ep1 | ep3 | best (epoch) | greedy chrF |
|---|---|---|---|---|---|
| CE + softmax + Adam (ceiling) | 23.3 | 26.8 | 30.0 | 33.1 (10) | 11.3 |
| hinge + softmax + Adam (F1) | 15.5 | 16.6 | - | 16.6 | 5.3 |
| hinge + exp2 + Adam (F2) | 15.2 | 16.1 | 17.1 | 17.1 | 5.2 |
| Weston-Watkins hinge + softmax + Adam (F4) | 11.9 | 16.8 | - | 16.8 | 5.2 |
| **base-2 CE, exact Z, k=0 + exp2 + Adam (F5)** | 20.4 | 23.4 | 26.1 | **28.7 (9)** | **9.8** |
| base-2 CE, k=3 (F6) | 13.6 | 15.1 | - | 15.1 | 4.0 |
| base-2 CE, power-of-two Z (F7) | 0.06 | - | - | collapses | 0.1 |
| any + BlockSGD replica (F3, F8) | NaN | - | - | float overflow | - |

So on the float side the hinge objective costs ~13 points and the exp2
attention costs nothing; a base-2 cross-entropy with an exact per-row
normaliser recovers most of it. The normaliser has to be exact: replacing Z
by the power of two above it leaves a residual push of up to -0.5 on every
confidently-correct target and training collapses.

Integer side, same head ported (`../jako_train_split.py --head ce2`: p_i =
2**(l_i - max) for classes within 20 of the max, r = 2**60 // Z once per row,
p_i/Z as 24-bit fixed point, everything else shifts), one change at a time:

| integer d192, BlockSGD split momentum unless noted | ep0 | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| hinge head (baseline) | 12.2 / 3.06 | 12.7 / 4.07 | 13.6 / 3.84 | 13.6 / 4.55 |
| base-2 CE head, exact Z | 12.3 / 3.55 | 13.8 / 3.95 | 13.7 / 4.52 | 12.7 / 3.72 |
| base-2 CE head, power-of-two Z | 10.5 / 0.49 | 12.5 / 0.00 | - | collapses |
| CE head + IntegerMasterSGD eta 2^-7 (no sign steps) | 3.2 / 0.06 | 2.2 / 0.06 | - | - |
| CE head + IntegerMasterSGD eta 2^-5 | 3.8 / 0.11 | 5.3 / 0.17 | - | - |
| CE head + 9-bit activations (TB 7, scores pinned) | 9.1 / 0.60 | 10.9 / 1.23 | - | - |
| **CE head + 10-bit weights** | 12.5 / 3.01 | 14.8 / 4.58 | 14.9 / 5.24 | 15.1 / 5.61 |
| CE head + 12-bit weights | 14.3 / 4.29 | 15.7 / 5.67 | 15.8 / 6.04 | 15.3 / 5.32 |
| CE head + 10-bit weights, uniform momentum 7 (no split) | 12.3 / 2.89 | 11.4 / 3.41 | - | - |

(cells: all-token % / content %). The 10-bit run kept climbing after the
table's window: ep5 18.8 / 7.16, ep7 19.2 / 8.47, ep10 20.6 / 9.96, ep12
**21.6 / 10.91**, ep13 22.5 / 9.56 (still rising in all-token accuracy when
paused; source gap +3 to +5, pos0 up to 13%). Its best checkpoint
(`runs_laptop/g67k_d192_w10_ce2_best.pt`, epoch 12; `_last.pt` for
`--resume`) scores held-out greedy chrF 5.85 at epoch 7 against the float
mirror's 9.4 at the same epoch with the same objective (F5) and the CE ceiling's
11.3. The 12-bit run is slower per epoch (each +-1 step is 4x smaller) and was
behind 10-bit at every epoch; the uniform-momentum control was behind the split
at epochs 0-1.

Why 10 bits helps: every int8 run logs `max_weight: 128` from epoch 0 - under
BlockSGD's error-feedback ramp the weights sit on the +-127 clamp and the
optimizer has nowhere left to move them. The 10-bit run reaches the +-511
clamp at epoch 4 and still improves; the 12-bit run never reaches its clamp
(max 734 of 2047), with zero accumulator saturations and no state wrap in
either. So the objective that moves the float mirror from 17 to 29 points
moves the integer model by about one point, finer activations make it worse,
the sub-grid master SGD does not learn at these learning rates, and the
binding constraint at this scale is the int8 weight range under the sign-step
optimizer's ramp. What remains between 10-bit integer (~22%) and the float
mirror with the same objective (~29%) is the sign-step optimizer itself.

### Continuing the 10-bit run, and averaging its checkpoints

Resumed from `g67k_d192_w10_ce2_last.pt` (epoch 13) with the same arguments
(`runs_laptop/g67k_d192_w10_ce2_r2*`; `--resume` restores the weights and the
target-side accumulators, the momentum buffers refill within ~128 steps). It kept
climbing for four epochs and then oscillated around the plateau until patience
stopped it:

| overall epoch | 14 | 15 | 16 | 17 | 18 | 19 | 20-25 |
|---|---|---|---|---|---|---|---|
| all-token % | 24.6 | 25.4 | 23.2 | **26.0** | 26.1 | 22.9 | 22.1-23.2 |
| content % | 12.7 | 13.2 | 10.4 | **14.5** | 14.2 | 9.5 | 8.6-10.8 |
| source gap | +6.5 | +7.3 | +4.0 | +6.3 | +7.6 | +1.9 | +1.5-3.6 |

Held-out greedy chrF of the single best checkpoints: 5.85 (ep7) -> 6.60 (ep12)
-> **7.08 (ep17**, token acc 26.0%, the first exact matches: 2/400).

The plateau is a +-1 sawtooth of the sign-step optimizer, so averaging the
weights of neighbouring epoch checkpoints (integer mean, ties away from zero)
removes the jitter for free:

| checkpoint | held-out chrF | BLEU | token acc | hinge/token |
|---|---|---|---|---|
| ep17 alone | 7.08 | 6.83 | 26.0% | 4.32 |
| mean of ep16-18 (3) | **7.86** | **7.66** | 27.3% | 3.94 |
| mean of ep14-18 (5) | 7.39 | 7.21 | **27.8%** | 3.52 |
| mean of ep14-25 (7, across the dip) | 5.10 | 4.29 | 23.1% | 3.44 |

Local windows around the best help (+0.3 to +0.8 chrF); a window that spans the
ep19 collapse mixes basins and hurts. Halving the step rate instead
(`--lr-bits 2` continued from ep17) lost the source conditioning within one
epoch (gap +6.3 -> 0.0, pos0 13% -> 3%) and kept falling - the source pathway,
which runs without momentum, needs the full step rate. The next continuation
(`_r4swa`) restarts from the ep16-18 mean at lr_bits 1 with a snapshot every
epoch, to average again.

The memorisation-scale results (512 pairs: 472/512 with `--momentum-bits 7`,
split momentum mechanism) live in `../blocksgd_scale/`.
