"""Greedy translation test for a memorisation checkpoint.

Loads a checkpoint saved by reversible_jako_overfit32.py or blocksgd_wbits_runner.py
(both store {"weights": [...]} in parameters() order), rebuilds the historical
d96/ccaf/positional-16/exp2-coupled model, and greedy-decodes:
  * the training pairs it was fitted on (exact-match count, samples),
  * 64 held-out pairs from the same corpus that were never trained on,
  * any extra Japanese sentences given with --ja.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import reversible_integer_final as R
import reversible_arch as A
import reversible_jako_mt as M
import reversible_jako_data as D
import reversible_jako_overfit32 as O

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--pairs", type=int, required=True, help="number of training pairs the checkpoint was fitted on")
ap.add_argument("--show", type=int, default=24)
ap.add_argument("--heldout", type=int, default=64)
ap.add_argument("--ja", nargs="*", default=[])
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default=None)
a = ap.parse_args()

torch.set_num_threads(a.threads)
ck = torch.load(a.ckpt, weights_only=False)
args = ck.get("args", {})
A.SCORE_LEAK, A.SCORE_EXP2, A.SCORE_EXP2_CAP = 3, 3, 12
A.SCORE_EXP2_COUPLED = bool(ck.get("exp2_coupled", True))
M.CROSS_QUERY_ANCHOR = False
M.ATTN_ROW_MASS_BITS = int(ck.get("attn_mass_bits", 0))
M.ATTN_SCORE_BITS = int(ck.get("attn_score_bits", 0))
wb = int(args.get("weight_bits", ck.get("weight_bits", 8)))
tb = int(args.get("tb", 5))
R.TB = M.TB = A.TB = tb
cfg = R.RunConfig(state_bits=16, weight_bits=wb, accum_bits=32)
R.CFG = cfg
MAX_LEN = 16
seed = int(args.get("seed", 0))
ds = D.prepare("both", 0, 5000, MAX_LEN, seed=seed, max_tok=14, tok_mode="char",
               base_pairs=8000, min_char_count=20)
tok = ds["tokenizer"]
model = M.JaKoMT(ds["info"]["vocab_size"], 96, 192, 2, 4, cfg, seed, enc_pattern="", dec_pattern="ccaf", positional=16)
assert len(ck["weights"]) == len(model.parameters()), (len(ck["weights"]), len(model.parameters()))
for w, s in zip(model.parameters(), ck["weights"]):
    w.copy_(s)

allp = O.pick_pairs(ds, tok, a.pairs + a.heldout, 1.0, MAX_LEN)
train, held = allp[:a.pairs], allp[a.pairs:]


def translate(pairs):
    src, _tin, _tout = O.encode(tok, pairs, MAX_LEN)
    gen = O.DEC.decode(model, src, MAX_LEN, "greedy", 0, False, 1, 1)
    return [M.ids_to_text(tok, gen[i]) for i in range(gen.shape[0])]


lines = []


def emit(s=""):
    print(s, flush=True)
    lines.append(s)


emit("checkpoint %s  epoch %s  (saved greedy %s, tf_content %s)" % (
    Path(a.ckpt).name, ck.get("epoch", "?"), ck.get("greedy_exact", "?"), ck.get("tf_content_acc", "?")))
emit("weight_bits %d  TB %d  momentum %s / source-momentum %s" % (
    wb, tb, args.get("momentum_bits", ck.get("momentum_bits", "?")), args.get("source_momentum_bits", "-")))

hyp = translate(train)
refs = [ko for _ja, ko in train]
exact = sum(1 for h, r in zip(hyp, refs) if h.strip() == r.strip())
emit("\n=== TRAINING PAIRS: greedy exact %d/%d   chrF %.3f   BLEU %.3f ===" % (
    exact, len(train), M.chrf(hyp, refs), M.bleu(hyp, refs)))
for i in range(min(a.show, len(train))):
    ok = "OK " if hyp[i].strip() == refs[i].strip() else "NG "
    emit("%s[%3d] JA  %s" % (ok, i, train[i][0]))
    emit("      REF %s" % refs[i])
    if ok == "NG ":
        emit("      GEN %s" % hyp[i])

if held:
    hh = translate(held)
    hr = [ko for _ja, ko in held]
    ex = sum(1 for h, r in zip(hh, hr) if h.strip() == r.strip())
    emit("\n=== HELD-OUT (never trained, same corpus): greedy exact %d/%d   chrF %.3f   BLEU %.3f ===" % (
        ex, len(held), M.chrf(hh, hr), M.bleu(hh, hr)))
    for i in range(min(16, len(held))):
        emit("     JA  %s" % held[i][0])
        emit("     REF %s" % hr[i])
        emit("     GEN %s" % hh[i])

if a.ja:
    emit("\n=== FREE INPUT ===")
    out = translate([(j, "") for j in a.ja])
    for j, o in zip(a.ja, out):
        emit("     JA  %s" % j)
        emit("     GEN %s" % o)

if a.out:
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
