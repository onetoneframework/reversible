"""Held-out translation quality of a trainer checkpoint.

Rebuilds the trainer's deterministic train/validation split with the same data
arguments, greedy-decodes the first N validation rows and reports chrF, BLEU,
exact match and token accuracy, with samples; optional free Japanese inputs.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import reversible_jako_mt as M
import reversible_jako_data as D

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--max-pairs", type=int, required=True)
ap.add_argument("--valid-frac", type=float, default=0.05)
ap.add_argument("--base-pairs", type=int, default=8000)
ap.add_argument("--min-char-count", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--offset", type=int, default=0, help="skip this many validation rows first (independent test rows)")
ap.add_argument("--show", type=int, default=20)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--ja", nargs="*", default=[])
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(a.threads)

model, tok, ck = M.load_ckpt(a.ckpt)
ds = D.prepare(ck.get("corpus") or "both", a.max_pairs, ck.get("vocab_size_arg") or ck["vocab"], ck["max_len"],
               seed=a.seed, valid_frac=a.valid_frac, max_tok=ck.get("max_tok", 0),
               tok_mode=ck.get("tok_mode", "char"), base_pairs=a.base_pairs,
               min_char_count=a.min_char_count)
assert ds["info"]["vocab_size"] == ck["vocab"], (ds["info"]["vocab_size"], ck["vocab"])
hb = int(ck.get("hidden_bits", 16))
valid = tuple(col[a.offset:] for col in ds["valid"]) if a.offset else ds["valid"]
vpairs = ds["valid_pairs"][a.offset:] if a.offset else ds["valid_pairs"]
r = M.evaluate(model, valid, hb, a.batch, a.n, decode=True, tok=tok, pairs=vpairs)
lines = ["checkpoint %s  epoch %s  train pairs %d  valid pairs %d" % (
    Path(a.ckpt).name, ck.get("epoch"), len(ds["train"][0]) if "train" in ds else -1, len(ds["valid"][0])),
    "HELD-OUT (%d rows): chrF %.3f  BLEU %.3f  exact %.2f%%  token_acc %.3f%%  hinge/token %.3f" % (
        len(r["samples"]), r["chrf"], r["bleu"], r["seq_exact_match"], r["token_acc"], r["hinge_per_token"])]
for ja, ref, hyp in r["samples"][:a.show]:
    lines += ["  JA  " + ja, "  REF " + ref, "  GEN " + hyp]
if a.ja:
    lines.append("FREE INPUT")
    for ja, hyp in zip(a.ja, M.translate(model, tok, a.ja, ck["max_len"])):
        lines += ["  JA  " + ja, "  GEN " + str(hyp)]
print("\n".join(lines), flush=True)
if a.out:
    Path(a.out).write_text("\n".join(lines) + "\n\n" + json.dumps(
        {k: v for k, v in r.items() if k != "samples"}, indent=1), encoding="utf-8")
