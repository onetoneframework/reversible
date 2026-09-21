"""Realised BlockSGD displacement per epoch, from init, for each arm configuration.

Runs the canonical BlockSGD for 3 epochs and counts, per parameter group, how
many weights actually moved (|q| = 1) in each epoch, plus the hinge on a fixed
128-example slice before and after each epoch (closed-loop effect).
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import reversible_integer_final as R
import reversible_arch as A
import reversible_jako_mt as M
import reversible_jako_data as D
import reversible_jako_overfit32 as O
from reversible_integer_final import BlockSGD

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

torch.set_num_threads(4)
A.SCORE_LEAK, A.SCORE_EXP2, A.SCORE_EXP2_CAP, A.SCORE_EXP2_COUPLED = 3, 3, 12, True
M.CROSS_QUERY_ANCHOR, M.ATTN_ROW_MASS_BITS, M.ATTN_SCORE_BITS = False, 0, 0
cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
R.CFG = cfg
MAX_LEN, SEED, HB = 16, 0, 16
ds = D.prepare("both", 0, 5000, MAX_LEN, seed=SEED, max_tok=14, tok_mode="char",
               base_pairs=8000, min_char_count=20)
tok = ds["tokenizer"]
VOCAB = ds["info"]["vocab_size"]
SETS = {128: O.encode(tok, O.pick_pairs(ds, tok, 128, 1.0, MAX_LEN), MAX_LEN),
        512: O.encode(tok, O.pick_pairs(ds, tok, 512, 1.0, MAX_LEN), MAX_LEN)}
GROUPS = ["emb_src", "emb_tgt", "pos", "enc_self", "enc_ffn", "dec_self", "dec_cross", "dec_ffn", "wout"]


def make_model():
    return M.JaKoMT(VOCAB, 96, 192, 2, 4, cfg, SEED, enc_pattern="", dec_pattern="ccaf", positional=16)


def groups_of(model):
    idx = M.param_index(model)
    lab = {idx["emb_src"]: "emb_src", idx["emb_tgt"]: "emb_tgt", idx["wout"]: "wout"}
    for stack, base, pre in ((model.enc, idx["enc"], "enc"), (model.dec, idx["dec"], "dec")):
        for f, off in zip(stack.fs, stack.block_offsets()):
            g = pre + (("_cross" if f.cross else "_self") if f.kind == "attn" else "_ffn")
            for c in range(len(f.blocks())):
                lab[base + off + c] = g
    if model.max_pos:
        lab[idx["pos_src"]] = lab[idx["pos_tgt"]] = "pos"
    return lab


def hinge(model, S):
    src, tin, tout = S
    n = min(128, src.shape[0])
    tot, corr = 0, 0
    for st in range(0, n, 32):
        mem, _c, _i, _x = model.encode(src[st:st + 32])
        dec, _c2, _i2, _x2 = model.decode(tin[st:st + 32], mem, src[st:st + 32])
        lg, _s = model.logits(dec)
        _g, stt = M.hinge_grad(lg, tout[st:st + 32], HB)
        tot += stt["hinge_sum"]; corr += stt["correct"]
    return tot, corr


CONFIGS = [(512, 32, 1), (512, 32, 3), (512, 128, 1), (128, 32, 1), (128, 8, 1)]
EPOCHS = 3
OUT = {}
for n, b, lr in CONFIGS:
    S = SETS[n]
    src, tin, tout = S
    model = make_model()
    lab = groups_of(model)
    sizes = {}
    for i, w in enumerate(model.parameters()):
        sizes[lab[i]] = sizes.get(lab[i], 0) + w.numel()
    opt = BlockSGD(model.parameters(), param_bits=12, lr_bits=lr, clip=1, momentum_bits=0)
    tag = "%d@b%d_lr%d" % (n, b, lr)
    h0, c0 = hinge(model, S)
    rows = []
    print("\n=== %s  (%d steps/epoch, shift %d)  hinge0 %d correct0 %d ===" % (tag, (n + b - 1) // b, opt.shift, h0, c0), flush=True)
    print("%-5s %9s %8s | %s" % ("epoch", "hinge", "correct", "  ".join("%-9s" % g for g in GROUPS)))
    for ep in range(1, EPOCHS + 1):
        moved = {g: 0 for g in GROUPS}
        for st in range(0, n, b):
            before = [w.clone() for w in model.parameters()]
            g, _s = M.forward_backward(model, src[st:st + b], tin[st:st + b], tout[st:st + b], HB, cached=False)
            opt.step(g)
            for i, (w0, w1) in enumerate(zip(before, model.parameters())):
                moved[lab[i]] += int((w0 != w1).sum())
        h, c = hinge(model, S)
        row = {"epoch": ep, "hinge": h, "correct": c, "moved_pct": {g: 100.0 * moved[g] / sizes[g] for g in GROUPS}}
        rows.append(row)
        print("%-5d %9d %8d | %s" % (ep, h, c, "  ".join("%8.3f%%" % row["moved_pct"][g] for g in GROUPS)), flush=True)
    OUT[tag] = {"hinge0": h0, "correct0": c0, "rows": rows, "sizes": sizes}
(SP / "epoch_firing_diag.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8")
print("\nwritten", SP / "epoch_firing_diag.json", flush=True)
