"""Where does the 512-pair source-conditioning signal get lost?

Two read-only measurements on the production model, at two states (fresh init
and the cp_p128d96 128-pair memoriser), on the historical 512 / 128 pair sets:

  1. gradient precision: SharedGrad at hidden_bits 16/20/24/32 against a 40-bit
     reference, per parameter group, both raw and as BlockSGD sees it after
     renorm_tensor(., 12).
  2. batch consensus: the 12-bit votes of every batch of an epoch versus their
     sum, for 512@b32 (16 batches), 512@b128 (4 batches) and 128@b32 (4 batches).
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

torch.set_num_threads(6)
A.SCORE_LEAK, A.SCORE_EXP2, A.SCORE_EXP2_CAP, A.SCORE_EXP2_COUPLED = 3, 3, 12, True
M.CROSS_QUERY_ANCHOR, M.ATTN_ROW_MASS_BITS, M.ATTN_SCORE_BITS = False, 0, 0
cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
R.CFG = cfg
MAX_LEN, SEED = 16, 0
ds = D.prepare("both", 0, 5000, MAX_LEN, seed=SEED, max_tok=14, tok_mode="char",
               base_pairs=8000, min_char_count=20)
tok = ds["tokenizer"]
VOCAB = ds["info"]["vocab_size"]
p128 = O.pick_pairs(ds, tok, 128, 1.0, MAX_LEN)
p512 = O.pick_pairs(ds, tok, 512, 1.0, MAX_LEN)
S128 = O.encode(tok, p128, MAX_LEN)
S512 = O.encode(tok, p512, MAX_LEN)


def make_model():
    return M.JaKoMT(VOCAB, 96, 192, 2, 4, cfg, SEED, enc_pattern="", dec_pattern="ccaf", positional=16)


def groups_of(model):
    """Parameter index -> group label."""
    idx = M.param_index(model)
    lab = {idx["emb_src"]: "emb_src", idx["emb_tgt"]: "emb_tgt", idx["wout"]: "wout"}
    for stack, base, pre in ((model.enc, idx["enc"], "enc"), (model.dec, idx["dec"], "dec")):
        for f, off in zip(stack.fs, stack.block_offsets()):
            if f.kind == "attn":
                g = pre + ("_cross" if f.cross else "_self")
            else:
                g = pre + "_ffn"
            for c in range(len(f.blocks())):
                lab[base + off + c] = g
    if model.max_pos:
        lab[idx["pos_src"]] = "pos"
        lab[idx["pos_tgt"]] = "pos"
    return lab


def states():
    yield "init", make_model()
    ck = torch.load(ROOT / "cp_p128d96.pt", weights_only=False)
    m = make_model()
    assert len(ck["weights"]) == len(m.parameters())
    for w, s in zip(m.parameters(), ck["weights"]):
        w.copy_(s)
    yield "cp_p128d96", m


def grads(model, S, lo, hi, hb):
    src, tin, tout = S
    g, _ = M.forward_backward(model, src[lo:hi], tin[lo:hi], tout[lo:hi], hb, cached=False)
    return [(gm, int(ge)) for gm, ge in g]


def as_float(gm, ge):
    return gm.double() * (2.0 ** ge)


OUT = {}

# ---------------------------------------------------------- 1. precision vs hb
HBS = [16, 20, 24, 32]
REF = 40
prec = {}
for sname, model in states():
    lab = groups_of(model)
    ref = grads(model, S512, 0, 32, REF)
    rows = {}
    for hb in HBS:
        g = grads(model, S512, 0, 32, hb)
        agg = {}
        for i, ((gm, ge), (rm, re)) in enumerate(zip(g, ref)):
            a = agg.setdefault(lab[i], {"dot": 0.0, "nn": 0.0, "rr": 0.0, "n": 0, "nz": 0, "nz_ref": 0,
                                         "vote_nz": 0, "vote_nz_ref": 0, "vote_both": 0, "vote_agree": 0,
                                         "vote_mass": 0, "vote_mass_agree": 0})
            f, fr = as_float(gm, ge).reshape(-1), as_float(rm, re).reshape(-1)
            a["dot"] += float((f * fr).sum()); a["nn"] += float((f * f).sum()); a["rr"] += float((fr * fr).sum())
            a["n"] += f.numel(); a["nz"] += int((gm != 0).sum()); a["nz_ref"] += int((rm != 0).sum())
            v, vr = R.renorm_tensor(gm, 12).reshape(-1), R.renorm_tensor(rm, 12).reshape(-1)
            both = (v != 0) & (vr != 0)
            a["vote_nz"] += int((v != 0).sum()); a["vote_nz_ref"] += int((vr != 0).sum())
            a["vote_both"] += int(both.sum())
            a["vote_agree"] += int((both & (torch.sign(v) == torch.sign(vr))).sum())
            a["vote_mass"] += int(v.abs().sum())
            a["vote_mass_agree"] += int(v.abs()[torch.sign(v) == torch.sign(vr)].sum())
        rows[hb] = {gname: {"cosine": a["dot"] / ((a["nn"] * a["rr"]) ** 0.5 + 1e-300),
                            "nonzero_frac": a["nz"] / a["n"], "nonzero_frac_ref": a["nz_ref"] / a["n"],
                            "vote_nonzero_frac": a["vote_nz"] / a["n"], "vote_nonzero_frac_ref": a["vote_nz_ref"] / a["n"],
                            "vote_sign_agree": (a["vote_agree"] / a["vote_both"]) if a["vote_both"] else None,
                            "vote_mass_agree": (a["vote_mass_agree"] / a["vote_mass"]) if a["vote_mass"] else None}
                    for gname, a in agg.items()}
    prec[sname] = rows
    print("\n=== precision vs hb%d reference, state %s, 512 batch[0:32] ===" % (REF, sname), flush=True)
    print("%-10s %4s %8s %10s %10s %11s %11s" % ("group", "hb", "cosine", "nz", "nz_ref", "vote_sign", "vote_mass"))
    for gname in ["emb_src", "emb_tgt", "pos", "enc_self", "enc_ffn", "dec_self", "dec_cross", "dec_ffn", "wout"]:
        for hb in HBS:
            r = rows[hb].get(gname)
            if r:
                print("%-10s %4d %8.4f %10.5f %10.5f %11s %11s" % (
                    gname, hb, r["cosine"], r["nonzero_frac"], r["nonzero_frac_ref"],
                    "%.4f" % r["vote_sign_agree"] if r["vote_sign_agree"] is not None else "-",
                    "%.4f" % r["vote_mass_agree"] if r["vote_mass_agree"] is not None else "-"), flush=True)
OUT["precision"] = prec

# ------------------------------------------------------------ 2. batch consensus
cons = {}
SHIFT = 12
for sname, model in states():
    lab = groups_of(model)
    cons[sname] = {}
    for tag, S, b in (("512@b32", S512, 32), ("512@b128", S512, 128), ("128@b32", S128, 32)):
        n = S[0].shape[0]
        votes = []
        for st in range(0, n, b):
            g = grads(model, S, st, st + b, 16)
            votes.append([R.renorm_tensor(gm, 12) for gm, _ in g])
        agg = {}
        for i in range(len(votes[0])):
            a = agg.setdefault(lab[i], {"sum_abs": 0, "abs_sum": 0, "n": 0, "fire": 0, "top": 0, "top_agree": 0,
                                         "nb": len(votes)})
            vs = torch.stack([v[i].reshape(-1) for v in votes])          # (B, N) int64
            full = vs.sum(0)
            a["sum_abs"] += int(full.abs().sum()); a["abs_sum"] += int(vs.abs().sum()); a["n"] += full.numel()
            a["fire"] += int((full.abs() >= (1 << SHIFT)).sum())
            top = vs.abs() >= 2048
            a["top"] += int(top.sum())
            a["top_agree"] += int((top & (torch.sign(vs) == torch.sign(full)[None, :])).sum())
        cons[sname][tag] = {gname: {"consensus_ratio": a["sum_abs"] / max(1, a["abs_sum"]),
                                    "epoch_fire_frac": a["fire"] / a["n"],
                                    "top_vote_agree_with_sum": (a["top_agree"] / a["top"]) if a["top"] else None,
                                    "batches": a["nb"]} for gname, a in agg.items()}
        print("\n=== consensus, state %s, %s (%d batches) ===" % (sname, tag, len(votes)), flush=True)
        print("%-10s %10s %12s %12s" % ("group", "consensus", "epoch_fire%", "top_agree"))
        for gname in ["emb_src", "emb_tgt", "pos", "enc_self", "enc_ffn", "dec_self", "dec_cross", "dec_ffn", "wout"]:
            r = cons[sname][tag].get(gname)
            if r:
                print("%-10s %10.4f %12.4f %12s" % (gname, r["consensus_ratio"], 100 * r["epoch_fire_frac"],
                                                  "%.4f" % r["top_vote_agree_with_sum"] if r["top_vote_agree_with_sum"] is not None else "-"), flush=True)
OUT["consensus"] = cons
(SP / "grad_precision_diag.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8")
print("\nwritten", SP / "grad_precision_diag.json", flush=True)
