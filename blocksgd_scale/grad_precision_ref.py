"""Is the integer gradient wrong because of the mantissa width (hidden_bits) or
because the backward's GEMM accumulators saturate at accum_bits = 32?

Reference: the same integer backward with accum_bits = 63 (no saturation) at
hidden_bits 32 -- int64-exact for this model (32 + 16 + 9 = 57 bits). Its
forward is checked to be bit-identical to the accum_bits = 32 forward, so the
only thing the reference changes is that no gradient accumulator is clipped.
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

torch.set_num_threads(5)
A.SCORE_LEAK, A.SCORE_EXP2, A.SCORE_EXP2_CAP, A.SCORE_EXP2_COUPLED = 3, 3, 12, True
M.CROSS_QUERY_ANCHOR, M.ATTN_ROW_MASS_BITS, M.ATTN_SCORE_BITS = False, 0, 0
cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
R.CFG = cfg
MAX_LEN, SEED = 16, 0
ds = D.prepare("both", 0, 5000, MAX_LEN, seed=SEED, max_tok=14, tok_mode="char",
               base_pairs=8000, min_char_count=20)
tok = ds["tokenizer"]
VOCAB = ds["info"]["vocab_size"]
p512 = O.pick_pairs(ds, tok, 512, 1.0, MAX_LEN)
src, tin, tout = O.encode(tok, p512, MAX_LEN)
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


def states():
    yield "init", make_model()
    ck = torch.load(ROOT / "cp_p128d96.pt", weights_only=False)
    m = make_model()
    for w, s in zip(m.parameters(), ck["weights"]):
        w.copy_(s)
    yield "cp_p128d96", m


def run(model, hb, accum, lo=0, hi=32):
    cfg.accum_bits = accum
    R.STATS["accum_saturations"] = 0
    g, _ = M.forward_backward(model, src[lo:hi], tin[lo:hi], tout[lo:hi], hb, cached=False)
    sat = int(R.STATS["accum_saturations"])
    cfg.accum_bits = 32
    return [(gm, int(ge)) for gm, ge in g], sat


def logits_of(model, accum):
    cfg.accum_bits = accum
    mem, _c, _i, _x = model.encode(src[:32])
    dec, _c2, _i2, _x2 = model.decode(tin[:32], mem, src[:32])
    lg, _s = model.logits(dec)
    cfg.accum_bits = 32
    return lg


def compare(g, ref, lab):
    agg = {}
    for i, ((gm, ge), (rm, re)) in enumerate(zip(g, ref)):
        a = agg.setdefault(lab[i], {"dot": 0.0, "nn": 0.0, "rr": 0.0, "both": 0, "agree": 0, "mass": 0, "mass_agree": 0})
        f = gm.double().reshape(-1) * (2.0 ** ge)
        fr = rm.double().reshape(-1) * (2.0 ** re)
        a["dot"] += float((f * fr).sum()); a["nn"] += float((f * f).sum()); a["rr"] += float((fr * fr).sum())
        v, vr = R.renorm_tensor(gm, 12).reshape(-1), R.renorm_tensor(rm, 12).reshape(-1)
        both = (v != 0) & (vr != 0)
        a["both"] += int(both.sum()); a["agree"] += int((both & (torch.sign(v) == torch.sign(vr))).sum())
        a["mass"] += int(v.abs().sum()); a["mass_agree"] += int(v.abs()[torch.sign(v) == torch.sign(vr)].sum())
    return {k: {"cosine": a["dot"] / ((a["nn"] * a["rr"]) ** 0.5 + 1e-300),
                "vote_sign_agree": (a["agree"] / a["both"]) if a["both"] else None,
                "vote_mass_agree": (a["mass_agree"] / a["mass"]) if a["mass"] else None}
            for k, a in agg.items()}


CONFIGS = [(16, 32), (16, 48), (16, 63), (20, 32), (24, 32), (24, 63), (28, 32), (32, 32), (32, 48)]
REF = (32, 63)
OUT = {}
for sname, model in states():
    lab = groups_of(model)
    same_fwd = torch.equal(logits_of(model, 32), logits_of(model, 63))
    ref, ref_sat = run(model, *REF)
    print("\n##### state %s: forward(accum32) == forward(accum63): %s ; reference saturations %d" % (sname, same_fwd, ref_sat), flush=True)
    OUT[sname] = {"forward_identical": same_fwd, "reference": {"hb": REF[0], "accum": REF[1], "saturations": ref_sat}, "rows": {}}
    print("%-9s %-6s %10s | %s" % ("hb/acc", "", "saturate", "  ".join("%-16s" % g for g in GROUPS)))
    for hb, acc in CONFIGS:
        g, sat = run(model, hb, acc)
        c = compare(g, ref, lab)
        OUT[sname]["rows"]["hb%d_acc%d" % (hb, acc)] = {"saturations": sat, "groups": c}
        print("%-9s %-6s %10d | %s" % ("hb%d" % hb, "acc%d" % acc, sat,
              "  ".join("%5.2f/%s" % (c[gname]["cosine"], ("%.2f" % c[gname]["vote_mass_agree"]) if c.get(gname, {}).get("vote_mass_agree") is not None else "  - ") if gname in c else "%-16s" % "-" for gname in GROUPS)), flush=True)
    print("   (cell = cosine vs reference / BlockSGD 12-bit vote mass sign-agreement)")
(SP / "grad_precision_ref.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8")
print("\nwritten", SP / "grad_precision_ref.json", flush=True)
