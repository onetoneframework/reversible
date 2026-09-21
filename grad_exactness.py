"""Element-level gradient exactness of the production backward (hidden_bits 16, accum_bits 32).

Three statements, each measured on one training batch of the 512-pair model at two states:
  1. hb16/acc32 == hb16/acc63 elementwise (mantissa and exponent), i.e. no accumulator ever clips;
  2. against the 32-bit-mantissa reference (hb32/acc63): unrounded cosine, max and mean relative
     deviation of the represented values, and the fraction of elements whose 16-bit mantissa is the
     correctly rounded/truncated image of the 32-bit one;
  3. the 12-bit BlockSGD votes derived from (1) and from the reference agree in sign and value.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(r"C:\Project\OnetoneFramework\framework\res\Platform\Python\reversible")
SP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import reversible_integer_final as R
import reversible_arch as A
import reversible_jako_mt as M
import reversible_jako_data as D
import reversible_jako_overfit32 as O

torch.set_num_threads(4)
A.SCORE_LEAK, A.SCORE_EXP2, A.SCORE_EXP2_CAP, A.SCORE_EXP2_COUPLED = 3, 3, 12, True
M.CROSS_QUERY_ANCHOR, M.ATTN_ROW_MASS_BITS, M.ATTN_SCORE_BITS = False, 0, 0
cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
R.CFG = cfg
ds = D.prepare("both", 0, 5000, 16, seed=0, max_tok=14, tok_mode="char", base_pairs=8000, min_char_count=20)
tok = ds["tokenizer"]
src, tin, tout = O.encode(tok, O.pick_pairs(ds, tok, 512, 1.0, 16), 16)
V = ds["info"]["vocab_size"]


def make():
    return M.JaKoMT(V, 96, 192, 2, 4, cfg, 0, enc_pattern="", dec_pattern="ccaf", positional=16)


def grads(model, hb, acc, lo=0, hi=32):
    cfg.accum_bits = acc
    R.STATS["accum_saturations"] = 0
    g, _ = M.forward_backward(model, src[lo:hi], tin[lo:hi], tout[lo:hi], hb, cached=False)
    sat = int(R.STATS["accum_saturations"])
    cfg.accum_bits = 32
    return [(gm.clone(), int(ge)) for gm, ge in g], sat


def states():
    yield "init", make()
    ck = torch.load(ROOT / "cp_p128d96.pt", weights_only=False)
    m = make()
    for w, s in zip(m.parameters(), ck["weights"]):
        w.copy_(s)
    yield "cp_p128d96", m


OUT = {}
for sname, model in states():
    for lo in (0, 256):
        p32, s32 = grads(model, 16, 32, lo, lo + 32)
        p63, s63 = grads(model, 16, 63, lo, lo + 32)
        ref, sref = grads(model, 32, 63, lo, lo + 32)
        ident = all(torch.equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(p32, p63))
        # represented values in float64: m * 2^e ; compare production to the 32-bit reference
        dots = nn = rr = 0.0
        maxrel = 0.0; sumrel = 0.0; cnt = 0; nz_ref = 0; exact_img = 0
        vote_agree = vote_both = 0; vote_equal = 0
        for (m, e), (rm, re) in zip(p32, ref):
            f = m.double() * (2.0 ** e); fr = rm.double() * (2.0 ** re)
            dots += float((f * fr).sum()); nn += float((f * f).sum()); rr += float((fr * fr).sum())
            mask = rm != 0
            nz_ref += int(mask.sum())
            rel = ((f - fr).abs() / fr.abs().clamp_min(1e-300))[mask]
            if rel.numel():
                maxrel = max(maxrel, float(rel.max())); sumrel += float(rel.sum()); cnt += rel.numel()
            # is the 16-bit mantissa the truncated image of the 32-bit one at the same exponent?
            d = e - re
            img = R.rshift0(rm, d) if d >= 0 else rm << (-d)
            exact_img += int((img == m)[mask].sum())
            v, vr = R.renorm_tensor(m, 12), R.renorm_tensor(rm, 12)
            both = (v != 0) & (vr != 0)
            vote_both += int(both.sum()); vote_agree += int((both & (torch.sign(v) == torch.sign(vr))).sum())
            vote_equal += int((v == vr).sum())
        total = sum(m.numel() for m, _ in p32)
        row = {"saturations_hb16_acc32": s32, "saturations_hb16_acc63": s63, "saturations_ref": sref,
               "hb16_acc32_equals_hb16_acc63_elementwise": ident,
               "cosine_vs_hb32_acc63": dots / (nn * rr) ** 0.5,
               "max_relative_deviation": maxrel, "mean_relative_deviation": sumrel / max(1, cnt),
               "nonzero_reference_elements": nz_ref,
               "fraction_16bit_mantissa_is_truncated_image_of_32bit": exact_img / max(1, nz_ref),
               "vote_sign_agreement": vote_agree / max(1, vote_both),
               "vote_value_equal_fraction": vote_equal / total}
        OUT["%s_batch%d" % (sname, lo // 32)] = row
        print(sname, "batch", lo // 32, json.dumps(row, indent=1), flush=True)
(SP / "grad_exactness.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8")
