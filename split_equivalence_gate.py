"""Routing-equivalence gate: SplitMomentumSGD(m7, s7) must equal BlockSGD(m7) bit for bit,
and SplitMomentumSGD(m7, s0) must equal BlockSGD(m7) on target-side tensors and BlockSGD(m0)
on source-pathway tensors, over real training steps of the 512-pair model."""

import sys
from pathlib import Path

import torch

ROOT = Path(r"C:\Project\OnetoneFramework\framework\res\Platform\Python\reversible")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "blocksgd_scale"))
import reversible_integer_final as R
import reversible_arch as A
import reversible_jako_mt as M
import reversible_jako_data as D
import reversible_jako_overfit32 as O
from reversible_integer_final import BlockSGD
import jako_train_split as JS

torch.set_num_threads(4)
A.SCORE_LEAK, A.SCORE_EXP2, A.SCORE_EXP2_CAP, A.SCORE_EXP2_COUPLED = 3, 3, 12, True
M.CROSS_QUERY_ANCHOR, M.ATTN_ROW_MASS_BITS, M.ATTN_SCORE_BITS = False, 0, 0
cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
R.CFG = cfg
ds = D.prepare("both", 0, 5000, 16, seed=0, max_tok=14, tok_mode="char", base_pairs=8000, min_char_count=20)
tok = ds["tokenizer"]
src, tin, tout = O.encode(tok, O.pick_pairs(ds, tok, 512, 1.0, 16), 16)
V = ds["info"]["vocab_size"]


def model():
    return M.JaKoMT(V, 96, 192, 2, 4, cfg, 0, enc_pattern="", dec_pattern="ccaf", positional=16)


def run(mk_opt, steps=48):
    m = model()
    opt = mk_opt(m)
    for st in range(steps):
        lo = (st * 32) % 512
        g, _ = M.forward_backward(m, src[lo:lo + 32], tin[lo:lo + 32], tout[lo:lo + 32], 16, cached=False)
        opt.step(g)
    return m


kw = dict(param_bits=12, lr_bits=1, clip=1, weight_bits=8)
uni7 = run(lambda m: BlockSGD(m.parameters(), momentum_bits=7, **kw))
uni0 = run(lambda m: BlockSGD(m.parameters(), momentum_bits=0, **kw))
s77 = run(lambda m: JS.SplitMomentumSGD(m.parameters(), JS.groups_of(m), 7, 7, **kw))
s70 = run(lambda m: JS.SplitMomentumSGD(m.parameters(), JS.groups_of(m), 7, 0, **kw))
lab = JS.groups_of(model())
same_77 = all(torch.equal(a, b) for a, b in zip(s77.parameters(), uni7.parameters()))
moved = sum(int((a != b).sum()) for a, b in zip(uni7.parameters(), model().parameters()))
tgt_ok = all(torch.equal(a, b) for i, (a, b) in enumerate(zip(s70.parameters(), uni7.parameters())) if lab[i] not in JS.SOURCE_GROUPS)
src_ok = all(torch.equal(a, b) for i, (a, b) in enumerate(zip(s70.parameters(), uni0.parameters())) if lab[i] in JS.SOURCE_GROUPS)
differs_from_uniform = any(not torch.equal(a, b) for a, b in zip(s70.parameters(), uni7.parameters()))
print("steps 48 on 512 pairs; weights moved by uniform m7: %d" % moved)
print("split(m7,s7) == uniform m7, all 25 tensors bit-identical:", same_77)
print("split(m7,s0) target-side tensors == uniform m7:", tgt_ok)
print("split(m7,s0) source-pathway tensors == uniform m0:", src_ok)
print("split(m7,s0) differs from uniform m7 somewhere (sanity):", differs_from_uniform)
