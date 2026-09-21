"""The historical memorisation harness with one extra knob: weight_bits for the
projection matrices. Embedding and positional tables keep the int8-scale init
(+-16) so the residual-branch balance of the reversible state is unchanged;
only the grid the optimizer moves on is finer. Everything else -- pairs, model,
attention, hinge, BlockSGD(param_bits 12, clip 1), batch order, report -- is
reversible_jako_overfit32.py's.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import reversible_integer_final as R
import reversible_arch as A
import reversible_jako_mt as M
import reversible_jako_data as D
import reversible_jako_metrics as MET
import reversible_jako_overfit32 as O
from reversible_integer_final import BlockSGD

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", type=int, default=512)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr-bits", type=int, default=1)
ap.add_argument("--weight-bits", type=int, default=10)
ap.add_argument("--table-bits", type=int, default=8, help="init scale of embedding/positional tables")
ap.add_argument("--momentum-bits", type=int, default=0)
ap.add_argument("--epochs", type=int, default=900)
ap.add_argument("--report-every", type=int, default=150)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--source-momentum-bits", type=int, default=None,
                help="momentum bits for the source pathway (emb_src, pos_src, enc_*, dec_cross); "
                     "--momentum-bits then applies to the target-side tensors only")
ap.add_argument("--tb", type=int, default=5, help="activation target bits TB; linear outputs are TB+2 bits (historical 5 -> 7)")
ap.add_argument("--score-bits", type=int, default=0, help="attention score bits; 0 = TB+2")
ap.add_argument("--save-every-report", action="store_true", help="also save <out>_epNNNN.pt at every report")
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.set_num_threads(a.threads)
A.SCORE_LEAK, A.SCORE_EXP2, A.SCORE_EXP2_CAP, A.SCORE_EXP2_COUPLED = 3, 3, 12, True
M.CROSS_QUERY_ANCHOR, M.ATTN_ROW_MASS_BITS, M.ATTN_SCORE_BITS = False, 0, a.score_bits
# TB is a module constant copied at import; every forward reads its own module's copy at call time
R.TB = M.TB = A.TB = a.tb
O.QUICK[0] = 1
MAX_LEN, HB = 16, 16
cfg = R.RunConfig(state_bits=16, weight_bits=a.weight_bits, accum_bits=32)
R.CFG = cfg
ds = D.prepare("both", 0, 5000, MAX_LEN, seed=a.seed, max_tok=14, tok_mode="char",
               base_pairs=8000, min_char_count=20)
tok = ds["tokenizer"]
pairs = O.pick_pairs(ds, tok, a.pairs, 1.0, MAX_LEN)
src, tin, tout = O.encode(tok, pairs, MAX_LEN)
refs = [ko for _ja, ko in pairs]
klass = MET.build_classifier(tok, tout.tolist())
n = src.shape[0]

model = M.JaKoMT(ds["info"]["vocab_size"], 96, 192, 2, 4, cfg, a.seed,
                 enc_pattern="", dec_pattern="ccaf", positional=16)
# tables at the historical int8 scale; only the projections live on the finer grid
tgen = torch.Generator().manual_seed(a.seed + 1)
lim = 1 << (a.table_bits - 4)
for t in (model.emb_src, model.emb_tgt, model.pos_src, model.pos_tgt):
    t.copy_(torch.randint(-lim, lim + 1, t.shape, dtype=torch.int64, generator=tgen))
SOURCE_GROUPS = {"emb_src", "pos_src", "enc_self", "enc_ffn", "dec_cross"}


def groups_of(model):
    idx = M.param_index(model)
    lab = {idx["emb_src"]: "emb_src", idx["emb_tgt"]: "emb_tgt", idx["wout"]: "wout"}
    for stack, base, pre in ((model.enc, idx["enc"], "enc"), (model.dec, idx["dec"], "dec")):
        for f, off in zip(stack.fs, stack.block_offsets()):
            g = pre + (("_cross" if f.cross else "_self") if f.kind == "attn" else "_ffn")
            for c in range(len(f.blocks())):
                lab[base + off + c] = g
    if model.max_pos:
        lab[idx["pos_src"]], lab[idx["pos_tgt"]] = "pos_src", "pos_tgt"
    return lab


class SplitMomentumSGD:
    """Two canonical BlockSGD instances over the same parameters; each index is
    stepped by exactly one of them, so every update is bit-identical to the
    canonical optimizer at that index's momentum setting. No optimizer code is
    copied; the only new behaviour is the routing."""

    def __init__(self, params, lab, tgt_bits, src_bits, **kw):
        self.tgt = BlockSGD(params, momentum_bits=tgt_bits, **kw)
        self.src = BlockSGD(params, momentum_bits=src_bits, **kw)
        self.route = [self.src if lab[i] in SOURCE_GROUPS else self.tgt for i in range(len(params))]
        self.shift, self.stable, self.weight_bits = self.tgt.shift, self.tgt.stable, self.tgt.weight_bits
        self.acc = self.tgt.acc

    def step(self, grad_pairs):
        return max(self.route[i]._step_one(i, gm) for i, (gm, _ge) in enumerate(grad_pairs))


kw = dict(param_bits=12, lr_bits=a.lr_bits, clip=1, weight_bits=a.weight_bits)
if a.source_momentum_bits is None:
    opt = BlockSGD(model.parameters(), momentum_bits=a.momentum_bits, **kw)
else:
    opt = SplitMomentumSGD(model.parameters(), groups_of(model), a.momentum_bits, a.source_momentum_bits, **kw)
print(json.dumps({"pairs": n, "batch": a.batch, "lr_bits": a.lr_bits, "weight_bits": a.weight_bits,
                  "table_bits": a.table_bits, "momentum_bits": a.momentum_bits,
                  "source_momentum_bits": a.source_momentum_bits, "shift": opt.shift,
                  "TB": a.tb, "activation_bits": a.tb + 2, "score_bits": a.score_bits or (a.tb + 2),
                  "stable": opt.stable, "proj_init_lim": 1 << (a.weight_bits - 4), "table_init_lim": lim,
                  "clamp": [opt.weight_bits and -(1 << (opt.weight_bits - 1)), (1 << (opt.weight_bits - 1)) - 1]}), flush=True)
print("\n epoch   tf_all% tf_content%      pos0%   greedy     give1     give3     real%     shuf%  zero%   sat   |W|max", flush=True)
hist, t0 = [], time.time()
for ep in range(a.epochs + 1):
    if ep % a.report_every == 0:
        r = O.report(model, tok, src, tin, tout, klass, refs, MAX_LEN, a.batch)
        r["greedy_bleu"] = round(M.bleu(r["hyp"], refs), 3)
        r["epoch"] = ep; r["minutes"] = round((time.time() - t0) / 60, 1)
        r["accum_saturations"] = int(R.STATS["accum_saturations"])
        r["w_max_abs"] = max(int(w.abs().max()) for w in model.parameters())
        hist.append({k: v for k, v in r.items() if k not in ("hyp", "hyp_given1")})
        print("%6d %9.3f %11.4f %7.2f %4d/%-4d %6d %6d %9.4f %9.4f %7.4f %6d %7d  %.1fm" % (
            ep, r["tf_all_acc"], r["tf_content_acc"], r["tf_pos0_acc"], r["greedy_exact"], n, 0, 0,
            r["real_content"], r["shuffled_content"], r["zeroed_content"], r["accum_saturations"],
            r["w_max_abs"], r["minutes"]), flush=True)
        R.STATS["accum_saturations"] = 0
        json.dump({"args": vars(a), "history": hist}, open(a.out, "w", encoding="utf-8"), indent=1)
        if a.save_every_report and ep > 0:
            torch.save({"weights": [w.clone() for w in model.parameters()], "epoch": ep,
                        "greedy_exact": r["greedy_exact"], "tf_content_acc": r["tf_content_acc"],
                        "args": vars(a)}, a.out.replace(".json", "_ep%04d.pt" % ep))
        if r["greedy_exact"] == n:
            break
    if ep == a.epochs:
        break
    for st in range(0, n, a.batch):
        g, _s = M.forward_backward(model, src[st:st + a.batch], tin[st:st + a.batch],
                                   tout[st:st + a.batch], HB, cached=False)
        opt.step(g)
best_greedy = max(h["greedy_exact"] for h in hist)
best_tf = max(h["tf_content_acc"] for h in hist)
print("\nbest greedy exact %d/%d   best tf content acc %.3f%%" % (best_greedy, n, best_tf), flush=True)
torch.save({"weights": [w.clone() for w in model.parameters()], "opt_acc": [x.clone() for x in opt.acc],
            "args": vars(a)}, a.out.replace(".json", ".pt"))
json.dump({"args": vars(a), "history": hist, "best_greedy_exact": best_greedy,
           "best_tf_content_acc": best_tf}, open(a.out, "w", encoding="utf-8"), indent=1)
