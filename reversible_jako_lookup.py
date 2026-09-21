#!/usr/bin/env python3
"""
Can this integer training system learn anything that *requires* the source?

The 32-pair overfit showed the decoder scoring 65.73% with a gold prefix and
7.30% without it, with the source worth 2.25 points either way -- and the two
wrong-prefix conditions landing on 7.3034% to four decimals whether the source
was right or wrong. That says the model took a target-side shortcut, but it does
not say the source path is incapable, because in that setup the shortcut was
sufficient.

This removes the shortcut instead of arguing about it.

  mode=lookup1   32 distinct source tokens, each mapped to a random target
                 token. The decoder input is BOS and nothing else, and the loss
                 is at position 0 only. There is no prefix to memorise and no
                 order to represent, so nothing but source identity can produce
                 the answer. A working encoder -> cross-attention -> decoder
                 path memorises 32 arbitrary pairs easily.

  mode=lookupN   the same, but each source is 2-4 random tokens and the target
                 is still a single label. Tests whether cross-attention can
                 tell multi-token sources apart, still with no prefix and no
                 order dependence.

  mode=copy      sequence copy with a freshly sampled batch every step, so
                 transition memorisation cannot help. Run this last: the model
                 has no positional encoding, so a strict ordered copy can fail
                 for a reason these first two tests do not have.

Each report gives accuracy with the real source and with sources shuffled
between examples. Shuffled accuracy near chance is the proof that what was
learned is source identity and not a marginal prior.

  python reversible_jako_lookup.py --mode lookup1 --items 32 --epochs 4000
"""

import argparse
import json
import sys
import time

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_integer_final as R
import reversible_arch as A
import reversible_jako_mt as M
from reversible_integer_final import BlockSGD


def make_task(mode, items, vocab, max_len, gen, src_len=3):
    """Returns (src, tin, tout). Loss sits at position 0 only."""
    lo = 4
    if mode == "lookup1":
        toks = torch.randperm(vocab - lo, generator=gen)[:items] + lo
        src = torch.full((items, max_len), M.PAD, dtype=torch.int64)
        src[:, 0] = toks
        src[:, 1] = M.EOS
    else:
        src = torch.full((items, max_len), M.PAD, dtype=torch.int64)
        for i in range(items):
            n = int(torch.randint(2, src_len + 1, (1,), generator=gen))
            src[i, :n] = torch.randint(lo, vocab, (n,), generator=gen)
            src[i, n] = M.EOS
    tgt = torch.randint(lo, vocab, (items,), generator=gen)
    tin = torch.full((items, max_len), M.PAD, dtype=torch.int64)
    tin[:, 0] = M.BOS
    tout = torch.full((items, max_len), M.PAD, dtype=torch.int64)
    tout[:, 0] = tgt
    return src, tin, tout


def evaluate(model, src, tin, tout, perm, batch):
    out = {}
    for name, s in (("real", src), ("shuffled", src[perm])):
        hit = 0
        ranks = []
        for st in range(0, src.shape[0], batch):
            mem, _c, _i, _x = model.encode(s[st:st + batch])
            dec, _c2, _i2, _x2 = model.decode(tin[st:st + batch], mem,
                                              s[st:st + batch])
            lg, _sh = model.logits(dec)
            z0 = lg[:, 0, :]
            gold = tout[st:st + batch, 0]
            gl = z0.gather(1, gold[:, None]).reshape(-1)
            rank = (z0 > gl[:, None]).sum(dim=1) + 1
            hit += int((rank == 1).sum())
            ranks.append(rank)
        rk = torch.cat(ranks)
        out[name] = {"acc": round(100.0 * hit / src.shape[0], 2),
                     "mean_rank": round(float(rk.double().mean()), 1),
                     "median_rank": int(rk.median()),
                     "top10": round(100.0 * float((rk <= 10).double().mean()), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lookup1", "lookupN"], default="lookup1")
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--src-len", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--report-every", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=5000)
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--ffn-hidden", type=int, default=192)
    ap.add_argument("--k-enc", type=int, default=2)
    ap.add_argument("--k-dec", type=int, default=4)
    ap.add_argument("--enc-pattern", default="")
    ap.add_argument("--dec-pattern", default="ccaf")
    ap.add_argument("--score-leak", type=int, default=3)
    ap.add_argument("--lr-bits", type=int, default=0)
    ap.add_argument("--hidden-bits", type=int, default=16)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    A.SCORE_LEAK = a.score_leak
    M.CROSS_QUERY_ANCHOR = False
    cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
    R.CFG = cfg
    out_path = a.out or ("jako_%s.json" % a.mode)

    gen = torch.Generator().manual_seed(a.seed)
    src, tin, tout = make_task(a.mode, a.items, a.vocab, a.max_len, gen, a.src_len)
    model = M.JaKoMT(a.vocab, a.dim, a.ffn_hidden, a.k_enc, a.k_dec, cfg, a.seed,
                     enc_pattern=a.enc_pattern, dec_pattern=a.dec_pattern)
    opt = BlockSGD(model.parameters(), param_bits=12, lr_bits=a.lr_bits, clip=1)
    perm = torch.randperm(a.items, generator=torch.Generator().manual_seed(99))
    chance = 100.0 / a.items

    print(json.dumps({"mode": a.mode, "items": a.items, "vocab": a.vocab,
                      "dim": a.dim, "k_dec": a.k_dec, "dec_pattern": a.dec_pattern,
                      "lr_bits": a.lr_bits, "loss_positions": 1,
                      "chance_acc_pct": round(chance, 2)}), flush=True)
    print("\n%7s %9s %9s %10s %10s %8s %11s %8s %6s" % (
        "epoch", "acc_real", "acc_shuf", "rank_real", "rank_shuf", "hinge",
        "|acc|max", "pinned%", "min"), flush=True)

    hist = []
    t0 = time.time()
    n = a.items
    for ep in range(a.epochs + 1):
        if ep % a.report_every == 0:
            ev = evaluate(model, src, tin, tout, perm, a.batch)
            _g, st = M.forward_backward(model, src, tin, tout, a.hidden_bits,
                                        cached=False)
            hp = st["hinge_sum"] / max(1, st["tokens"])
            # Error-feedback health. The accumulator drains 2**shift per step
            # and gains up to 2**param_bits - 1 from renorm_tensor, so a shift
            # below param_bits lets it grow without bound and the weight ramps
            # in one direction until the width clamp catches it.
            accmax = max(int(x.abs().max()) for x in opt.acc)
            lo, hi = -(1 << 7), (1 << 7) - 1
            pinned = sum(int(((w <= lo) | (w >= hi)).sum()) for w in model.parameters())
            total = sum(w.numel() for w in model.parameters())
            row = {"epoch": ep, "acc_real": ev["real"]["acc"],
                   "acc_shuffled": ev["shuffled"]["acc"],
                   "mean_rank_real": ev["real"]["mean_rank"],
                   "mean_rank_shuffled": ev["shuffled"]["mean_rank"],
                   "top10_real": ev["real"]["top10"], "hinge": round(hp, 3),
                   "acc_buffer_max": accmax, "drain_per_step": 1 << opt.shift,
                   "weights_pinned_pct": round(100.0 * pinned / total, 3),
                   "minutes": round((time.time() - t0) / 60, 1)}
            hist.append(row)
            print("%7d %8.2f%% %8.2f%% %10.1f %10.1f %8.3f %11d %7.3f%% %6.1f" % (
                ep, row["acc_real"], row["acc_shuffled"], row["mean_rank_real"],
                row["mean_rank_shuffled"], row["hinge"], row["acc_buffer_max"],
                row["weights_pinned_pct"], row["minutes"]), flush=True)
            if row["acc_real"] >= 100.0:
                print("all %d learned; stopping early" % n, flush=True)
                break
        if ep == a.epochs:
            break
        order = torch.randperm(n, generator=gen)
        for st in range(0, n, a.batch):
            b = order[st:st + a.batch]
            g, _s = M.forward_backward(model, src[b], tin[b], tout[b],
                                       a.hidden_bits, cached=False)
            opt.step(g)

    best = max(h["acc_real"] for h in hist)
    best_shuf = max(h["acc_shuffled"] for h in hist)
    if best >= 90.0:
        verdict = ("PASS: the encoder -> cross-attention -> decoder path learns a "
                   "source-determined mapping; the translation failure is training "
                   "dynamics, not a broken source path")
    elif best >= 3 * chance:
        verdict = ("PARTIAL: the source path carries signal but does not converge; "
                   "suspect optimisation granularity or capacity, not wiring")
    else:
        verdict = ("FAIL: with the target-prefix shortcut removed the model cannot "
                   "learn %d arbitrary source->target pairs; the source-conditioned "
                   "path itself is the defect" % n)
    print("\nbest acc real %.2f%%  shuffled %.2f%%  chance %.2f%%" %
          (best, best_shuf, chance))
    print("VERDICT %s" % verdict)
    json.dump({"config": vars(a), "history": hist, "verdict": verdict,
               "best_acc_real": best, "best_acc_shuffled": best_shuf,
               "chance": chance}, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
