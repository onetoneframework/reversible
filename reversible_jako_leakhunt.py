#!/usr/bin/env python3
"""
Locate the first tensor in the decoder whose rows before `cut` change when the
target tokens at and after `cut` change. Runs the decoder twice on inputs that
agree on positions < cut and reports, per block and per intermediate, whether
the earlier rows stayed bit-identical.

  python reversible_jako_leakhunt.py --cut 3
"""

import argparse
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_integer_final as R
import reversible_jako_mt as M

TAPS = ("Q", "K", "V", "raw", "shift", "S", "Aw", "out")


def instrument(model, store):
    """Record every attention intermediate, plus each block's input and output."""
    for i, f in enumerate(model.dec.fs):
        if f.kind == "attn":
            def wrapped(x, ctx, _f=f, _i=i):
                cfg = _f.cfg
                kv = ctx["memory"] if _f.cross else x
                mask = ctx["cross_mask"] if _f.cross else ctx["self_mask"]
                B, Tq, d = x.shape
                Tk = kv.shape[1]
                f2, cache = M.AttnF.forward(_f, x, ctx)
                (a2, k2, Q, K, V, S, Aw, o2, sq, sk, sv, ss, so, sf,
                 mk, _B, _Tq, _Tk, anchored, _sn) = cache
                raw = R.saturate(R.imatmul(Q, K.transpose(1, 2)), cfg.accum_bits)
                store.append((_i, _f.kind + ("-cross" if _f.cross else "-self"), {
                    "in": x, "Q": Q, "K": K, "V": V, "raw": raw,
                    "shift": ss.reshape(B, Tq, 1), "S": S, "Aw": Aw, "out": f2}))
                return f2, cache
        else:
            def wrapped(x, ctx, _f=f, _i=i):
                f2, cache = M.FFNF.forward(_f, x, ctx)
                store.append((_i, "ffn", {"in": x, "out": f2}))
                return f2, cache
        f.forward = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=600)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ffn-hidden", type=int, default=128)
    ap.add_argument("--k-enc", type=int, default=2)
    ap.add_argument("--k-dec", type=int, default=8)
    ap.add_argument("--enc-pattern", default="af")
    ap.add_argument("--dec-pattern", default="ccafccaf")
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
    R.CFG = cfg
    model = M.JaKoMT(a.vocab, a.dim, a.ffn_hidden, a.k_enc, a.k_dec, cfg, 0,
                     enc_pattern=a.enc_pattern, dec_pattern=a.dec_pattern)

    g = torch.Generator().manual_seed(0)
    src = torch.randint(4, a.vocab, (a.batch, a.max_len), generator=g)
    ta = torch.randint(4, a.vocab, (a.batch, a.max_len), generator=g)
    ta[:, 0] = M.BOS
    tb = ta.clone()
    tb[:, a.cut:] = torch.randint(4, a.vocab, (a.batch, a.max_len - a.cut), generator=g)

    runs = []
    for tin in (ta, tb):
        store = []
        instrument(model, store)
        mem, _c, _i, _x = model.encode(src)
        dec, _c2, _i2, _x2 = model.decode(tin, mem, src)
        lg, _s = model.logits(dec)
        runs.append((store, lg))
    (sa, lga), (sb, lgb) = runs

    print("cut=%d  (target positions >= %d differ; everything before must not)"
          % (a.cut, a.cut))
    print("%-6s %-12s %s" % ("block", "kind", "  ".join("%-7s" % t for t in ("in",) + TAPS)))
    first = None
    for (i, kind, A), (_j, _k, B) in zip(sa, sb):
        cells = []
        for t in ("in",) + TAPS:
            if t not in A:
                cells.append("%-7s" % "-")
                continue
            x, y = A[t][:, :a.cut], B[t][:, :a.cut]
            same = torch.equal(x, y)
            cells.append("%-7s" % ("same" if same else "DIFF"))
            if not same and first is None:
                first = (i, kind, t)
        print("%-6d %-12s %s" % (i, kind, "  ".join(cells)))
    print("logits[:, :%d] identical: %s" % (a.cut, torch.equal(lga[:, :a.cut], lgb[:, :a.cut])))
    print("first divergence: %s" % (first,))


if __name__ == "__main__":
    raise SystemExit(main())
