#!/usr/bin/env python3
"""
Gates 2-5 for the reversible-integer JA->KO translator.

  Gate 2  exact inverse of the encoder and decoder reversible stacks
  Gate 3  cache-free reconstructed backward == cached backward, bit for bit,
          on every parameter gradient (mantissa AND exponent)
  Gate 4  per-layer optimizer stepping == batched stepping, weights and
          accumulators identical after every step
  Gate 5  attention gradients vs a float autograd mirror that reuses the
          integer path's own shifts, quantised scores and discrete masks

Gate 5's oracle deliberately consumes the integer forward's decisions. Building
it from unquantised scores gives a wrong oracle: that mistake produced a false
"formula error" verdict earlier in this project.

  python reversible_jako_verify.py --all
"""

import argparse
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_integer_final as R
import reversible_jako_mt as M
from reversible_integer_final import SharedGrad, BlockSGD, dense_linear_backward


def tiny_batch(vocab, B=4, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(4, vocab, (B, T), generator=g)
    tin = torch.randint(4, vocab, (B, T), generator=g)
    tout = torch.randint(4, vocab, (B, T), generator=g)
    src[:, -1] = M.EOS
    tin[:, 0] = M.BOS
    src[:, -2:] = M.PAD              # exercise the padding mask
    return src, tin, tout


def make(vocab=64, d=16, hidden=32, k_enc=2, k_dec=3, seed=0, hb=16):
    cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
    return M.JaKoMT(vocab, d, hidden, k_enc, k_dec, cfg, seed), cfg, hb


# -----------------------------------------------------------------------------
def gate2_inverse(vocab=64):
    model, cfg, hb = make(vocab)
    src, tin, _t = tiny_batch(vocab)
    enc_self, causal, cross = model.masks(src, tin)

    v0 = model.embed(src, model.emb_src)
    ctx = {"self_mask": enc_self, "cross_mask": None, "memory": None}
    enc_out = model.enc.forward(v0, ctx)
    enc_rec = model.enc.inverse(enc_out, ctx)
    enc_ok = bool(torch.equal(enc_rec, v0))

    mem = enc_out
    d0 = model.embed(tin, model.emb_tgt)
    dctx = {"self_mask": causal, "cross_mask": cross, "memory": mem}
    dec_out = model.dec.forward(d0, dctx)
    dec_rec = model.dec.inverse(dec_out, dctx)
    dec_ok = bool(torch.equal(dec_rec, d0))
    return {"gate": 2, "encoder_inverse_exact": enc_ok,
            "decoder_inverse_exact": dec_ok,
            "encoder_max_abs_state": int(enc_out.abs().max()),
            "decoder_max_abs_state": int(dec_out.abs().max()),
            "passed": enc_ok and dec_ok}


def gate3_cachefree(vocab=64):
    model, cfg, hb = make(vocab)
    src, tin, tout = tiny_batch(vocab)
    g_cached, st_c = M.forward_backward(model, src, tin, tout, hb, cached=True)
    g_free, st_f = M.forward_backward(model, src, tin, tout, hb, cached=False)
    same_m = [bool(torch.equal(a[0], b[0])) for a, b in zip(g_cached, g_free)]
    same_e = [int(a[1]) == int(b[1]) for a, b in zip(g_cached, g_free)]
    bad = [i for i, (m, e) in enumerate(zip(same_m, same_e)) if not (m and e)]
    return {"gate": 3, "tensors_compared": len(g_cached),
            "all_mantissas_equal": all(same_m), "all_exponents_equal": all(same_e),
            "first_mismatch_index": (bad[0] if bad else None),
            "stats_match": st_c == st_f,
            "passed": all(same_m) and all(same_e) and st_c == st_f}


def gate4_step_modes(vocab=64, steps=8):
    out = {"gate": 4, "steps": steps}
    ma, _c, hb = make(vocab)
    mb, _c2, _h = make(vocab)
    oa = BlockSGD(ma.parameters(), param_bits=12, lr_bits=2, clip=1)
    ob = BlockSGD(mb.parameters(), param_bits=12, lr_bits=2, clip=1)
    first_bad = None
    for s in range(steps):
        src, tin, tout = tiny_batch(vocab, seed=100 + s)
        grads, _ = M.forward_backward(ma, src, tin, tout, hb, cached=False)
        oa.step(grads)                                   # batched
        M.forward_backward(mb, src, tin, tout, hb, cached=False, opt=ob)  # per-layer
        w_ok = all(torch.equal(x, y) for x, y in zip(ma.parameters(), mb.parameters()))
        a_ok = all(torch.equal(x, y) for x, y in zip(oa.acc, ob.acc))
        if not (w_ok and a_ok) and first_bad is None:
            first_bad = s
            break
    out["first_divergent_step"] = first_bad
    out["passed"] = first_bad is None
    return out


def gate5_attention_cosine(vocab=64, adjoint_bits=(6, 12, 20),
                           hb_sweep=(10, 12, 16, 20, 24, 30, 40)):
    """Float mirror reusing the integer path's shifts, quantised S and mask."""
    model, cfg, _hb = make(vocab)
    src, tin, _t = tiny_batch(vocab)
    enc_self, _c, _x = model.masks(src, src)
    ctx = {"self_mask": enc_self, "cross_mask": None, "memory": None}
    a = model.embed(src, model.emb_src)[:, :, :model.d]
    layer = next(f for f in model.enc.fs if f.kind == "attn")

    def cos(u, v):
        u = u.reshape(-1).double(); v = v.reshape(-1).double()
        return float((u @ v / (u.norm() * v.norm())).item()) if u.norm() > 0 and v.norm() > 0 else float("nan")

    HB = 16          # production setting; see the sweep below for the safe window
    rows = []
    for gbits in adjoint_bits:
        f, cache = layer.forward(a, ctx)
        (a2, k2, Q, K, V, S, Aw, o2, sq, sk, sv, ss, so, sf, mask, B, Tq, Tk,
         _anchored, _sn) = cache
        gen = torch.Generator().manual_seed(7)
        lim = 1 << gbits
        gbm = torch.randint(-lim, lim + 1, f.shape, dtype=torch.int64, generator=gen)
        _gq, _gkv, gws = layer.backward(SharedGrad(gbm, 0), cache, HB)

        d = model.d
        Wq, Wk, Wv, Wo = (layer.Wq.double().requires_grad_(), layer.Wk.double().requires_grad_(),
                          layer.Wv.double().requires_grad_(), layer.Wo.double().requires_grad_())
        af = a2.double()

        def p(x, s):
            """Straight-through floor: forward matches `>> s` exactly, backward
            is multiplication by 2**-s, which is what the integer adjoint
            assumes. Plain division would silently change the forward values."""
            q = x / torch.pow(2.0, s.double())
            return q + (torch.floor(q) - q).detach()

        Qf = p(af @ Wq.T, sq).reshape(B, Tq, d)
        Kf = p(af @ Wk.T, sk).reshape(B, Tk, d)
        Vf = p(af @ Wv.T, sv).reshape(B, Tk, d)
        # The integer path masks the raw scores before normalising, so the
        # mirror must too or it stops being a mirror.
        Sf = p(((Qf @ Kf.transpose(1, 2)) * mask.double()).reshape(B * Tq, Tk),
               ss).reshape(B, Tq, Tk)
        # Reuse the integer path's own discrete decisions: the ReLU mask comes
        # from the quantised S, and the padding/causal mask is the same tensor.
        Af = Sf * (S > 0).double() * mask.double()
        Of = p((Af @ Vf).reshape(B * Tq, d), so).reshape(B, Tq, d)
        Ff = p(Of.reshape(B * Tq, d) @ Wo.T, sf).reshape(B, Tq, d)
        # The mirror must reproduce the integer forward exactly, or the oracle
        # is wrong rather than the gradient.
        assert torch.equal(Ff.detach().to(torch.int64), f), "float mirror diverged from integer forward"
        (Ff * gbm.double()).sum().backward()
        ref = {"gW_q": Wq.grad, "gW_k": Wk.grad, "gW_v": Wv.grad, "gW_o": Wo.grad}
        got = {"gW_q": gws[0][0], "gW_k": gws[1][0], "gW_v": gws[2][0], "gW_o": gws[3][0]}
        rows.append({"adjoint_bits": gbits, **{k: round(cos(got[k], ref[k]), 4) for k in ref}})
    # Sweep the adjoint mantissa width: too few bits loses the gradient, too
    # many overflows int64 inside the adjoint GEMMs. Record the usable window.
    sweep = []
    f, cache = layer.forward(a, ctx)
    gen = torch.Generator().manual_seed(7)
    gbm = torch.randint(-(1 << 20), (1 << 20) + 1, f.shape, dtype=torch.int64, generator=gen)
    for hb in hb_sweep:
        _q, _kv, gws = layer.backward(SharedGrad(gbm, 0), cache, hb)
        mx = max(int(g[0].abs().max()) for g in gws)
        sweep.append({"hidden_bits": hb, "max_abs_grad_mantissa": mx,
                      "near_int64_limit": mx > (1 << 58)})
    ok = all(r[k] >= 0.98 for r in rows for k in ("gW_q", "gW_k", "gW_v", "gW_o"))
    return {"gate": 5, "production_hidden_bits": HB, "rows": rows,
            "adjoint_width_sweep": sweep,
            "note": "adjoint magnitudes above ~2**24 overflow int64 inside the "
                    "adjoint GEMMs; the hinge adjoint is bounded by 2**hidden_bits "
                    "so production never enters that regime",
            "passed": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gate", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--query-anchor", action="store_true",
                    help="run the gates with cross-attention BOS query anchoring on")
    ap.add_argument("--out", default="jako_gates.json")
    a = ap.parse_args()
    M.CROSS_QUERY_ANCHOR = bool(a.query_anchor)
    torch.set_num_threads(a.threads)

    results = []
    fns = {2: gate2_inverse, 3: gate3_cachefree, 4: gate4_step_modes,
           5: gate5_attention_cosine}
    for g in (sorted(fns) if a.all else a.gate):
        r = fns[g]()
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    ok = all(r["passed"] for r in results)
    print(f"\nALL GATES PASSED: {ok}")
    json.dump({"gates": results, "all_passed": ok}, open(a.out, "w"), indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
