#!/usr/bin/env python3
"""
Causality gates for the reversible-integer JA->KO decoder.

These run before the correctness gates, because a model that leaks future
information is optimising an objective that free-running decoding cannot serve,
and no amount of exactness makes that useful.

  A  future-target mutation. Change the target tokens at and after position
     `cut` and every causally-allowed quantity before `cut` must be
     bit-identical: decoder block inputs and outputs, Q, the allowed columns of
     the raw scores, the per-row shift, the normalised scores, the attention
     weights, and the final logits. The full raw score matrix is allowed to
     differ in its masked-out columns -- that is what the mask is for. Logits at
     and after `cut` must actually change, otherwise the mutation was a no-op
     and the gate proves nothing.

  B  padding invariance. The same sentence padded to different lengths must
     produce bit-identical encoder memory, decoder state and logits on its real
     tokens. Padding is not content.

  C  padding-content mutation. Writing garbage into the encoder memory rows
     that the cross mask excludes must leave every valid output bit-identical.

Gate A is what caught the original defect: `pow2norm` took its row maximum over
scores that were about to be masked away, so the shift applied to the surviving
scores at position t depended on target tokens after t.

  python reversible_jako_causalgate.py --ckpt jako_50k_best.pt
"""

import argparse
import json
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_integer_final as R
import reversible_jako_mt as M


def capture(model):
    """Record every attention block's intermediates, keyed by block index."""
    rec = {}
    for tag, stack in (("dec", model.dec), ("enc", model.enc)):
        for i, f in enumerate(stack.fs):
            if f.kind == "attn":
                def wa(x, ctx, _f=f, _k="%s%d" % (tag, i)):
                    f2, cache = M.AttnF.forward(_f, x, ctx)
                    (a2, k2, Q, K, V, S, Aw, o2, sq, sk, sv, ss, so, sf,
                     mk, B, Tq, Tk, anchored, _sn) = cache
                    raw = R.saturate(R.imatmul(Q, K.transpose(1, 2)), _f.cfg.accum_bits)
                    rec[_k] = {"in": x.clone(), "Q": Q.clone(), "K": K.clone(),
                               "V": V.clone(), "raw": raw, "mask": mk.clone(),
                               "shift": ss.reshape(B, Tq, 1).clone(),
                               "S": S.clone(), "Aw": Aw.clone(), "out": f2.clone(),
                               "cross": _f.cross}
                    return f2, cache
                f.forward = wa
            else:
                def wf(x, ctx, _f=f, _k="%s%d" % (tag, i)):
                    f2, cache = M.FFNF.forward(_f, x, ctx)
                    rec[_k] = {"in": x.clone(), "out": f2.clone()}
                    return f2, cache
                f.forward = wf
    return rec


def run(model, src, tin):
    rec = capture(model)
    mem, _c, _i, _x = model.encode(src)
    dec, _c2, _i2, _x2 = model.decode(tin, mem, src)
    lg, _s = model.logits(dec)
    return rec, mem, dec, lg


def build(vocab, dim, ffn, k_enc, k_dec, seed, enc_p, dec_p, positional=0):
    cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
    R.CFG = cfg
    return M.JaKoMT(vocab, dim, ffn, k_enc, k_dec, cfg, seed,
                    enc_pattern=enc_p, dec_pattern=dec_p, positional=positional)


def gate_a(model, vocab, L, cut, seed=0, batch=4):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(4, vocab, (batch, L), generator=g)
    src[:, -3:] = M.PAD
    ta = torch.randint(4, vocab, (batch, L), generator=g)
    ta[:, 0] = M.BOS
    tb = ta.clone()
    tb[:, cut:] = torch.randint(4, vocab, (batch, L - cut), generator=g)
    if torch.equal(ta, tb):
        return {"gate": "A", "cut": cut, "passed": False, "why": "mutation was a no-op"}

    ra, _ma, da, lga = run(model, src, ta)
    rb, _mb, db, lgb = run(model, src, tb)

    bad = []
    for k in ra:
        A, B = ra[k], rb[k]
        for field in ("in", "out", "Q", "K", "V", "shift", "S", "Aw"):
            if field not in A:
                continue
            x, y = A[field], B[field]
            if x.dim() == 3 and x.shape[1] >= cut and k.startswith("dec"):
                x, y = x[:, :cut], y[:, :cut]
            if not torch.equal(x, y):
                bad.append("%s.%s" % (k, field))
        # Raw scores: only the columns the mask keeps have to agree.
        if "raw" in A and k.startswith("dec"):
            mk = A["mask"][:, :cut].to(torch.bool)
            if not torch.equal(A["raw"][:, :cut][mk], B["raw"][:, :cut][mk]):
                bad.append("%s.raw(allowed)" % k)
    state_ok = bool(torch.equal(da[:, :cut], db[:, :cut]))
    logits_ok = bool(torch.equal(lga[:, :cut], lgb[:, :cut]))
    changed = bool(not torch.equal(lga[:, cut:], lgb[:, cut:]))
    return {"gate": "A", "cut": cut, "leaked": bad,
            "decoder_state_before_cut_identical": state_ok,
            "logits_before_cut_identical": logits_ok,
            "logits_after_cut_changed": changed,
            "passed": not bad and state_ok and logits_ok and changed}


def gate_b(model, vocab, seed=0):
    """Same sentence, different padded lengths -> identical on real tokens."""
    g = torch.Generator().manual_seed(seed)
    n = 7
    toks = torch.randint(4, vocab, (1, n), generator=g)
    tgt = torch.randint(4, vocab, (1, 5), generator=g)
    tgt[0, 0] = M.BOS
    outs = {}
    for L in (10, 16, 24):
        src = torch.full((1, L), M.PAD, dtype=torch.int64)
        src[0, :n] = toks[0]
        tin = torch.full((1, L), M.PAD, dtype=torch.int64)
        tin[0, :5] = tgt[0]
        _r, mem, dec, lg = run(model, src, tin)
        outs[L] = (mem[:, :n].clone(), dec[:, :5].clone(), lg[:, :5].clone())
    ref = outs[10]
    bad = []
    for L, v in outs.items():
        if L == 10:
            continue
        for name, i in (("memory", 0), ("decoder_state", 1), ("logits", 2)):
            if not torch.equal(ref[i], v[i]):
                bad.append("pad%d.%s" % (L, name))
    return {"gate": "B", "padded_lengths": sorted(outs), "differs": bad,
            "passed": not bad}


def gate_c(model, vocab, L=12, seed=0):
    """Garbage in masked-out encoder memory rows must not reach any output."""
    g = torch.Generator().manual_seed(seed)
    n = 8
    src = torch.full((2, L), M.PAD, dtype=torch.int64)
    src[:, :n] = torch.randint(4, vocab, (2, n), generator=g)
    tin = torch.randint(4, vocab, (2, L), generator=g)
    tin[:, 0] = M.BOS

    mem, _c, _i, _x = model.encode(src)
    dec_a, _c2, _i2, _x2 = model.decode(tin, mem, src)
    lg_a, _s = model.logits(dec_a)

    mem2 = mem.clone()
    noise = torch.randint(-(1 << 14), 1 << 14, mem2[:, n:].shape,
                          dtype=torch.int64, generator=g)
    mem2[:, n:] = noise
    dec_b, _c3, _i3, _x3 = model.decode(tin, mem2, src)
    lg_b, _s2 = model.logits(dec_b)
    return {"gate": "C", "garbage_rows": int(L - n),
            "decoder_state_identical": bool(torch.equal(dec_a, dec_b)),
            "logits_identical": bool(torch.equal(lg_a, lg_b)),
            "passed": bool(torch.equal(lg_a, lg_b) and torch.equal(dec_a, dec_b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=600)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ffn-hidden", type=int, default=128)
    ap.add_argument("--k-enc", type=int, default=2)
    ap.add_argument("--k-dec", type=int, default=8)
    ap.add_argument("--enc-pattern", default="af")
    ap.add_argument("--dec-pattern", default="ccafccaf")
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--query-anchor", action="store_true")
    ap.add_argument("--positional", type=int, default=0)
    ap.add_argument("--out", default="jako_causalgate.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    M.CROSS_QUERY_ANCHOR = bool(a.query_anchor)

    model = build(a.vocab, a.dim, a.ffn_hidden, a.k_enc, a.k_dec, 0,
                  a.enc_pattern, a.dec_pattern, a.positional)
    rows = []
    print("--- A: future target tokens must not reach anything before the cut ---")
    for cut in (1, 3, 6):
        r = gate_a(model, a.vocab, a.max_len, cut)
        rows.append(r)
        print("  cut=%d  leaked=%s  state=%s logits=%s after-changed=%s  %s"
              % (cut, r.get("leaked"), r.get("decoder_state_before_cut_identical"),
                 r.get("logits_before_cut_identical"), r.get("logits_after_cut_changed"),
                 "PASS" if r["passed"] else "FAIL"))

    print("--- B: padding length must not change a sentence's own outputs ---")
    rb = gate_b(model, a.vocab)
    rows.append(rb)
    print("  lengths %s  differs=%s  %s" % (rb["padded_lengths"], rb["differs"],
                                            "PASS" if rb["passed"] else "FAIL"))

    print("--- C: garbage in masked encoder memory must not reach the output ---")
    rc = gate_c(model, a.vocab)
    rows.append(rc)
    print("  state=%s logits=%s  %s" % (rc["decoder_state_identical"],
                                        rc["logits_identical"],
                                        "PASS" if rc["passed"] else "FAIL"))

    ok = all(r["passed"] for r in rows)
    print("\nCAUSALITY GATES PASSED: %s   (query_anchor=%s)" % (ok, a.query_anchor))
    json.dump({"rows": rows, "query_anchor": a.query_anchor, "passed": ok},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
