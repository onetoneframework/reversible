#!/usr/bin/env python3
"""
Exactness gate for the chunked / sparse output head.

The claim under test is that walking the vocabulary in chunks and taking the
sparse hinge backward produces *bit-identical* results to materialising the
whole B x T x V logit tensor and running dense_linear_backward over it. Nothing
here is allowed to pass on a tolerance except the one float diagnostic
(logit_std), which is a reduction over ~5 million doubles in a different order
and is reported with its relative difference rather than asserted equal.

  python reversible_jako_headgate.py --chunks 256 512 1024 2048
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


def build(vocab, d, ffn, k_enc, k_dec, seed, enc_pattern, dec_pattern):
    cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
    R.CFG = cfg
    return M.JaKoMT(vocab, d, ffn, k_enc, k_dec, cfg, seed,
                    enc_pattern=enc_pattern, dec_pattern=dec_pattern)


def run(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    model = build(a.vocab, a.dim, a.ffn_hidden, a.k_enc, a.k_dec, a.seed,
                  a.enc_pattern, a.dec_pattern)
    g = torch.Generator().manual_seed(a.seed)
    src = torch.randint(4, a.vocab, (a.batch, a.max_len), generator=g)
    tin = torch.randint(4, a.vocab, (a.batch, a.max_len), generator=g)
    tout = torch.randint(4, a.vocab, (a.batch, a.max_len), generator=g)
    # Real batches are padded; the PAD path is the one that decides `active`.
    tout[:, a.max_len // 2:] = M.PAD

    mem, _ec, _ei, _et = model.encode(src, cached=False)
    dec_out, _dc, _di, _dt = model.decode(tin, mem, src, cached=False)

    # Reference: full logits, dense hinge, dense_linear_backward.
    logits, s_out = model.logits(dec_out)
    g_log, ref_stats = M.hinge_grad(logits, tout, a.hidden_bits)
    B, T, V = logits.shape
    W2 = 2 * model.d
    g2 = M.SharedGrad(g_log.m.reshape(B * T, V), g_log.e)
    ref_gx, ref_gw = M.dense_linear_backward(
        g2, dec_out.reshape(B * T, W2), model.Wout, s_out, a.hidden_bits, 30,
        model.cfg.accum_bits)

    rows = []
    ok_all = True
    for chunk in a.chunks:
        tv, bv, bi, sh, st = M.chunked_head_forward(model, dec_out, tout,
                                                    a.hidden_bits, chunk)
        gx, gw = M.sparse_head_backward(model, dec_out, tout, tv, bv, bi, sh,
                                        a.hidden_bits)

        flat = logits.reshape(B * T, V)
        idx = torch.arange(B * T)
        masked = flat.clone()
        masked[idx, tout.reshape(-1)] = torch.iinfo(torch.int64).min // 4
        ref_j = masked.argmax(dim=1)

        r = {
            "chunk": chunk,
            "shift_equal": bool(torch.equal(sh, s_out)),
            "true_logit_equal": bool(torch.equal(tv, flat[idx, tout.reshape(-1)])),
            "best_wrong_val_equal": bool(torch.equal(bv, flat[idx, ref_j])),
            "best_wrong_idx_equal": bool(torch.equal(bi, ref_j)),
            "grad_input_mantissa_equal": bool(torch.equal(gx.m, ref_gx.m)),
            "grad_input_exponent_equal": bool(gx.e == ref_gx.e),
            "grad_weight_mantissa_equal": bool(torch.equal(gw[0], ref_gw[0])),
            "grad_weight_exponent_equal": bool(gw[1] == ref_gw[1]),
            "stats_int_equal": all(st[k] == ref_stats[k] for k in
                                   ("tokens", "active", "hinge_sum", "correct")),
            "margin_mean_equal": st["margin_mean"] == ref_stats["margin_mean"],
            "logit_std_rel_diff": abs(st["logit_std"] - ref_stats["logit_std"])
            / max(1e-12, abs(ref_stats["logit_std"])),
        }
        r["passed"] = all(v for k, v in r.items()
                          if k not in ("chunk", "logit_std_rel_diff", "passed"))
        ok_all = ok_all and r["passed"] and r["logit_std_rel_diff"] < 1e-9
        rows.append(r)
        print(json.dumps(r), flush=True)

    # Whole-step equivalence: every gradient in the model, not just the head.
    ga, sa = M.forward_backward(model, src, tin, tout, a.hidden_bits,
                                cached=False, chunk_head=0)
    gb, sb = M.forward_backward(model, src, tin, tout, a.hidden_bits,
                                cached=False, chunk_head=a.chunks[0])
    diffs = [i for i in range(len(ga))
             if not (torch.equal(ga[i][0], gb[i][0]) and ga[i][1] == gb[i][1])]
    step = {"gate": "full_step",
            "tensors": len(ga),
            "mismatched_tensors": diffs,
            "stats_int_equal": all(sa[k] == sb[k] for k in
                                   ("tokens", "active", "hinge_sum", "correct")),
            "passed": not diffs and all(sa[k] == sb[k] for k in
                                        ("tokens", "active", "hinge_sum", "correct"))}
    print(json.dumps(step), flush=True)
    ok_all = ok_all and step["passed"]

    # Peak logit-side allocation, measured rather than argued.
    from reversible_integer_memory_bench import LiveByteTracker
    peaks = {}
    with LiveByteTracker() as t:
        M.forward_backward(model, src, tin, tout, a.hidden_bits, cached=False,
                           chunk_head=0)
        peaks["dense_peak_bytes"] = t.peak
    with LiveByteTracker() as t:
        M.forward_backward(model, src, tin, tout, a.hidden_bits, cached=False,
                           chunk_head=a.chunks[0])
        peaks["chunked_peak_bytes"] = t.peak
    peaks["chunk"] = a.chunks[0]
    peaks["ratio"] = round(peaks["dense_peak_bytes"]
                           / max(1, peaks["chunked_peak_bytes"]), 3)
    print(json.dumps(peaks), flush=True)

    print("\nHEAD GATE PASSED: " + str(ok_all))
    json.dump({"rows": rows, "full_step": step, "peak": peaks, "passed": ok_all},
              open(a.out, "w"), indent=2)
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--vocab", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ffn-hidden", type=int, default=256)
    ap.add_argument("--k-enc", type=int, default=2)
    ap.add_argument("--k-dec", type=int, default=4)
    ap.add_argument("--enc-pattern", default="af")
    ap.add_argument("--dec-pattern", default="ccaf")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--hidden-bits", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--query-anchor", action="store_true",
                    help="run the gate with cross-attention BOS query anchoring on")
    ap.add_argument("--out", default="jako_headgate.json")
    a = ap.parse_args()
    M.CROSS_QUERY_ANCHOR = bool(a.query_anchor)
    sys.exit(run(a))


if __name__ == "__main__":
    main()
