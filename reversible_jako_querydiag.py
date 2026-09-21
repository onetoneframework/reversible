#!/usr/bin/env python3
"""
Why does the cross-attention row not depend on the decoder position?

crossdiag found the concentration statistics identical across decoder
positions, which means the attention row is a function of the source alone.
This asks where the query goes missing: is Q the same at every position, is
Q.K^T dominated by a component of K that every query shares, or does the
per-row 7-bit (TB+2) pow2norm of the scores quantise the query-specific part
away? All three are measured from the cached Q, K, S of the real forward.

  python reversible_jako_querydiag.py --ckpt jako_50k_best.pt --n 128
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
import reversible_jako_data as D
from reversible_integer_final import saturate, pow2norm

TB = R.TB


def pairwise_cos_within(rows, mask_rows=None):
    """Mean pairwise cosine between the rows of one (Tq, X) matrix."""
    x = rows.double()
    if mask_rows is not None:
        x = x * mask_rows
    n = x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    G = (x / n) @ (x / n).T
    k = G.shape[0]
    return float((G.sum() - G.diagonal().sum()) / max(1, k * (k - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="jako_50k_best.pt")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--corpus", default="mixed")
    ap.add_argument("--vocab-size", type=int, default=5000)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--max-tok", type=int, default=30)
    ap.add_argument("--base-pairs", type=int, default=25000)
    ap.add_argument("--subs-pairs", type=int, default=25000)
    ap.add_argument("--base-pool", type=int, default=100000)
    ap.add_argument("--subs-pool", type=int, default=100000)
    ap.add_argument("--min-char-count", type=int, default=20)
    ap.add_argument("--out", default="jako_querydiag.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    model, tok, ck = M.load_ckpt(a.ckpt)
    rec = {}
    for i, f in enumerate(model.dec.fs):
        if f.kind != "attn" or not f.cross:
            continue

        def wrapped(x, ctx, _f=f, _i=i):
            f2, cache = M.AttnF.forward(_f, x, ctx)
            (a2, k2, Q, K, V, S, Aw, o2, sq, sk, sv, ss, so, sf, mask, B, Tq, Tk,
             _anchored, _sn) = cache
            rec.setdefault(_i, []).append((Q.clone(), K.clone(), S.clone(), mask.clone(),
                                           a2.reshape(B, Tq, -1).clone()))
            return f2, cache
        f.forward = wrapped

    ds = D.prepare(a.corpus, 0, a.vocab_size, a.max_len, seed=0, max_tok=a.max_tok,
                   tok_mode=ck.get("tok_mode", "char"), base_pairs=a.base_pairs,
                   subs_pairs=a.subs_pairs, base_pool=a.base_pool,
                   subs_pool=a.subs_pool, min_char_count=a.min_char_count)
    n = min(a.n, len(ds["valid"][0]))
    src = torch.tensor(ds["valid"][0][:n], dtype=torch.int64)
    tin = torch.tensor(ds["valid"][1][:n], dtype=torch.int64)
    for st in range(0, n, a.batch):
        mem, _c, _i, _x = model.encode(src[st:st + a.batch])
        model.decode(tin[st:st + a.batch], mem, src[st:st + a.batch])

    report = {"epoch": ck.get("epoch"), "n": n}
    print("ckpt %s  epoch %s" % (a.ckpt, ck.get("epoch")))
    print("dec  |  Q-row cos  |  raw-score row cos  |  S-row cos (after 7-bit)  |  "
          "K common-mode  |  common-term share of raw var  |  raw bits (row max)  | shift")
    for i, recs in rec.items():
        qcos, rawcos, scos, kcm, share, bits, shifts = [], [], [], [], [], [], []
        acos, acm, amag = [], [], []
        for Q, K, S, mask, A2 in recs:
            B, Tq, d = Q.shape
            raw = saturate(R.imatmul(Q, K.transpose(1, 2)), model.cfg.accum_bits)
            _s2, ss = M.masked_pow2norm(raw, mask, TB + 2)
            shifts.append(ss.double().mean())
            for b in range(B):
                valid = mask[b, 0].bool()
                Ts = int(valid.sum())
                if Ts < 2:
                    continue
                Kb = K[b, valid].double()                        # (Ts, d)
                Qb = Q[b].double()                               # (Tq, d)
                Ab = A2[b].double()                              # (Tq, d) query-side state
                acos.append(pairwise_cos_within(Ab))
                acm.append(float(Ab.mean(0).norm() / Ab.norm(dim=1).mean().clamp(min=1e-12)))
                amag.append(float(Ab.abs().mean()))
                qcos.append(pairwise_cos_within(Qb))
                rawb = raw[b][:, valid]                          # (Tq, Ts)
                rawcos.append(pairwise_cos_within(rawb))
                scos.append(pairwise_cos_within(S[b][:, valid]))
                kmean = Kb.mean(0)
                kcm.append(float(kmean.norm() / Kb.norm(dim=1).mean().clamp(min=1e-12)))
                # Decompose raw = Qmean.K_j (common, same for every query) + rest.
                qmean = Qb.mean(0)
                common = (Kb @ qmean)                            # (Ts,)
                tot_var = float(rawb.double().var())
                rest = rawb.double() - common[None, :]
                share.append(1.0 - float(rest.var()) / max(tot_var, 1e-12))
                bits.append(float(torch.log2(rawb.double().abs().amax(dim=1).clamp(min=1)).mean()))
        row = {
            "a_row_cos": round(sum(acos) / len(acos), 4),
            "a_common_mode": round(sum(acm) / len(acm), 4),
            "a_mean_abs": round(sum(amag) / len(amag), 1),
            "q_row_cos": round(sum(qcos) / len(qcos), 4),
            "raw_score_row_cos": round(sum(rawcos) / len(rawcos), 4),
            "S_row_cos_after_7bit": round(sum(scos) / len(scos), 4),
            "K_common_mode": round(sum(kcm) / len(kcm), 4),
            "common_term_share_of_raw_var": round(sum(share) / len(share), 4),
            "raw_row_max_bits": round(sum(bits) / len(bits), 2),
            "mean_shift": round(float(sum(shifts) / len(shifts)), 2),
        }
        report["dec%d" % i] = row
        print("dec%-2d %10.3f %19.3f %25.3f %15.3f %30.3f %20.2f %7.2f" % (
            i, row["q_row_cos"], row["raw_score_row_cos"], row["S_row_cos_after_7bit"],
            row["K_common_mode"], row["common_term_share_of_raw_var"],
            row["raw_row_max_bits"], row["mean_shift"]))
    json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
