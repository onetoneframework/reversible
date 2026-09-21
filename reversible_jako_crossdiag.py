#!/usr/bin/env python3
"""
Is cross-attention carrying source *content*, or just a source *summary*?

The source-ablation result (shuffling sources between examples leaves
teacher-forced accuracy unchanged, zeroing the memory halves it) says the
decoder uses the fact that a memory exists but not which memory it is. The
hypothesis under test: the attention rows are so flat that every query reads
roughly the mean of the encoder memory -- a bag-of-source signal that varies
per sentence but cannot align a target word to a source word.

Four measurements on one checkpoint, per cross-attention block:

  1. concentration of the real attention rows. The weights here are signed and
     not row-normalised (leaky score map, no softmax), so Shannon entropy is
     taken over |a_i| / sum|a_i| and reported alongside max share, top-k share,
     effective support exp(H), coefficient of variation and negative-mass share.
  2. the same, split by decoder position: 0 / 1-3 / 4-7 / 8+.
  3. counterfactual attention. The block's own output is recomputed with its
     row replaced by (a) a uniform row carrying the same total weight, and
     (b) the same weights permuted among the valid source positions. The
     forward semantics are untouched; only the diagnostic re-runs the block's
     tail, and it first proves the re-run is bit-identical on the real weights.
     Reported: change in that block's output, in the final logits, in
     teacher-forced accuracy, and in position-0 predictions -- one block at a
     time and all blocks together.
  4. diversity of the attention pattern across sources at a fixed decoder
     position: where the largest weight lands (first token / last token /
     elsewhere) and how alike the binned patterns are across sentences.

  python reversible_jako_crossdiag.py --ckpt jako_50k_best.pt --n 256
"""

import argparse
import json
import math
import sys
import time

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_integer_final as R
import reversible_jako_mt as M
import reversible_jako_data as D
import reversible_jako_decode as DEC
from reversible_integer_final import pow2norm, saturate

TB = R.TB
GROUPS = (("pos0", 0, 1), ("pos1-3", 1, 4), ("pos4-7", 4, 8), ("pos8+", 8, 10 ** 6))


# -----------------------------------------------------------------------------
# Hooking the cross blocks
# -----------------------------------------------------------------------------
CTRL = {"mode": "record", "targets": None, "rec": None, "perm_seed": 0}


def block_tail(f, Aw, V, B, Tq, d):
    """The two lines of AttnF.forward after the weights are formed."""
    o2, _so = pow2norm(saturate(R.imatmul(Aw, V), f.cfg.accum_bits).reshape(B * Tq, d),
                       TB + 2)
    f2, _sf = pow2norm(saturate(R.imatmul(o2, f.Wo.T), f.cfg.accum_bits), TB + 2)
    return f2.reshape(B, Tq, d)


def uniform_rows(Aw, mask):
    """Each row keeps its total weight, spread evenly over the valid positions.
    Integer division truncates toward zero, as everywhere else in the model."""
    tot = (Aw * mask).sum(dim=-1, keepdim=True)
    cnt = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean = torch.sign(tot) * (torch.abs(tot) // cnt)
    return mean.expand_as(Aw) * mask


def permuted_rows(Aw, mask, seed):
    """Each row's weights shuffled among its own valid positions."""
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros_like(Aw)
    B, Tq, Tk = Aw.shape
    for b in range(B):
        valid = torch.nonzero(mask[b, 0]).reshape(-1)     # same for every query
        if valid.numel() <= 1:
            out[b] = Aw[b] * mask[b]
            continue
        for q in range(Tq):
            p = valid[torch.randperm(valid.numel(), generator=g)]
            out[b, q, valid] = Aw[b, q, p]
    return out


def install(model):
    """Wrap every cross-attention block's forward; returns their indices."""
    idx = []
    for i, f in enumerate(model.dec.fs):
        if f.kind != "attn" or not f.cross:
            continue
        idx.append(i)

        def wrapped(a, ctx, _f=f, _i=i):
            f2, cache = M.AttnF.forward(_f, a, ctx)
            (a2, k2, Q, K, V, S, Aw, o2, sq, sk, sv, ss, so, sf, mask, B, Tq, Tk) = cache
            mode = CTRL["mode"]
            hit = CTRL["targets"] is None or _i in CTRL["targets"]
            if mode == "record" and CTRL["rec"] is not None:
                CTRL["rec"].setdefault(_i, []).append((Aw.clone(), mask.clone()))
                return f2, cache
            if mode == "verify":
                f2b = block_tail(_f, Aw, V, B, Tq, _f.d)
                assert torch.equal(f2b, f2), "tail re-run diverged from the block"
                return f2, cache
            if hit and mode in ("uniform", "permute"):
                Acf = (uniform_rows(Aw, mask) if mode == "uniform"
                       else permuted_rows(Aw, mask, CTRL["perm_seed"] + _i))
                f2cf = block_tail(_f, Acf, V, B, Tq, _f.d)
                if CTRL["rec"] is not None:
                    num = float((f2cf - f2).double().norm())
                    den = float(f2.double().norm()) or 1.0
                    CTRL["rec"].setdefault(("delta", _i), []).append(num / den)
                return f2cf, cache
            return f2, cache

        f.forward = wrapped
    return idx


# -----------------------------------------------------------------------------
# Statistics on the real rows
# -----------------------------------------------------------------------------
def row_stats(Aw, mask):
    """Per-row concentration numbers. Returns dict of (B,Tq) tensors."""
    ab = Aw.abs().double() * mask
    tot = ab.sum(dim=-1)
    ts = mask.sum(dim=-1).double()
    p = ab / tot.clamp(min=1e-12).unsqueeze(-1)
    logp = torch.where(p > 0, torch.log(p.clamp(min=1e-300)), torch.zeros_like(p))
    H = -(p * logp).sum(dim=-1)
    Hn = torch.where(ts > 1, H / torch.log(ts.clamp(min=2)), torch.zeros_like(H))
    srt = p.sort(dim=-1, descending=True).values
    signed = Aw.double() * mask
    mean_abs = tot / ts.clamp(min=1)
    var = (((signed - signed.sum(-1, keepdim=True) / ts.clamp(min=1).unsqueeze(-1)) ** 2)
           * mask).sum(-1) / ts.clamp(min=1)
    neg = ((Aw < 0).double() * ab).sum(-1) / tot.clamp(min=1e-12)
    am = ab.argmax(dim=-1)
    return {
        "zero_row": (tot == 0).double(),
        "Hn": Hn, "eff": torch.exp(H), "max": srt[..., 0],
        "top2": srt[..., :2].sum(-1), "top4": srt[..., :4].sum(-1),
        "cv": torch.sqrt(var) / mean_abs.clamp(min=1e-12),
        "neg": neg, "uniform_max": 1.0 / ts.clamp(min=1),
        "argmax_first": (am == 0).double(),
        "argmax_last": (am == (ts - 1).long()).double(),
        "argmax_rel": am.double() / (ts - 1).clamp(min=1),
        "ts": ts, "p": p,
    }


def summarise(recs):
    """recs: list of (Aw, mask). Aggregates row_stats by decoder position group."""
    per = {g: {} for g, _a, _b in GROUPS}
    for Aw, mask in recs:
        st = row_stats(Aw, mask)
        Tq = Aw.shape[1]
        for g, lo, hi in GROUPS:
            hi = min(hi, Tq)
            if lo >= hi:
                continue
            sel = st["zero_row"][:, lo:hi] == 0          # rows with any mass
            for k in ("Hn", "eff", "max", "top2", "top4", "cv", "neg", "uniform_max",
                      "argmax_first", "argmax_last"):
                v = st[k][:, lo:hi][sel]
                per[g].setdefault(k, []).append(v)
            per[g].setdefault("zero_row", []).append(st["zero_row"][:, lo:hi].reshape(-1))
    out = {}
    for g, d in per.items():
        if not d:
            continue
        out[g] = {k: round(float(torch.cat(v).mean()), 4) for k, v in d.items() if v}
        out[g]["rows"] = int(sum(v.numel() for v in d["zero_row"]))
    return out


def pattern_diversity(recs, pos, bins=8):
    """Mean pairwise cosine between binned |a| patterns across sequences at one
    decoder position. 1.0 means every sentence attends the same way."""
    vecs = []
    for Aw, mask in recs:
        if Aw.shape[1] <= pos:
            continue
        st = row_stats(Aw[:, pos:pos + 1], mask[:, pos:pos + 1])
        p = st["p"][:, 0]                        # (B, Tk)
        ts = st["ts"][:, 0]
        for b in range(p.shape[0]):
            n = int(ts[b])
            if n < 2 or float(p[b].sum()) == 0:
                continue
            rel = torch.arange(n, dtype=torch.float64) / max(1, n - 1)
            bi = torch.clamp((rel * bins).long(), max=bins - 1)
            v = torch.zeros(bins, dtype=torch.float64).index_add_(0, bi, p[b, :n])
            vecs.append(v / (v.norm() + 1e-12))
    if len(vecs) < 2:
        return None
    Vm = torch.stack(vecs)
    G = Vm @ Vm.T
    n = G.shape[0]
    off = (G.sum() - G.diagonal().sum()) / (n * (n - 1))
    return {"sequences": n, "mean_pairwise_cosine": round(float(off), 4),
            "mean_binned_pattern": [round(float(x), 3) for x in Vm.mean(0)]}


# -----------------------------------------------------------------------------
# Teacher-forced pass and comparison
# -----------------------------------------------------------------------------
def tf_logits(model, mem, src, tin, batch):
    outs = []
    for st in range(0, src.shape[0], batch):
        dec, _c, _i, _x = model.decode(tin[st:st + batch], mem[st:st + batch],
                                       src[st:st + batch])
        lg, _s = model.logits(dec)
        outs.append(lg)
    return torch.cat(outs)


def compare(lg_real, lg_cf, tout):
    notpad = tout != M.PAD
    pr, pc = lg_real.argmax(-1), lg_cf.argmax(-1)
    d = (lg_cf - lg_real).double().abs()
    return {
        "tf_acc": round(100.0 * float(((pc == tout) & notpad).sum()) / float(notpad.sum()), 3),
        "pos0_acc": round(100.0 * float((pc[:, 0] == tout[:, 0]).sum()) / tout.shape[0], 3),
        "argmax_changed_pct": round(100.0 * float(((pr != pc) & notpad).sum())
                                    / float(notpad.sum()), 2),
        "pos0_argmax_changed_pct": round(100.0 * float((pr[:, 0] != pc[:, 0]).sum())
                                         / tout.shape[0], 2),
        "rel_logit_change": round(float(d[notpad].mean() / lg_real.double().abs()[notpad].mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="jako_50k_best.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="jako_crossdiag.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()

    model, tok, ck = M.load_ckpt(a.ckpt)
    cross_idx = install(model)
    print("checkpoint %s  epoch %s  cross blocks at dec indices %s  SCORE_LEAK=%s"
          % (a.ckpt, ck.get("epoch"), cross_idx, __import__("reversible_arch").SCORE_LEAK),
          flush=True)

    ds = D.prepare(a.corpus, 0, a.vocab_size, a.max_len, seed=a.seed,
                   max_tok=a.max_tok, tok_mode=ck.get("tok_mode", "char"),
                   base_pairs=a.base_pairs, subs_pairs=a.subs_pairs,
                   base_pool=a.base_pool, subs_pool=a.subs_pool,
                   min_char_count=a.min_char_count)
    n = min(a.n, len(ds["valid"][0]))
    src = torch.tensor(ds["valid"][0][:n], dtype=torch.int64)
    tin = torch.tensor(ds["valid"][1][:n], dtype=torch.int64)
    tout = torch.tensor(ds["valid"][2][:n], dtype=torch.int64)
    print("valid sequences %d  (data ready in %.0fs)" % (n, time.time() - t0), flush=True)

    mems = []
    for st in range(0, n, a.batch):
        m, _c, _i, _x = model.encode(src[st:st + a.batch])
        mems.append(m)
    mem = torch.cat(mems)

    report = {"ckpt": a.ckpt, "epoch": ck.get("epoch"), "n": n, "cross_blocks": cross_idx}

    # -- 0. prove the re-run tail is faithful, then record the real rows -------
    CTRL.update(mode="verify", targets=None, rec=None)
    _ = tf_logits(model, mem, src[:a.batch], tin[:a.batch], a.batch)
    print("tail re-run verified bit-identical on real weights", flush=True)

    rec = {}
    CTRL.update(mode="record", targets=None, rec=rec)
    lg_real = tf_logits(model, mem, src, tin, a.batch)
    base = compare(lg_real, lg_real, tout)
    report["real"] = {"tf_acc": base["tf_acc"], "pos0_acc": base["pos0_acc"]}
    print("real: tf_acc %.3f%%  pos0_acc %.3f%%" % (base["tf_acc"], base["pos0_acc"]))

    # -- 1 & 2. concentration by block and decoder position ----------------------
    print("\n--- 1/2. real attention concentration (|a|/sum|a|) ---")
    print("  %-6s %-7s %6s %7s %7s %7s %7s %7s %6s %6s %7s %7s %6s" % (
        "block", "pos", "rows", "H/logT", "eff", "max", "1/T", "top2", "top4",
        "cv", "neg", "am@0", "am@T"))
    report["concentration"] = {}
    for i in cross_idx:
        s = summarise(rec[i])
        report["concentration"][str(i)] = s
        for g, _lo, _hi in GROUPS:
            if g not in s:
                continue
            r = s[g]
            print("  dec%-3d %-7s %6d %7.3f %7.2f %7.3f %7.3f %7.3f %7.3f %6.2f %6.3f %7.3f %7.3f" % (
                i, g, r["rows"], r["Hn"], r["eff"], r["max"], r["uniform_max"],
                r["top2"], r["top4"], r["cv"], r["neg"], r["argmax_first"], r["argmax_last"]))
        print("         zero-mass rows: %s" % {g: s[g]["zero_row"] for g in s})

    # -- 4. pattern diversity across sources ------------------------------------
    print("\n--- 4. pattern diversity across sources (binned |a|, 8 bins) ---")
    report["diversity"] = {}
    for i in cross_idx:
        for pos in (0, 2, 6):
            dv = pattern_diversity(rec[i], pos)
            if dv:
                report["diversity"]["dec%d_pos%d" % (i, pos)] = dv
                print("  dec%-3d pos%-2d  n=%-4d pairwise-cos %.3f   mean pattern %s" % (
                    i, pos, dv["sequences"], dv["mean_pairwise_cosine"], dv["mean_binned_pattern"]))

    # -- 3. counterfactual attention --------------------------------------------
    print("\n--- 3. counterfactual attention ---")
    print("  %-9s %-10s %10s %8s %8s %9s %10s %9s" % (
        "mode", "blocks", "blk_out_d", "tf_acc", "pos0acc", "argmax_ch", "pos0_ch", "rel_dlogit"))
    report["counterfactual"] = []
    for mode in ("uniform", "permute"):
        for targets in [[i] for i in cross_idx] + [None]:
            rec2 = {}
            CTRL.update(mode=mode, targets=(None if targets is None else set(targets)),
                        rec=rec2, perm_seed=7)
            lg = tf_logits(model, mem, src, tin, a.batch)
            c = compare(lg_real, lg, tout)
            deltas = [x for k, v in rec2.items() if isinstance(k, tuple) for x in v]
            c["block_output_rel_change"] = round(sum(deltas) / max(1, len(deltas)), 4)
            c["mode"] = mode
            c["blocks"] = "all" if targets is None else str(targets[0])
            report["counterfactual"].append(c)
            print("  %-9s %-10s %10.4f %7.3f%% %7.3f%% %8.2f%% %9.2f%% %9.4f" % (
                mode, c["blocks"], c["block_output_rel_change"], c["tf_acc"], c["pos0_acc"],
                c["argmax_changed_pct"], c["pos0_argmax_changed_pct"], c["rel_logit_change"]))

    # -- free-running probe under the counterfactuals ---------------------------
    import reversible_jako_probe as P
    ids = []
    for ja, _r in P.PROBE:
        e = tok.encode(ja).ids[:a.max_len - 1] + [M.EOS]
        ids.append(e + [M.PAD] * (a.max_len - len(e)))
    psrc = torch.tensor(ids, dtype=torch.int64)
    outs = {}
    for mode in ("record", "uniform", "permute"):
        CTRL.update(mode=mode, targets=None, rec=None, perm_seed=7)
        g = DEC.decode(model, psrc, a.max_len, "greedy", 0, False, 1, 1)
        outs[mode] = [M.ids_to_text(tok, g[i]) for i in range(len(P.PROBE))]
    print("\n--- free-running probe: real vs all-uniform vs all-permuted attention ---")
    for i, (ja, ref) in enumerate(P.PROBE):
        print("JA       " + ja)
        print("REAL     " + outs["record"][i])
        print("UNIFORM  " + outs["uniform"][i])
        print("PERMUTE  " + outs["permute"][i])
    report["probe"] = {"ja": [j for j, _ in P.PROBE], **outs}
    CTRL.update(mode="record", rec=None)

    json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nwrote %s  (%.0fs)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
