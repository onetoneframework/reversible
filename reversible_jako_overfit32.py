#!/usr/bin/env python3
"""
Can the causal-fixed architecture represent and learn a source->target mapping
at all? A 32-pair memorisation test, judged on generation rather than on
teacher forcing.

Teacher-forced accuracy on its own has already fooled this project once: an
earlier run reached 84% teacher-forced while reproducing 0 of its training
sentences under greedy decoding. So the verdict here is the greedy column, and
all four of these are reported together:

  1. teacher-forced content accuracy (EOS and punctuation excluded)
  2. greedy reproduction of the very 32 sentences it trained on
  3. real source vs sources shuffled between examples
  4. real source vs a zeroed encoder memory

Verdicts:
  A  greedy reproduces most of the 32 -> the mechanism can learn a mapping;
     the failure at 7.6k pairs is generalisation, and positional encoding is a
     justified next single intervention.
  B  teacher forcing high, greedy still fails -> a target-prefix shortcut
     survives the causality fix; do not add structure until it is found.
  C  teacher forcing cannot even memorise 32 pairs -> representational or
     optimisation bottleneck; more data is pointless.

The 32 pairs are chosen so their source token multisets do not overlap much.
The model has no positional encoding, so sentences that are anagrams of each
other are not distinguishable in principle, and including them would fail the
test for a reason this test is not asking about.

  python reversible_jako_overfit32.py --pairs 32 --epochs 2000 --threads 3
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
import reversible_jako_data as D
import reversible_jako_metrics as MET
import reversible_jako_decode as DEC
from reversible_integer_final import BlockSGD


def pick_pairs(ds, tok, n, max_jaccard, max_len):
    """Distinct sentences whose source token multisets do not overlap much."""
    chosen, sets = [], []
    for rec in ds["train_pairs"]:
        ja, ko = rec[0], rec[1]
        s = set(tok.encode(ja).ids)
        if len(s) < 4:
            continue
        if any(len(s & t) / max(1, len(s | t)) > max_jaccard for t in sets):
            continue
        chosen.append((ja, ko))
        sets.append(s)
        if len(chosen) == n:
            break
    return chosen


def encode(tok, pairs, max_len):
    src, tin, tout = [], [], []
    for ja, ko in pairs:
        s = tok.encode(ja).ids[:max_len - 1] + [M.EOS]
        t = tok.encode(ko).ids[:max_len - 1]
        ti = [M.BOS] + t
        to = t + [M.EOS]
        src.append(s + [M.PAD] * (max_len - len(s)))
        tin.append(ti + [M.PAD] * (max_len - len(ti)))
        tout.append(to + [M.PAD] * (max_len - len(to)))
    return (torch.tensor(src, dtype=torch.int64),
            torch.tensor(tin, dtype=torch.int64),
            torch.tensor(tout, dtype=torch.int64))


def greedy_with_prefix(model, src, tout, max_len, k):
    """Greedy decoding after being handed the first k gold target tokens.

    If generation only fails because position 0 is wrong, handing it that one
    token repairs everything downstream; if it does not, the failure is not
    the first token.
    """
    B = src.shape[0]
    mem, _c, _i, _x = model.encode(src)
    cur = torch.full((B, max_len), M.PAD, dtype=torch.int64)
    cur[:, 0] = M.BOS
    for j in range(min(k, max_len - 1)):
        cur[:, j + 1] = tout[:, j]
    out = torch.full((B, max_len), M.PAD, dtype=torch.int64)
    out[:, :k] = tout[:, :k]
    done = torch.zeros(B, dtype=torch.bool)
    for t in range(k, max_len):
        dec, _c2, _i2, _x2 = model.decode(cur, mem, src)
        lg, _s = model.logits(dec)
        nxt = lg[:, t].argmax(dim=1)
        nxt = torch.where(done, torch.full_like(nxt, M.PAD), nxt)
        out[:, t] = nxt
        done = done | (nxt == M.EOS)
        if t + 1 < max_len:
            cur[:, t + 1] = torch.where(done, torch.full_like(nxt, M.PAD), nxt)
        if bool(done.all()):
            break
    return out


QUICK = [0]


def report(model, tok, src, tin, tout, klass, refs, max_len, batch):
    if QUICK[0]:
        return quick_report(model, tok, src, tin, tout, klass, refs, max_len, batch)
    return full_report(model, tok, src, tin, tout, klass, refs, max_len, batch)


def quick_report(model, tok, src, tin, tout, klass, refs, max_len, batch):
    """Everything the discriminating question needs and nothing else.

    Drops the give1/give3 prefix-oracle decodes, which triple the decoding cost,
    and scores the source ablation on a 64-sequence slice. Greedy reproduction
    and the real-vs-shuffled gap are still measured on the full set.
    """
    data = (src.tolist(), tin.tolist(), tout.tolist())
    tf = MET.teacher_forced(model, data, 16, batch, tok, klass, None, None)
    sc = MET.source_conditioning(model, data, batch, klass, n=min(64, src.shape[0]))
    gen = DEC.decode(model, src, max_len, "greedy", 0, False, 1, 1)
    hyp = [M.ids_to_text(tok, gen[i]) for i in range(gen.shape[0])]
    exact = sum(1 for h, r in zip(hyp, refs) if h.strip() == r.strip())
    return {"tf_all_acc": tf["all_token_acc"], "tf_content_acc": tf["content_acc"],
            "tf_eos_share": tf["eos_share_of_correct"],
            "tf_pos0_acc": tf["position_acc"].get("pos0", 0.0),
            "tf_pos1_acc": 0.0, "tf_pos2_acc": 0.0, "tf_pos8plus_acc": 0.0,
            "greedy_exact": exact, "greedy_given1_exact": 0,
            "greedy_given3_exact": 0, "greedy_given1_chrf": 0.0,
            "hyp_given1": [], "greedy_chrf": round(M.chrf(hyp, refs), 3),
            "real_content": sc["real"]["content_acc"],
            "shuffled_content": sc["shuffled"]["content_acc"],
            "zeroed_content": sc["zeroed"]["content_acc"],
            "gap_real_minus_shuffled": sc["content_acc_gap_real_minus_shuffled"],
            "gap_real_minus_zeroed": round(sc["real"]["content_acc"]
                                           - sc["zeroed"]["content_acc"], 4),
            "hyp": hyp}


def full_report(model, tok, src, tin, tout, klass, refs, max_len, batch):
    data = (src.tolist(), tin.tolist(), tout.tolist())
    tf = MET.teacher_forced(model, data, 16, batch, tok, klass, None, None)
    sc = MET.source_conditioning(model, data, batch, klass, n=src.shape[0])
    gen = DEC.decode(model, src, max_len, "greedy", 0, False, 1, 1)
    hyp = [M.ids_to_text(tok, gen[i]) for i in range(gen.shape[0])]
    exact = sum(1 for h, r in zip(hyp, refs) if h.strip() == r.strip())
    g1 = greedy_with_prefix(model, src, tout, max_len, 1)
    h1 = [M.ids_to_text(tok, g1[i]) for i in range(g1.shape[0])]
    e1 = sum(1 for h, r in zip(h1, refs) if h.strip() == r.strip())
    g3 = greedy_with_prefix(model, src, tout, max_len, 3)
    h3 = [M.ids_to_text(tok, g3[i]) for i in range(g3.shape[0])]
    e3 = sum(1 for h, r in zip(h3, refs) if h.strip() == r.strip())
    return {
        "tf_all_acc": tf["all_token_acc"],
        "tf_content_acc": tf["content_acc"],
        "tf_eos_share": tf["eos_share_of_correct"],
        "tf_pos0_acc": tf["position_acc"].get("pos0", 0.0),
        "tf_pos1_acc": tf["position_acc"].get("pos1", 0.0),
        "tf_pos2_acc": tf["position_acc"].get("pos2", 0.0),
        "tf_pos8plus_acc": tf["position_acc"].get("pos8+", 0.0),
        "greedy_exact": exact,
        "greedy_given1_exact": e1,
        "greedy_given3_exact": e3,
        "greedy_given1_chrf": round(M.chrf(h1, refs), 3),
        "hyp_given1": h1,
        "greedy_chrf": round(M.chrf(hyp, refs), 3),
        "real_content": sc["real"]["content_acc"],
        "shuffled_content": sc["shuffled"]["content_acc"],
        "zeroed_content": sc["zeroed"]["content_acc"],
        "gap_real_minus_shuffled": sc["content_acc_gap_real_minus_shuffled"],
        "gap_real_minus_zeroed": round(sc["real"]["content_acc"]
                                       - sc["zeroed"]["content_acc"], 4),
        "hyp": hyp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--max-jaccard", type=float, default=0.5)
    # Same architecture as run2, read from the historical small checkpoint.
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--ffn-hidden", type=int, default=192)
    ap.add_argument("--k-enc", type=int, default=2)
    ap.add_argument("--k-dec", type=int, default=4)
    ap.add_argument("--enc-pattern", default="")
    ap.add_argument("--dec-pattern", default="ccaf")
    ap.add_argument("--score-leak", type=int, default=3)
    ap.add_argument("--lr-bits", type=int, default=0)
    ap.add_argument("--hidden-bits", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--max-tok", type=int, default=14)
    ap.add_argument("--vocab-size", type=int, default=5000)
    ap.add_argument("--min-char-count", type=int, default=20)
    ap.add_argument("--corpus", default="both")
    ap.add_argument("--base-pairs", type=int, default=8000)
    ap.add_argument("--query-anchor", action="store_true")
    ap.add_argument("--prefix-corrupt", type=int, default=0)
    ap.add_argument("--scheduled-sampling", type=int, default=0)
    ap.add_argument("--positional", type=int, default=0)
    ap.add_argument("--attn-mass-bits", type=int, default=0)
    ap.add_argument("--attn-score-bits", type=int, default=0)
    ap.add_argument("--score-exp2", type=int, default=None)
    ap.add_argument("--score-exp2-cap", type=int, default=12)
    ap.add_argument("--momentum-bits", type=int, default=0)
    ap.add_argument("--exp2-diag", action="store_true",
                    help="use the per-entry adjoint instead of the coupled one")
    ap.add_argument("--quick", action="store_true",
                    help="drop the prefix-oracle decodes; keep greedy and the gap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="jako_overfit32.json")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    A.SCORE_LEAK = a.score_leak
    A.SCORE_EXP2 = a.score_exp2
    A.SCORE_EXP2_CAP = a.score_exp2_cap
    A.SCORE_EXP2_COUPLED = not a.exp2_diag
    M.CROSS_QUERY_ANCHOR = bool(a.query_anchor)
    M.ATTN_ROW_MASS_BITS = int(a.attn_mass_bits)
    M.ATTN_SCORE_BITS = int(a.attn_score_bits)
    QUICK[0] = 1 if a.quick else 0
    cfg = R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
    R.CFG = cfg

    ds = D.prepare(a.corpus, 0, a.vocab_size, a.max_len, seed=a.seed,
                   max_tok=a.max_tok, tok_mode="char", base_pairs=a.base_pairs,
                   min_char_count=a.min_char_count)
    tok = ds["tokenizer"]
    pairs = pick_pairs(ds, tok, a.pairs, a.max_jaccard, a.max_len)
    src, tin, tout = encode(tok, pairs, a.max_len)
    refs = [ko for _ja, ko in pairs]
    klass = MET.build_classifier(tok, tout.tolist())
    for ja, ko in pairs[:4]:
        print("  sample  %s  ->  %s" % (ja, ko), flush=True)

    model = M.JaKoMT(ds["info"]["vocab_size"], a.dim, a.ffn_hidden, a.k_enc,
                     a.k_dec, cfg, a.seed, enc_pattern=a.enc_pattern,
                     dec_pattern=a.dec_pattern, positional=a.positional)
    opt = BlockSGD(model.parameters(), param_bits=12, lr_bits=a.lr_bits, clip=1,
                   momentum_bits=a.momentum_bits)
    print(json.dumps({"pairs": len(pairs), "vocab": ds["info"]["vocab_size"],
                      "max_len": a.max_len, "dim": a.dim, "k_dec": a.k_dec,
                      "dec_pattern": a.dec_pattern, "lr_bits": a.lr_bits,
                      "optimizer_stable": opt.stable,
                      "train_prefix_corruption": a.prefix_corrupt / 100.0,
                      "train_scheduled_sampling": a.scheduled_sampling / 100.0,
                      "eval_prefix_corruption": 0.0,
                      "positional": a.positional,
                      "attn_row_mass_bits": a.attn_mass_bits,
                      "attn_score_bits": a.attn_score_bits,
                      "score_exp2": a.score_exp2,
                      "exp2_coupled": not a.exp2_diag,
                      "query_anchor": a.query_anchor,
                      "causal_fix": "mask-aware pow2norm"}, ensure_ascii=False),
          flush=True)

    n = src.shape[0]
    cgen = torch.Generator().manual_seed(a.seed + 7919)
    hist = []
    t0 = time.time()
    print("\n%6s %9s %11s %10s %8s %9s %9s %9s %9s %6s" % (
        "epoch", "tf_all%", "tf_content%", "pos0%", "greedy", "give1",
        "give3", "real%", "shuf%", "zero%"), flush=True)
    for ep in range(a.epochs + 1):
        if ep % a.report_every == 0:
            r = report(model, tok, src, tin, tout, klass, refs, a.max_len, a.batch)
            r["epoch"] = ep
            r["minutes"] = round((time.time() - t0) / 60, 1)
            hist.append({k: v for k, v in r.items()
                         if k not in ("hyp", "hyp_given1")})
            print("%6d %9.3f %11.4f %7.2f %4d/%-3d %6d %6d %9.4f %9.4f %9.4f  %.1fm" % (
                ep, r["tf_all_acc"], r["tf_content_acc"], r["tf_pos0_acc"],
                r["greedy_exact"], n, r["greedy_given1_exact"],
                r["greedy_given3_exact"], r["real_content"],
                r["shuffled_content"], r["zeroed_content"], r["minutes"]),
                flush=True)
            if r["greedy_exact"] == n:
                print("all %d reproduced; stopping early" % n, flush=True)
                break
        if ep == a.epochs:
            break
        for st in range(0, n, a.batch):
            ti = tin[st:st + a.batch]
            if a.scheduled_sampling:
                ti = M.scheduled_sample(model, src[st:st + a.batch], ti,
                                        a.scheduled_sampling, cgen)
            if a.prefix_corrupt:
                ti = M.corrupt_prefix(ti, a.prefix_corrupt, cgen)
            g, _s = M.forward_backward(model, src[st:st + a.batch], ti,
                                       tout[st:st + a.batch],
                                       a.hidden_bits, cached=False)
            opt.step(g)

    final = report(model, tok, src, tin, tout, klass, refs, a.max_len, a.batch)
    print("\n--- final greedy reproduction of the training sentences ---")
    for i, (ja, ko) in enumerate(pairs):
        mark = "OK " if final["hyp"][i].strip() == ko.strip() else "   "
        print("%s[%2d] JA  %s" % (mark, i, ja))
        print("     REF %s" % ko)
        print("     GEN %s" % final["hyp"][i])

    best_greedy = max(h["greedy_exact"] for h in hist)
    best_tf = max(h["tf_content_acc"] for h in hist)
    if best_greedy >= 0.5 * n:
        verdict = "A: the mechanism learns a source->target mapping; the 7.6k failure is generalisation"
    elif best_tf >= 50.0:
        verdict = "B: teacher forcing memorises but generation does not; a prefix shortcut survives"
    else:
        verdict = "C: teacher forcing cannot memorise 32 pairs; representational or optimisation bottleneck"
    print("\nbest greedy exact %d/%d   best tf content acc %.3f%%" % (best_greedy, n, best_tf))
    print("VERDICT %s" % verdict)
    import os as _os
    torch.save({"weights": [w.clone() for w in model.parameters()],
                "dim": a.dim, "ffn_hidden": a.ffn_hidden, "k_enc": a.k_enc,
                "k_dec": a.k_dec, "enc_pattern": a.enc_pattern,
                "dec_pattern": a.dec_pattern, "vocab": model.vocab,
                "max_len": a.max_len, "state_bits": 16, "weight_bits": 8,
                "accum_bits": 32, "score_leak": a.score_leak,
                "query_anchor": a.query_anchor, "opt_acc": None,
                "positional": a.positional, "score_exp2": a.score_exp2,
                "score_exp2_cap": a.score_exp2_cap, "exp2_coupled": not a.exp2_diag,
                "attn_mass_bits": a.attn_mass_bits, "attn_score_bits": a.attn_score_bits,
                "epoch": a.epochs, "tok_mode": "char", "lr_bits": a.lr_bits,
                "hidden_bits": a.hidden_bits,
                "tokenizer_path": _os.path.join(D.CACHE, ds["info"]["tokenizer"])},
               a.out.replace(".json", ".pt"))
    json.dump({"pairs": [{"ja": j, "ko": k} for j, k in pairs], "history": hist,
               "final": final, "verdict": verdict,
               "best_greedy_exact": best_greedy, "best_tf_content_acc": best_tf},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
