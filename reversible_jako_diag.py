#!/usr/bin/env python3
"""
Diagnostics for the reversible-integer JA->KO translator.

  overfit      can it memorise 32/64/128 real pairs? teacher-forced accuracy AND
               greedy reproduction of the very sentences it trained on
  conditioning is the decoder actually reading the source, or is it a Korean
               language model? counterfactual source swap + encoder-memory ablation
  repetition   degeneracy statistics on generated output

Nothing here changes the integer training path; it only measures it.

  python reversible_jako_diag.py --mode overfit --pairs 32 64 128
  python reversible_jako_diag.py --mode conditioning --ckpt jako_model.pt
"""

import argparse
import json
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_integer_final as R
import reversible_jako_data as D
import reversible_jako_mt as M
from reversible_integer_final import BlockSGD


def build(vocab, args):
    cfg = R.RunConfig(state_bits=args.state_bits, weight_bits=args.weight_bits,
                      accum_bits=args.accum_bits)
    R.CFG = cfg
    return M.JaKoMT(vocab, args.dim, args.ffn_hidden, args.k_enc, args.k_dec,
                    cfg, args.seed), cfg


def rep_stats(ids):
    """Degeneracy statistics for one generated id sequence."""
    seq = []
    for i in ids.tolist():
        if i in (M.EOS, M.PAD):
            break
        seq.append(i)
    if not seq:
        return {"len": 0, "uniq_ratio": 0.0, "rep_uni_pct": 0.0, "rep_bi_pct": 0.0}
    uni = len(set(seq)) / len(seq)
    rep_uni = 1.0 - uni
    bg = [tuple(seq[i:i + 2]) for i in range(len(seq) - 1)]
    rep_bi = 0.0 if not bg else 1.0 - len(set(bg)) / len(bg)
    return {"len": len(seq), "uniq_ratio": round(uni, 3),
            "rep_uni_pct": round(100 * rep_uni, 1),
            "rep_bi_pct": round(100 * rep_bi, 1)}


def korean_ratio(text):
    if not text:
        return 0.0
    ko = sum(1 for c in text if "가" <= c <= "힣")
    bad = sum(1 for c in text if c == "�")
    return round(100.0 * ko / len(text), 1), round(100.0 * bad / len(text), 1)


# -----------------------------------------------------------------------------
def overfit(args):
    out = []
    for n in args.pairs:
        ds = D.prepare(args.corpus, max(n * 3, 64), args.vocab_size, args.max_len,
                       valid_frac=0.01)
        src_all, tin_all, tout_all = ds["train"]
        n = min(n, len(src_all))
        src = torch.tensor(src_all[:n], dtype=torch.int64)
        tin = torch.tensor(tin_all[:n], dtype=torch.int64)
        tout = torch.tensor(tout_all[:n], dtype=torch.int64)
        pairs = ds["train_pairs"][:n]
        tok = ds["tokenizer"]

        model, _cfg = build(ds["info"]["vocab_size"], args)
        opt = BlockSGD(model.parameters(), param_bits=12, lr_bits=args.lr_bits, clip=1)
        hb = args.hidden_bits
        b = args.batch
        best = 0.0
        for ep in range(args.epochs):
            for st in range(0, n, b):
                g, _s = M.forward_backward(model, src[st:st + b], tin[st:st + b],
                                           tout[st:st + b], hb, cached=False)
                opt.step(g)
            if (ep + 1) % args.report_every == 0 or ep == args.epochs - 1:
                logits, _ = model.logits(model.decode(
                    tin, model.encode(src)[0], src)[0])
                _g, stt = M.hinge_grad(logits, tout, hb)
                acc = 100.0 * stt["correct"] / max(1, stt["tokens"])
                best = max(best, acc)
                print(f"  pairs={n} ep={ep+1:>4} teacher_forced_token_acc={acc:6.2f}% "
                      f"hinge/tok={stt['hinge_sum']/max(1,stt['tokens']):7.2f}", flush=True)
        # Greedy reproduction of the very sentences it trained on.
        gen = M.greedy(model, src, args.max_len)
        exact = 0
        shown = []
        for i in range(n):
            h = M.ids_to_text(tok, gen[i])
            r = pairs[i][1]
            exact += int(h.strip() == r.strip())
            if i < 3:
                shown.append({"ja": pairs[i][0], "ref": r, "gen": h})
        row = {"pairs": n, "epochs": args.epochs,
               "best_teacher_forced_token_acc": round(best, 2),
               "greedy_train_exact_match_pct": round(100.0 * exact / n, 2),
               "samples": shown}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        out.append(row)
    return out


# -----------------------------------------------------------------------------
def conditioning(args):
    """Does the decoder read the source at all?"""
    ck = torch.load(args.ckpt, weights_only=False)
    cfg = R.RunConfig(state_bits=ck["state_bits"], weight_bits=ck["weight_bits"],
                      accum_bits=ck["accum_bits"])
    R.CFG = cfg
    model = M.JaKoMT(ck["vocab"], ck["dim"], ck["ffn_hidden"], ck["k_enc"],
                     ck["k_dec"], cfg, 0)
    for dst, s in zip(model.parameters(), ck["weights"]):
        dst.copy_(s)
    tok = D.load_tokenizer(ck["tokenizer_path"])
    L = ck["max_len"]

    sents = args.text or ["私は学生です。",
                          "私は先生です。",
                          "今日は雨です。",
                          "明日は学校に行きます。"]

    def enc_ids(s):
        e = tok.encode(s).ids[:L - 1] + [M.EOS]
        return e + [M.PAD] * (L - len(e))

    src = torch.tensor([enc_ids(s) for s in sents], dtype=torch.int64)
    mem, _c, _i, _t = model.encode(src)

    # Same decoder prefix (just BOS), different sources: do the first steps differ?
    prefix = torch.full((len(sents), L), M.PAD, dtype=torch.int64)
    prefix[:, 0] = M.BOS
    dec, _c2, _i2, _t2 = model.decode(prefix, mem, src)
    logits, _s = model.logits(dec)
    first = logits[:, 0].argmax(dim=1).tolist()
    top5 = [logits[i, 0].topk(5).indices.tolist() for i in range(len(sents))]

    # Encoder-memory ablation: zero the memory and regenerate.
    zero_mem = torch.zeros_like(mem)
    dec0, _a, _b2, _c3 = model.decode(prefix, zero_mem, src)
    logits0, _s0 = model.logits(dec0)
    first0 = logits0[:, 0].argmax(dim=1).tolist()

    gen_real = M.greedy(model, src, L)
    # greedy with zeroed memory, done by hand so the ablation is honest
    cur = torch.full((len(sents), L), M.PAD, dtype=torch.int64)
    cur[:, 0] = M.BOS
    outz = torch.full((len(sents), L), M.PAD, dtype=torch.int64)
    for t in range(L - 1):
        d, _x, _y, _z = model.decode(cur, zero_mem, src)
        lg, _ = model.logits(d)
        nx = lg[:, t].argmax(dim=1)
        outz[:, t] = nx
        if t + 1 < L:
            cur[:, t + 1] = nx

    rows = []
    for i, s in enumerate(sents):
        real = M.ids_to_text(tok, gen_real[i])
        abl = M.ids_to_text(tok, outz[i])
        kr, badr = korean_ratio(real)
        rows.append({"ja": s, "first_token_with_source": first[i],
                     "first_token_zero_memory": first0[i],
                     "top5_with_source": top5[i],
                     "generation_with_source": real,
                     "generation_zero_memory": abl,
                     "identical_to_ablation": real == abl,
                     "korean_char_pct": kr, "replacement_char_pct": badr,
                     **rep_stats(gen_real[i])})
    n_same = sum(r["identical_to_ablation"] for r in rows)
    n_distinct_first = len(set(first))
    print(json.dumps({"sentences": len(sents),
                      "distinct_first_tokens_across_sources": n_distinct_first,
                      "generations_identical_when_memory_zeroed": n_same,
                      "verdict": ("decoder ignores the source"
                                  if n_same == len(sents) or n_distinct_first == 1
                                  else "source does influence the decoder")},
                     ensure_ascii=False), flush=True)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overfit", "conditioning"], default="overfit")
    ap.add_argument("--corpus", default="tatoeba")
    ap.add_argument("--pairs", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--report-every", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ffn-hidden", type=int, default=128)
    ap.add_argument("--k-enc", type=int, default=2)
    ap.add_argument("--k-dec", type=int, default=3)
    ap.add_argument("--lr-bits", type=int, default=0)
    ap.add_argument("--state-bits", type=int, default=16)
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--accum-bits", type=int, default=32)
    ap.add_argument("--hidden-bits", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt", default="jako_model.pt")
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = overfit(a) if a.mode == "overfit" else conditioning(a)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
