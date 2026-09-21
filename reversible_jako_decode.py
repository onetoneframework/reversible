#!/usr/bin/env python3
"""
Practical integer decoding for the reversible JA->KO model.

Everything here works on raw integer logits. No softmax, no log-probabilities,
no floating-point scores. Three modes:

  greedy          plain argmax over integer logits. This is the research
                  baseline and must always be reported alongside anything else.
  greedy-repeat   greedy plus integer repetition control: a fixed penalty
                  subtracted from already-emitted tokens, an outright ban on
                  repeating a bigram, and minimum/maximum length handling on EOS.
  beam            width-k beam search ranked by cumulative integer logit.

Any run that used a decoding heuristic must say so; `greedy` is the honest
baseline for the paper.
"""

import argparse
import json
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_jako_mt as M

PAD, BOS, EOS = M.PAD, M.BOS, M.EOS
NEG = -(1 << 40)          # integer stand-in for "impossible", never -inf


def _step_logits(model, cur, mem, src, t):
    dec, _c, _i, _tp = model.decode(cur, mem, src)
    logits, _s = model.logits(dec)
    return logits[:, t]


def decode(model, src, max_len, mode="greedy", rep_penalty=0, no_repeat_bigram=False,
           min_len=1, beam=1):
    if mode == "beam" and beam > 1:
        return _beam(model, src, max_len, beam, rep_penalty, no_repeat_bigram, min_len)
    B = src.shape[0]
    mem, _c, _i, _t = model.encode(src)
    cur = torch.full((B, max_len), PAD, dtype=torch.int64)
    cur[:, 0] = BOS
    out = torch.full((B, max_len), PAD, dtype=torch.int64)
    done = torch.zeros(B, dtype=torch.bool)
    emitted = [dict() for _ in range(B)]
    prev = [None] * B
    seen_bg = [set() for _ in range(B)]

    for t in range(max_len - 1):
        sc = _step_logits(model, cur, mem, src, t).clone()
        if mode != "greedy":
            for b in range(B):
                if rep_penalty:
                    for tokid, n in emitted[b].items():
                        sc[b, tokid] -= rep_penalty * n
                if no_repeat_bigram and prev[b] is not None:
                    for (p, q) in seen_bg[b]:
                        if p == prev[b]:
                            sc[b, q] = NEG
                if t < min_len:
                    sc[b, EOS] = NEG
            sc[:, PAD] = NEG
        nxt = sc.argmax(dim=1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        out[:, t] = nxt
        for b in range(B):
            if done[b]:
                continue
            k = int(nxt[b])
            emitted[b][k] = emitted[b].get(k, 0) + 1
            if prev[b] is not None:
                seen_bg[b].add((prev[b], k))
            prev[b] = k
        done = done | (nxt == EOS)
        if bool(done.all()):
            break
        cur[:, t + 1] = nxt
    return out


def _beam(model, src, max_len, width, rep_penalty, no_repeat_bigram, min_len):
    """Cumulative-integer-logit beam search, one sentence at a time."""
    outs = []
    for b in range(src.shape[0]):
        s1 = src[b:b + 1]
        mem, _c, _i, _t = model.encode(s1)
        beams = [{"ids": [], "score": 0, "done": False}]
        for t in range(max_len - 1):
            cand = []
            for bm in beams:
                if bm["done"]:
                    cand.append(bm)
                    continue
                cur = torch.full((1, max_len), PAD, dtype=torch.int64)
                cur[0, 0] = BOS
                for i, v in enumerate(bm["ids"]):
                    cur[0, i + 1] = v
                sc = _step_logits(model, cur, mem, s1, t)[0].clone()
                if rep_penalty:
                    for tokid in set(bm["ids"]):
                        sc[tokid] -= rep_penalty * bm["ids"].count(tokid)
                if no_repeat_bigram and bm["ids"]:
                    p = bm["ids"][-1]
                    for i in range(len(bm["ids"]) - 1):
                        if bm["ids"][i] == p:
                            sc[bm["ids"][i + 1]] = NEG
                if len(bm["ids"]) < min_len:
                    sc[EOS] = NEG
                sc[PAD] = NEG
                top = sc.topk(width)
                for v, tokid in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append({"ids": bm["ids"] + [tokid], "score": bm["score"] + v,
                                 "done": tokid == EOS})
            cand.sort(key=lambda x: -x["score"])
            beams = cand[:width]
            if all(x["done"] for x in beams):
                break
        best = beams[0]["ids"]
        row = torch.full((max_len,), PAD, dtype=torch.int64)
        for i, v in enumerate(best[:max_len]):
            row[i] = v
        outs.append(row)
    return torch.stack(outs)


def stats(tok, ids_batch):
    """Degeneracy and script statistics over a batch of generated sequences."""
    tot = dict(n=0, rep_uni=0.0, rep_bi=0.0, ko=0.0, bad=0.0, ln=0.0, uniq=0)
    texts = []
    for i in range(ids_batch.shape[0]):
        seq = []
        for v in ids_batch[i].tolist():
            if v in (EOS, PAD):
                break
            seq.append(v)
        txt = M.ids_to_text(tok, ids_batch[i])
        texts.append(txt)
        tot["n"] += 1
        tot["ln"] += len(seq)
        if seq:
            tot["rep_uni"] += 1.0 - len(set(seq)) / len(seq)
            bg = [tuple(seq[j:j + 2]) for j in range(len(seq) - 1)]
            if bg:
                tot["rep_bi"] += 1.0 - len(set(bg)) / len(bg)
        if txt:
            tot["ko"] += sum(1 for c in txt if "가" <= c <= "힣") / len(txt)
            tot["bad"] += sum(1 for c in txt if c == "�") / len(txt)
    n = max(1, tot["n"])
    return {"mean_len": round(tot["ln"] / n, 1),
            "repeat_unigram_pct": round(100 * tot["rep_uni"] / n, 1),
            "repeat_bigram_pct": round(100 * tot["rep_bi"] / n, 1),
            "korean_char_pct": round(100 * tot["ko"] / n, 1),
            "replacement_char_pct": round(100 * tot["bad"] / n, 1),
            "distinct_outputs": len(set(texts)), "outputs": n}, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="jako_model.pt")
    ap.add_argument("--decode", choices=["greedy", "greedy-repeat", "beam"],
                    default="greedy")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--rep-penalty", type=int, default=0)
    ap.add_argument("--no-repeat-bigram", action="store_true")
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--text", nargs="*", default=[])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, tok, ck = M.load_ckpt(a.ckpt)
    L = a.max_len or ck["max_len"]
    sents = a.text or ["今日は雨です。", "私は学生です。", "明日は学校に行きます。"]
    ids = []
    for s in sents:
        e = tok.encode(s).ids[:L - 1] + [EOS]
        ids.append(e + [PAD] * (L - len(e)))
    src = torch.tensor(ids, dtype=torch.int64)
    out = decode(model, src, L, a.decode, a.rep_penalty, a.no_repeat_bigram,
                 a.min_len, a.beam)
    st, texts = stats(tok, out)
    for s, t in zip(sents, texts):
        print("JA:   " + s)
        print("KO:   " + t)
        print()
    print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    main()
