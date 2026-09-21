#!/usr/bin/env python3
"""
Validation metrics that separate what the model learned from what teacher
forcing gave it for free.

Pooled token accuracy is not a usable training signal here. Measured on the
run1 ep6 checkpoint, 83% of every correct token was EOS, whose position the
gold prefix already announces; content tokens sat at 0.200%. A checkpoint
selector reading the pooled number therefore prefers models that stop early,
and that is what it picked.

So this reports, every validation:

  by token class     EOS, punctuation, high-frequency, content -- with the
                     share of the correct predictions each class carries.
  by position        0, 1, 2, 3, 4-7, 8+. Position 0 sees only BOS, so the
                     source can reach it only through cross-attention; it is
                     the sharpest available probe of whether the model
                     translates or merely continues Korean.
  by corpus          content accuracy and content hinge per source corpus.
  source conditioning  the same batch decoded against its real source, against
                     sources shuffled between examples, and against a zeroed
                     encoder memory. content_acc(real) - content_acc(shuffled)
                     is the number that says source identity is being used at
                     all; it was 0 for run1.
  free running       a fixed held-out probe, greedy and repeat-suppressed, with
                     degeneracy statistics and a count of source-specific
                     lexical anchors that survive into the output.

The probe sentences are diagnostics only and never select a checkpoint.
"""

import collections

import torch

import reversible_jako_mt as M

PUNCT = set(" .,?!'\"()[]-~:;…·・、。！？　▁")

# Source-specific content words. A hit means the Korean side appeared in the
# output for the sentence whose Japanese side contains the trigger -- the
# sentence can still be wrong; this counts lexical transfer, not translation.
LEXICAL_ANCHORS = [
    ("学生", ("학생",)),
    ("三つ", ("세 ", "셋", "３개", "3개")),
    ("暑い", ("덥", "더워", "더운")),
    ("駅", ("역",)),
    ("水", ("물",)),
    ("飲み", ("마시", "마셔", "마십", "먹")),
    ("わかりません", ("모르", "몰라")),
    ("始まり", ("시작",)),
]


def build_classifier(tok, tout_rows, top_n=50):
    """Classify each target token id as eos / punct / high_freq / content."""
    freq = collections.Counter()
    for row in tout_rows:
        freq.update(v for v in row if v != M.PAD)
    top = set(t for t, _c in freq.most_common(top_n)) - {M.EOS}
    cache = {}

    def klass(v):
        if v == M.EOS:
            return "eos"
        if v in cache:
            return cache[v]
        s = tok.decode([v])
        k = "punct" if (s and all(c in PUNCT for c in s)) else (
            "high_freq" if v in top else "content")
        cache[v] = k
        return k
    return klass


def _hinge_parts(logits, tgt):
    """Per-token true logit, best wrong logit and prediction. No gradient."""
    B, T, V = logits.shape
    flat = logits.reshape(B * T, V)
    t = tgt.reshape(-1)
    rows = torch.arange(B * T)
    true_l = flat[rows, t]
    masked = flat.clone()
    masked[rows, t] = torch.iinfo(torch.int64).min // 4
    j = masked.argmax(dim=1)
    best_w = flat[rows, j]
    pred = flat.argmax(dim=1)
    return true_l, best_w, pred, t


def teacher_forced(model, data, hb, batch, tok, klass, tags=None, max_eval=None):
    """Full class/position breakdown over the validation set."""
    src_all, tin_all, tout_all = data
    n = len(src_all) if max_eval is None else min(len(src_all), max_eval)
    T = len(tout_all[0])
    cls_hit, cls_tot, cls_hinge = collections.Counter(), collections.Counter(), collections.Counter()
    pos_hit = torch.zeros(T, dtype=torch.int64)
    pos_tot = torch.zeros(T, dtype=torch.int64)
    eos_fp = non_eos = 0
    hinge_sum = tok_tot = 0
    per_tag = collections.defaultdict(lambda: {"hit": 0, "tot": 0, "hinge": 0})

    for st in range(0, n, batch):
        src = torch.tensor(src_all[st:st + batch], dtype=torch.int64)
        tin = torch.tensor(tin_all[st:st + batch], dtype=torch.int64)
        tout = torch.tensor(tout_all[st:st + batch], dtype=torch.int64)
        mem, _c, _i, _x = model.encode(src)
        dec, _c2, _i2, _x2 = model.decode(tin, mem, src)
        lg, _s = model.logits(dec)
        true_l, best_w, pred, t = _hinge_parts(lg, tout)
        hinge = torch.clamp_min(best_w + 1 - true_l, 0)
        ok = (pred == t)
        notpad = t != M.PAD

        B = lg.shape[0]
        p2 = pred.reshape(B, T)
        o2 = ok.reshape(B, T)
        n2 = notpad.reshape(B, T)
        pos_hit += (o2 & n2).sum(dim=0)
        pos_tot += n2.sum(dim=0)
        hinge_sum += int(hinge[notpad].sum())
        tok_tot += int(notpad.sum())
        is_eos = (t == M.EOS)
        non_eos += int((notpad & ~is_eos).sum())
        eos_fp += int(((pred == M.EOS) & notpad & ~is_eos).sum())

        tl = t.tolist()
        okl = ok.tolist()
        hl = hinge.tolist()
        for idx in range(len(tl)):
            v = tl[idx]
            if v == M.PAD:
                continue
            k = klass(v)
            cls_tot[k] += 1
            cls_hit[k] += int(okl[idx])
            cls_hinge[k] += int(hl[idx])
            if tags is not None:
                tg = tags[st + idx // T]
                d = per_tag[tg]
                d["tot"] += 1
                if k == "content":
                    d.setdefault("c_tot", 0)
                    d["c_tot"] = d.get("c_tot", 0) + 1
                    d["hit"] += int(okl[idx])
                    d["hinge"] += int(hl[idx])

    correct = sum(cls_hit.values())
    out = {
        "tokens": tok_tot,
        "all_token_acc": round(100.0 * correct / max(1, tok_tot), 3),
        "hinge_per_token": round(hinge_sum / max(1, tok_tot), 4),
        "eos_recall": round(100.0 * cls_hit["eos"] / max(1, cls_tot["eos"]), 3),
        "eos_false_positive": round(100.0 * eos_fp / max(1, non_eos), 3),
        "content_acc": round(100.0 * cls_hit["content"] / max(1, cls_tot["content"]), 4),
        "content_hinge": round(cls_hinge["content"] / max(1, cls_tot["content"]), 4),
        "high_freq_acc": round(100.0 * cls_hit["high_freq"] / max(1, cls_tot["high_freq"]), 3),
        "punct_acc": round(100.0 * cls_hit["punct"] / max(1, cls_tot["punct"]), 3),
        "eos_share_of_correct": round(100.0 * cls_hit["eos"] / max(1, correct), 2),
        "class_tokens": {k: cls_tot[k] for k in cls_tot},
    }
    groups = [("pos0", 0, 1), ("pos1", 1, 2), ("pos2", 2, 3), ("pos3", 3, 4),
              ("pos4-7", 4, 8), ("pos8+", 8, T)]
    out["position_acc"] = {}
    for name, lo, hi in groups:
        hi = min(hi, T)
        if lo >= hi:
            continue
        h = int(pos_hit[lo:hi].sum())
        tt = int(pos_tot[lo:hi].sum())
        if tt:
            out["position_acc"][name] = round(100.0 * h / tt, 3)
    if tags is not None:
        out["by_corpus"] = {
            tg: {"content_acc": round(100.0 * d["hit"] / max(1, d.get("c_tot", 0)), 4),
                 "content_hinge": round(d["hinge"] / max(1, d.get("c_tot", 0)), 4),
                 "content_tokens": d.get("c_tot", 0)}
            for tg, d in sorted(per_tag.items()) if d.get("c_tot", 0) >= 20}
        vals = [v["content_acc"] for v in out["by_corpus"].values()]
        out["balanced_content_acc"] = round(sum(vals) / len(vals), 4) if vals else out["content_acc"]
        hv = [v["content_hinge"] for v in out["by_corpus"].values()]
        out["balanced_content_hinge"] = round(sum(hv) / len(hv), 4) if hv else out["content_hinge"]
    else:
        out["balanced_content_acc"] = out["content_acc"]
        out["balanced_content_hinge"] = out["content_hinge"]
    return out


def source_conditioning(model, data, batch, klass, n=128, seed=1234):
    """real vs shuffled vs zeroed encoder memory, on content tokens and pos 0."""
    src_all, tin_all, tout_all = data
    n = min(n, len(src_all))
    src = torch.tensor(src_all[:n], dtype=torch.int64)
    tin = torch.tensor(tin_all[:n], dtype=torch.int64)
    tout = torch.tensor(tout_all[:n], dtype=torch.int64)
    mems = []
    for st in range(0, n, batch):
        m, _c, _i, _x = model.encode(src[st:st + batch])
        mems.append(m)
    mem = torch.cat(mems)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    res = {}
    for name, mm, sm in (("real", mem, src),
                         ("shuffled", mem[perm], src[perm]),
                         ("zeroed", torch.zeros_like(mem), src)):
        chit = ctot = p0hit = p0tot = 0
        for st in range(0, n, batch):
            dec, _c, _i, _x = model.decode(tin[st:st + batch], mm[st:st + batch],
                                           sm[st:st + batch])
            lg, _s = model.logits(dec)
            _tl, _bw, pred, t = _hinge_parts(lg, tout[st:st + batch])
            B, T = tout[st:st + batch].shape
            p2 = pred.reshape(B, T)
            tg = tout[st:st + batch]
            p0tot += int((tg[:, 0] != M.PAD).sum())
            p0hit += int(((p2[:, 0] == tg[:, 0]) & (tg[:, 0] != M.PAD)).sum())
            tl, okl = t.tolist(), (pred == t).tolist()
            for i2 in range(len(tl)):
                if tl[i2] == M.PAD or klass(tl[i2]) != "content":
                    continue
                ctot += 1
                chit += int(okl[i2])
        res[name] = {"content_acc": round(100.0 * chit / max(1, ctot), 4),
                     "pos0_acc": round(100.0 * p0hit / max(1, p0tot), 3)}
    res["content_acc_gap_real_minus_shuffled"] = round(
        res["real"]["content_acc"] - res["shuffled"]["content_acc"], 4)
    res["pos0_acc_gap_real_minus_shuffled"] = round(
        res["real"]["pos0_acc"] - res["shuffled"]["pos0_acc"], 3)
    res["sequences"] = n
    return res


def free_running(model, tok, max_len, rep_penalty=8, min_len=2):
    """Fixed probe, greedy and repeat-suppressed, with lexical-anchor counting."""
    import reversible_jako_decode as DEC
    import reversible_jako_probe as P
    ids = []
    for ja, _ref in P.PROBE:
        e = tok.encode(ja).ids[:max_len - 1] + [M.EOS]
        ids.append(e + [M.PAD] * (max_len - len(e)))
    src = torch.tensor(ids, dtype=torch.int64)
    out = {}
    for name, args in (("greedy", ("greedy", 0, False, 1, 1)),
                       ("repeat", ("greedy-repeat", rep_penalty, True, min_len, 1))):
        gen = DEC.decode(model, src, max_len, *args)
        st, hyp = DEC.stats(tok, gen)
        lens = [len(h) for h in hyp]
        eos_pos = []
        for i in range(gen.shape[0]):
            row = gen[i].tolist()
            eos_pos.append(row.index(M.EOS) if M.EOS in row else -1)
        anchors = 0
        for i, (ja, _ref) in enumerate(P.PROBE):
            for trig, kos in LEXICAL_ANCHORS:
                if trig in ja and any(k in hyp[i] for k in kos):
                    anchors += 1
                    break
        ref_len = sum(len(r) for _j, r in P.PROBE) / len(P.PROBE)
        out[name] = {
            "outputs": hyp,
            "blank_outputs": sum(1 for h in hyp if not h.strip()),
            "korean_pct": st["korean_char_pct"],
            "replacement_pct": st["replacement_char_pct"],
            "repeat_unigram_pct": st["repeat_unigram_pct"],
            "repeat_bigram_pct": st["repeat_bigram_pct"],
            "distinct_outputs": st["distinct_outputs"],
            "mean_len": round(sum(lens) / len(lens), 2),
            "median_len": sorted(lens)[len(lens) // 2],
            "len_ratio_vs_reference": round((sum(lens) / len(lens)) / ref_len, 3),
            "eos_positions": eos_pos,
            "lexical_anchor_hits": anchors,
            "lexical_anchor_max": len(P.PROBE),
        }
    return out


def epoch_report(model, data, hb, batch, tok, klass, tags, max_len,
                 max_eval=512, cond_n=128, with_free=True):
    r = {"teacher_forced": teacher_forced(model, data, hb, batch, tok, klass,
                                          tags, max_eval)}
    r["source_conditioning"] = source_conditioning(model, data, batch, klass, cond_n)
    if with_free:
        r["free_running"] = free_running(model, tok, max_len)
    return r


def one_line(r):
    """A compact line for the training log."""
    tf = r["teacher_forced"]
    sc = r["source_conditioning"]
    fr = r.get("free_running", {}).get("greedy", {})
    return ("all %6.3f%% content %7.4f%% bal %7.4f%% | cont r/s/z %6.3f/%6.3f/%6.3f "
            "gap %+7.4f | pos0 r/s/z %5.2f/%5.2f/%5.2f | eosFP %5.2f%% eos_sh %5.1f%% "
            "| anchors %d/%d blank %d len %.1f kor %.0f%% rep %.1f%%" % (
                tf["all_token_acc"], tf["content_acc"], tf["balanced_content_acc"],
                sc["real"]["content_acc"], sc["shuffled"]["content_acc"],
                sc["zeroed"]["content_acc"], sc["content_acc_gap_real_minus_shuffled"],
                sc["real"]["pos0_acc"], sc["shuffled"]["pos0_acc"],
                sc["zeroed"]["pos0_acc"], tf["eos_false_positive"],
                tf["eos_share_of_correct"],
                fr.get("lexical_anchor_hits", 0), fr.get("lexical_anchor_max", 0),
                fr.get("blank_outputs", 0), fr.get("mean_len", 0.0),
                fr.get("korean_pct", 0.0), fr.get("repeat_unigram_pct", 0.0)))
