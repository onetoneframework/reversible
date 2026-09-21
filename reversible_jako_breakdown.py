#!/usr/bin/env python3
"""
Where does the validation token accuracy actually come from?

Teacher-forced accuracy and free-running translation are different things, and
a run can improve one while the other gets worse. This answers three questions
about a checkpoint, offline, without touching a training job:

  1. position-wise accuracy -- position 0 sees only BOS, so the source can only
     reach it through cross-attention. If position 0 is dead while later
     positions improve, the model is learning target-side continuation rather
     than translation.
  2. EOS behaviour -- recall, false-positive rate, and the position-0 margin of
     EOS over the best non-EOS token. Premature EOS shows up here.
  3. source ablation -- teacher-forced and free-running, with the real source,
     with sources shuffled between examples, and with the encoder memory zeroed.
     If the three agree, the decoder is not using the source.

Plus a token-class breakdown (EOS / punctuation / high-frequency / content), so
it is visible whether the accuracy is carried by easy tokens.

  python reversible_jako_breakdown.py --ckpt jako_50k_best.pt --n 512
"""

import argparse
import collections
import json
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_jako_mt as M
import reversible_jako_data as D

PUNCT = set(" .,?!'\"()[]-~:;…·・、。！？　▁")


def tf_predict(model, src, tin, memory=None, src_mask_ids=None, batch=32):
    """Teacher-forced argmax over the whole batch, in chunks."""
    outs = []
    for st in range(0, src.shape[0], batch):
        s = src[st:st + batch]
        t = tin[st:st + batch]
        if memory is None:
            mem, _c, _i, _x = model.encode(s)
            sm = s
        else:
            mem = memory[st:st + batch]
            sm = src_mask_ids[st:st + batch]
        dec, _c2, _i2, _x2 = model.decode(t, mem, sm)
        lg, _sh = model.logits(dec)
        outs.append(lg)
    return outs


def acc_from_logits(chunks, tout, batch=32):
    """Overall and per-position accuracy from teacher-forced logits."""
    T = tout.shape[1]
    hit = torch.zeros(T, dtype=torch.int64)
    tot = torch.zeros(T, dtype=torch.int64)
    st = 0
    for lg in chunks:
        b = lg.shape[0]
        tgt = tout[st:st + b]
        pred = lg.argmax(dim=2)
        ok = (pred == tgt) & (tgt != M.PAD)
        hit += ok.sum(dim=0)
        tot += (tgt != M.PAD).sum(dim=0)
        st += b
    return hit, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="jako_50k_best.pt")
    ap.add_argument("--n", type=int, default=512, help="validation sequences")
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="jako_breakdown.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    model, tok, ck = M.load_ckpt(a.ckpt)
    print("checkpoint %s  epoch %s  vocab %d  dim %d"
          % (a.ckpt, ck.get("epoch"), model.vocab, model.d), flush=True)

    ds = D.prepare(a.corpus, 0, a.vocab_size, a.max_len, seed=a.seed,
                   max_tok=a.max_tok, tok_mode=ck.get("tok_mode", "char"),
                   base_pairs=a.base_pairs, subs_pairs=a.subs_pairs,
                   base_pool=a.base_pool, subs_pool=a.subs_pool,
                   min_char_count=a.min_char_count)
    n = min(a.n, len(ds["valid"][0]))
    src = torch.tensor(ds["valid"][0][:n], dtype=torch.int64)
    tin = torch.tensor(ds["valid"][1][:n], dtype=torch.int64)
    tout = torch.tensor(ds["valid"][2][:n], dtype=torch.int64)
    print("valid sequences %d, tokens %d"
          % (n, int((tout != M.PAD).sum())), flush=True)

    report = {"ckpt": a.ckpt, "epoch": ck.get("epoch"), "n": n}

    # ---------------------------------------------------------------- 1. position
    chunks = tf_predict(model, src, tin, batch=a.batch)
    hit, tot = acc_from_logits(chunks, tout, a.batch)
    pos = []
    for t in range(tout.shape[1]):
        if int(tot[t]) == 0:
            continue
        pos.append({"pos": t, "tokens": int(tot[t]),
                    "acc": round(100.0 * int(hit[t]) / int(tot[t]), 3)})
    overall = round(100.0 * int(hit.sum()) / max(1, int(tot.sum())), 3)
    report["overall_tf_acc"] = overall
    report["position_acc"] = pos
    print("\n--- 1. position-wise teacher-forced accuracy ---")
    print("overall %.3f%%" % overall)
    for r in pos[:10]:
        print("  pos %-2d  tokens %-6d  acc %6.3f%%" % (r["pos"], r["tokens"], r["acc"]))
    late_h = sum(int(hit[t]) for t in range(8, tout.shape[1]))
    late_t = sum(int(tot[t]) for t in range(8, tout.shape[1]))
    if late_t:
        report["pos_8plus_acc"] = round(100.0 * late_h / late_t, 3)
        print("  pos 8+  tokens %-6d  acc %6.3f%%" % (late_t, report["pos_8plus_acc"]))

    # ------------------------------------------------------------ 2. token classes
    freq = collections.Counter()
    for row in ds["valid"][2]:
        freq.update(v for v in row if v != M.PAD)
    top = set(t for t, _c in freq.most_common(50))

    def klass(v):
        if v == M.EOS:
            return "eos"
        s = tok.decode([v])
        if s and all(c in PUNCT for c in s):
            return "punct"
        return "high_freq" if v in top else "content"

    cls_hit = collections.Counter()
    cls_tot = collections.Counter()
    st = 0
    for lg in chunks:
        b = lg.shape[0]
        tgt = tout[st:st + b]
        pred = lg.argmax(dim=2)
        for i in range(b):
            for t in range(tgt.shape[1]):
                v = int(tgt[i, t])
                if v == M.PAD:
                    continue
                k = klass(v)
                cls_tot[k] += 1
                cls_hit[k] += int(pred[i, t] == v)
        st += b
    report["token_class"] = {k: {"tokens": cls_tot[k],
                                 "acc": round(100.0 * cls_hit[k] / max(1, cls_tot[k]), 3),
                                 "share_of_correct": round(
                                     100.0 * cls_hit[k] / max(1, sum(cls_hit.values())), 2)}
                             for k in cls_tot}
    print("\n--- 2. accuracy by target token class ---")
    print("  %-10s %8s %9s %16s" % ("class", "tokens", "acc", "share of correct"))
    for k in ("eos", "punct", "high_freq", "content"):
        if k in cls_tot:
            r = report["token_class"][k]
            print("  %-10s %8d %8.3f%% %15.2f%%" % (k, r["tokens"], r["acc"],
                                                    r["share_of_correct"]))

    # ------------------------------------------------------------------- 3. EOS
    eos_hit = eos_tot = fp = nontot = 0
    p0_margin = []
    st = 0
    for lg in chunks:
        b = lg.shape[0]
        tgt = tout[st:st + b]
        pred = lg.argmax(dim=2)
        is_eos = tgt == M.EOS
        notpad = tgt != M.PAD
        eos_tot += int(is_eos.sum())
        eos_hit += int(((pred == M.EOS) & is_eos).sum())
        nonz = notpad & ~is_eos
        nontot += int(nonz.sum())
        fp += int(((pred == M.EOS) & nonz).sum())
        z0 = lg[:, 0, :].clone()
        e0 = z0[:, M.EOS].clone()
        z0[:, M.EOS] = torch.iinfo(torch.int64).min // 4
        p0_margin.append((e0 - z0.max(dim=1).values).double())
        st += b
    marg = torch.cat(p0_margin)
    report["eos"] = {
        "recall": round(100.0 * eos_hit / max(1, eos_tot), 3),
        "false_positive_rate": round(100.0 * fp / max(1, nontot), 3),
        "eos_targets": eos_tot, "non_eos_targets": nontot,
        "pos0_eos_minus_best_other_mean": round(float(marg.mean()), 2),
        "pos0_eos_is_argmax_pct": round(100.0 * float((marg > 0).double().mean()), 3),
    }
    print("\n--- 3. EOS behaviour ---")
    for k, v in report["eos"].items():
        print("  %-32s %s" % (k, v))

    # -------------------------------------------------------- 4. source ablation
    print("\n--- 4. source ablation ---", flush=True)
    mems = []
    for st in range(0, n, a.batch):
        m, _c, _i, _x = model.encode(src[st:st + a.batch])
        mems.append(m)
    mem_all = torch.cat(mems)
    g = torch.Generator().manual_seed(1234)
    perm = torch.randperm(n, generator=g)

    abl = {}
    for name, mem, smask in (("real", mem_all, src),
                             ("shuffled", mem_all[perm], src[perm]),
                             ("zeroed", torch.zeros_like(mem_all), src)):
        ch = tf_predict(model, src, tin, memory=mem, src_mask_ids=smask, batch=a.batch)
        h, t = acc_from_logits(ch, tout, a.batch)
        abl[name] = {
            "tf_acc": round(100.0 * int(h.sum()) / max(1, int(t.sum())), 3),
            "pos0_acc": round(100.0 * int(h[0]) / max(1, int(t[0])), 3),
        }
        del ch
    report["ablation_teacher_forced"] = abl
    print("  teacher-forced")
    for k, v in abl.items():
        print("    %-9s overall %7.3f%%   pos0 %7.3f%%" % (k, v["tf_acc"], v["pos0_acc"]))

    # Free-running: does the generated text change when the source changes?
    import reversible_jako_decode as DEC
    nf = min(128, n)
    gen_real = DEC.decode(model, src[:nf], a.max_len, "greedy", 0, False, 1, 1)
    gen_shuf = DEC.decode(model, src[perm][:nf], a.max_len, "greedy", 0, False, 1, 1)
    same_first = int((gen_real[:, 0] == gen_shuf[:, 0]).sum())
    txt_r = [M.ids_to_text(tok, gen_real[i]) for i in range(nf)]
    txt_s = [M.ids_to_text(tok, gen_shuf[i]) for i in range(nf)]
    same_full = sum(1 for x, y in zip(txt_r, txt_s) if x == y)
    report["ablation_free_running"] = {
        "sequences": nf,
        "identical_first_token_pct": round(100.0 * same_first / nf, 2),
        "identical_output_pct": round(100.0 * same_full / nf, 2),
        "distinct_outputs_real": len(set(txt_r)),
        "distinct_first_tokens_real": len(set(gen_real[:, 0].tolist())),
        "mean_generated_len": round(float((gen_real != M.PAD).sum(dim=1).double().mean()), 2),
        "mean_reference_len": round(float((tout[:nf] != M.PAD).sum(dim=1).double().mean()), 2),
    }
    print("  free-running (greedy, %d sequences)" % nf)
    for k, v in report["ablation_free_running"].items():
        print("    %-28s %s" % (k, v))

    json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nwrote " + a.out)


if __name__ == "__main__":
    main()
