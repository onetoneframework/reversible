#!/usr/bin/env python3
"""
Is the failure the first token, or is the whole decoder a target-suffix
memoriser?

The 32-pair overfit reached 68% teacher-forced content accuracy and 0/32 greedy
reproduction, with a zeroed encoder memory scoring 63% -- so the memorisation
runs through the gold target prefix, not through the source. This narrows that
down with four measurements on the trained model:

  1. position 0 under real / shuffled / zeroed source. Position 0 sees only BOS,
     so a correct first token has to come from the source. If the three agree,
     source -> first-token was never learned.

  2. the rank of the gold first token at position 0. Being second or fifth means
     the signal exists and the margin is weak; being thousandth means there is
     no lexical mapping to speak of.

  3. an oracle-prefix sweep: hand the decoder k gold target tokens, free-run the
     rest, for k = 0,1,2,4,8. The shape of that curve separates "the first token
     is the bottleneck" (jumps at k=1) from "it memorised target suffixes"
     (climbs steadily with k).

  4. the same sweep with sources shuffled between examples. If the curve does
     not move, the decoder is not using the source at any position, and the
     memorisation is purely target-side.

  python reversible_jako_oracle.py --ckpt jako_overfit32b.pt --pairs jako_overfit32b.json
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


def free_run(model, mem, src_mask, tout, max_len, k):
    """Greedy after being handed k gold target tokens. k=0 is plain greedy."""
    B = mem.shape[0]
    cur = torch.full((B, max_len), M.PAD, dtype=torch.int64)
    cur[:, 0] = M.BOS
    for j in range(min(k, max_len - 1)):
        cur[:, j + 1] = tout[:, j]
    out = torch.full((B, max_len), M.PAD, dtype=torch.int64)
    out[:, :k] = tout[:, :k]
    done = torch.zeros(B, dtype=torch.bool)
    for t in range(k, max_len):
        dec, _c, _i, _x = model.decode(cur, mem, src_mask)
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


def score(out, tout, tok, refs):
    """Exact reconstruction, token accuracy over non-pad targets, chrF."""
    notpad = tout != M.PAD
    hit = int(((out == tout) & notpad).sum())
    tot = int(notpad.sum())
    hyp = [M.ids_to_text(tok, out[i]) for i in range(out.shape[0])]
    exact = sum(1 for h, r in zip(hyp, refs) if h.strip() == r.strip())
    return exact, round(100.0 * hit / max(1, tot), 3), round(M.chrf(hyp, refs), 3), hyp


def prefix_ablation(model, tok, src, tin, tout, klass, perm, batch=32):
    """
    Cross the source with the target prefix, teacher-forced, no generation.

    Four cells: source real or shuffled between examples, prefix gold or taken
    from another sentence, plus a neutral prefix that is BOS and nothing else.
    The target being scored is always the real one. If content accuracy holds up
    when the source is wrong but collapses when the prefix is wrong, then the
    accuracy was coming from the target prefix, and this shows it directly
    instead of inferring it from free-running behaviour.
    """
    import reversible_jako_metrics as MET
    n = src.shape[0]
    neutral = torch.full_like(tin, M.PAD)
    neutral[:, 0] = M.BOS
    cells = {}
    for sname, s_ids in (("real", src), ("shuffled", src[perm])):
        for pname, p_ids in (("gold", tin), ("shuffled", tin[perm]),
                             ("neutral", neutral)):
            chit = ctot = p0hit = 0
            for st in range(0, n, batch):
                mem, _c, _i, _x = model.encode(s_ids[st:st + batch])
                dec, _c2, _i2, _x2 = model.decode(p_ids[st:st + batch], mem,
                                                  s_ids[st:st + batch])
                lg, _s = model.logits(dec)
                tgt = tout[st:st + batch]
                pred = lg.argmax(-1)
                p0hit += int((pred[:, 0] == tgt[:, 0]).sum())
                tl = tgt.reshape(-1).tolist()
                okl = (pred.reshape(-1) == tgt.reshape(-1)).tolist()
                for i2 in range(len(tl)):
                    if tl[i2] == M.PAD or klass(tl[i2]) != "content":
                        continue
                    ctot += 1
                    chit += int(okl[i2])
            cells["source=%s,prefix=%s" % (sname, pname)] = {
                "content_acc": round(100.0 * chit / max(1, ctot), 4),
                "pos0_acc": round(100.0 * p0hit / n, 3)}
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="jako_overfit32b.pt")
    ap.add_argument("--pairs", default="jako_overfit32b.json")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="jako_oracle.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    model, tok, ck = M.load_ckpt(a.ckpt)
    blob = json.load(open(a.pairs, encoding="utf-8"))
    pairs = [(p["ja"], p["ko"]) for p in blob["pairs"]]
    L = ck["max_len"]
    src, tin, tout = encode(tok, pairs, L)
    refs = [ko for _ja, ko in pairs]
    n = src.shape[0]
    print("ckpt %s  pairs %d  max_len %d  vocab %d" % (a.ckpt, n, L, model.vocab))

    mem, _c, _i, _x = model.encode(src)
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(n, generator=g)
    conds = {"real": (mem, src),
             "shuffled": (mem[perm], src[perm]),
             "zeroed": (torch.zeros_like(mem), src)}

    report = {"ckpt": a.ckpt, "pairs": n}

    # ---- 1 & 2. position 0 -----------------------------------------------
    print("\n--- 1/2. position 0: accuracy, and where the gold first token ranks ---")
    print("  %-9s %8s %10s %11s %8s %9s %11s" % (
        "source", "pos0_acc", "mean_rank", "median_rank", "top5", "top10",
        "margin(gold-best)"))
    report["pos0"] = {}
    for name, (mm, sm) in conds.items():
        dec, _c2, _i2, _x2 = model.decode(tin, mm, sm)
        lg, _s = model.logits(dec)
        z0 = lg[:, 0, :]
        gold = tout[:, 0]
        gold_logit = z0.gather(1, gold[:, None]).reshape(-1)
        rank = (z0 > gold_logit[:, None]).sum(dim=1) + 1
        best = z0.max(dim=1).values
        report["pos0"][name] = {
            "acc": round(100.0 * float((rank == 1).double().mean()), 3),
            "mean_rank": round(float(rank.double().mean()), 1),
            "median_rank": int(rank.median()),
            "top5": round(100.0 * float((rank <= 5).double().mean()), 2),
            "top10": round(100.0 * float((rank <= 10).double().mean()), 2),
            "margin": round(float((gold_logit - best).double().mean()), 2),
        }
        r = report["pos0"][name]
        print("  %-9s %7.2f%% %10.1f %11d %7.1f%% %8.1f%% %11.2f" % (
            name, r["acc"], r["mean_rank"], r["median_rank"], r["top5"],
            r["top10"], r["margin"]))

    # ---- 3 & 4. oracle-prefix sweep --------------------------------------
    print("\n--- 3/4. oracle-prefix sweep: k gold target tokens, then free-run ---")
    print("  %-4s %-9s %9s %11s %8s" % ("k", "source", "exact", "token_acc", "chrF"))
    report["sweep"] = []
    for k in a.ks:
        for name in ("real", "shuffled"):
            mm, sm = conds[name]
            out = free_run(model, mm, sm, tout, L, k)
            ex, ta, cf, hyp = score(out, tout, tok, refs)
            report["sweep"].append({"k": k, "source": name, "exact": ex,
                                    "token_acc": ta, "chrf": cf})
            print("  %-4d %-9s %6d/%-3d %10.3f%% %8.3f" % (k, name, ex, n, ta, cf))
            if k == 1 and name == "real":
                report["given1_outputs"] = hyp

    # ---- 5. prefix ablation ----------------------------------------------
    import reversible_jako_metrics as MET
    klass = MET.build_classifier(tok, tout.tolist())
    print("\n--- 5. teacher-forced 2x2: which input is the accuracy coming from? ---")
    cells = prefix_ablation(model, tok, src, tin, tout, klass, perm)
    report["prefix_ablation"] = cells
    print("  %-28s %13s %10s" % ("condition", "content_acc", "pos0_acc"))
    for k in sorted(cells):
        print("  %-28s %12.4f%% %9.3f%%" % (k, cells[k]["content_acc"],
                                            cells[k]["pos0_acc"]))
    gg = cells["source=real,prefix=gold"]["content_acc"]
    sg = cells["source=shuffled,prefix=gold"]["content_acc"]
    gs = cells["source=real,prefix=shuffled"]["content_acc"]
    print("  wrong source, gold prefix : %+.4f%%  (cost of losing the source)"
          % (sg - gg))
    print("  gold source, wrong prefix : %+.4f%%  (cost of losing the prefix)"
          % (gs - gg))

    real = {r["k"]: r for r in report["sweep"] if r["source"] == "real"}
    shuf = {r["k"]: r for r in report["sweep"] if r["source"] == "shuffled"}
    k1 = real.get(1, {}).get("exact", 0)
    kmax = max(a.ks)
    print("\nk=0 exact %d/%d   k=1 exact %d/%d   k=%d exact %d/%d" % (
        real.get(0, {}).get("exact", 0), n, k1, n, kmax,
        real.get(kmax, {}).get("exact", 0), n))
    print("token-accuracy gap real - shuffled: " + "  ".join(
        "k=%d %+.3f%%" % (k, real[k]["token_acc"] - shuf[k]["token_acc"])
        for k in sorted(real)))
    json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nwrote " + a.out)


if __name__ == "__main__":
    main()
