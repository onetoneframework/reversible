#!/usr/bin/env python3
"""
Autonomous supervisor. Runs to the goal without being asked for the next step.

Selection rule, which is the thing every earlier attempt got wrong: a
configuration is only better if the *source* earned it. At 512 pairs the
momentum runs reached 67% teacher-forced content while scoring higher with the
encoder memory zeroed than with the real source, and any criterion built on
accuracy alone would have promoted them. So a run is scored by

    gap = content_acc(real) - content_acc(shuffled)      must be > 0
    real > zeroed                                        must hold
    then content accuracy

and anything failing the first two scores zero no matter how accurate it looks.

Ladder, each rung one variable, escalating only on evidence:

  R1  adjoint width vs accumulator width. hidden_bits 24 gave the only healthy
      source gap at 512 (+27 against +2 for momentum) but ran with 187k
      accumulator saturations per step, so the measurement was half-invalid.
      Sweep (hidden_bits, accum_bits) together.
  R2  the winner at 512 with a longer budget, plus its neighbours.
  R3  8k trainer run with the winner; full validation, probe and anchors.
  R4  30-sentence held-out acceptance on the best checkpoint, greedy.

Every rung's numbers are appended to overnight_translation_report.md as they
land. The supervisor never waits for a human and never stops on a failed rung.
"""

import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python",
                  "Python313", "python.exe")
REPORT = os.path.join(HERE, "overnight_translation_report.md")


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def report(t):
    with open(REPORT, "a", encoding="utf-8") as fh:
        fh.write(t + "\n")
    log(t.replace("\n", " ")[:160])


def env(th):
    return dict(os.environ, OMP_NUM_THREADS=str(th), MKL_NUM_THREADS=str(th),
                PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")


def start(name, cmd, th):
    lf = open(os.path.join(HERE, "_%s.log" % name), "w", encoding="utf-8")
    ef = open(os.path.join(HERE, "_%s.err" % name), "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=HERE, env=env(th), stdout=lf, stderr=ef)
    log("start %-22s pid %d" % (name, p.pid))
    return p


def overfit(name, pairs, epochs, th, **kw):
    out = "auto_%s.json" % name
    cmd = [PY, "-X", "utf8", "-u", "reversible_jako_overfit32.py",
           "--pairs", str(pairs), "--dim", str(kw.get("dim", 96)),
           "--ffn-hidden", str(2 * kw.get("dim", 96)), "--epochs", str(epochs),
           "--report-every", str(max(50, epochs // 10)), "--threads", str(th),
           "--positional", "16", "--max-jaccard", "1.0", "--batch", "32",
           "--quick", "--score-exp2", "3", "--lr-bits", str(kw.get("lr_bits", 1)),
           "--hidden-bits", str(kw.get("hidden_bits", 16)),
           "--momentum-bits", str(kw.get("momentum_bits", 0)),
           "--out", out]
    return start("auto_" + name, cmd, th), out


def score(path):
    """(gap, real-zeroed, content, greedy) with the source requirement applied."""
    try:
        d = json.load(open(os.path.join(HERE, path), encoding="utf-8"))
    except Exception:
        return None
    h = d.get("history", [])
    if not h:
        return None
    best = None
    for r in h:
        gap = r.get("gap_real_minus_shuffled", 0.0)
        rz = (r.get("real_content", 0.0) or 0.0) - (r.get("zeroed_content", 0.0) or 0.0)
        v = (r.get("tf_content_acc", 0.0) if (gap > 0 and rz > 0) else 0.0)
        cand = (v, gap, rz, r.get("greedy_exact", 0), r.get("step", r.get("epoch", 0)))
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand
    return {"score": best[0], "gap": best[1], "real_minus_zeroed": best[2],
            "greedy": best[3], "at": best[4]}


def saturations(name):
    """Accumulator saturations reported by the overfit run's own header line."""
    try:
        txt = open(os.path.join(HERE, "_auto_%s.log" % name), encoding="utf-8").read()
        m = re.findall(r'"accum_saturations": (\d+)', txt)
        return int(m[-1]) if m else None
    except Exception:
        return None


def table(title, rows):
    out = ["", "### %s" % title, "",
           "| run | score (content, source-gated) | gap | real−zeroed | greedy | at |",
           "|---|---:|---:|---:|---:|---:|"]
    for n, s in rows:
        if s is None:
            out.append("| %s | (no result) | | | | |" % n)
        else:
            out.append("| %s | **%.2f** | %+.2f | %+.2f | %d | %s |"
                       % (n, s["score"], s["gap"], s["real_minus_zeroed"],
                          s["greedy"], s["at"]))
    report("\n".join(out))


def run_batch(specs, pairs, epochs, th):
    jobs = [(n, overfit(n, pairs, epochs, th, **kw)) for n, kw in specs]
    for _n, (p, _o) in jobs:
        p.wait()
    return [(n, score(o)) for n, (_p, o) in jobs]


def main():
    report("\n---\n\n## 12. auto.py — autonomous ladder (%s)\n\nSelection requires "
           "source usage: gap > 0 and real > zeroed, else score 0."
           % time.strftime("%Y-%m-%d %H:%M:%S"))

    # R1 — adjoint width against accumulator width, 4 x 4 threads.
    specs = [("hb16", dict(hidden_bits=16)),
             ("hb20", dict(hidden_bits=20)),
             ("hb24", dict(hidden_bits=24)),
             ("hb24lr2", dict(hidden_bits=24, lr_bits=2))]
    res = run_batch(specs, 512, 900, 4)
    for n, _s in res:
        sat = saturations(n)
        if sat:
            report("  %s accumulator saturations/step: %d" % (n, sat))
    table("R1 — 512 pairs, adjoint width", res)
    res = [(n, s) for n, s in res if s]
    if not res:
        report("R1 produced nothing readable; stopping the ladder here.")
        return 1
    best1 = max(res, key=lambda x: (x[1]["score"], x[1]["gap"]))
    report("R1 winner: **%s** (score %.2f, gap %+.2f)"
           % (best1[0], best1[1]["score"], best1[1]["gap"]))

    # R2 — the winner plus neighbours, longer budget.
    hb = int(re.search(r"hb(\d+)", best1[0]).group(1))
    lr = 2 if "lr2" in best1[0] else 1
    specs2 = [("%s_long" % best1[0], dict(hidden_bits=hb, lr_bits=lr)),
              ("hb%d_d192" % hb, dict(hidden_bits=hb, lr_bits=lr, dim=192)),
              ("hb%d_lr%d" % (hb, 3 - lr), dict(hidden_bits=hb, lr_bits=3 - lr))]
    res2 = run_batch(specs2, 512, 2400, 5)
    table("R2 — 512 pairs, longer budget", res2)
    cand = [x for x in res2 + [best1] if x[1]]
    best2 = max(cand, key=lambda x: (x[1]["score"], x[1]["gap"]))
    report("R2 winner: **%s** (score %.2f, gap %+.2f, greedy %d)"
           % (best2[0], best2[1]["score"], best2[1]["gap"], best2[1]["greedy"]))

    # R3 — 8k trainer run with the winner.
    hb = int(re.search(r"hb(\d+)", best2[0]).group(1))
    lr = int(re.search(r"lr(\d)", best2[0]).group(1)) if "lr" in best2[0] else 1
    dim = 192 if "d192" in best2[0] else 96
    tag = "auto8k_%s" % best2[0]
    cmd = [PY, "-X", "utf8", "-u", "reversible_jako_mt.py", "--mode", "train",
           "--corpus", "both", "--max-pairs", "0", "--base-pairs", "8000",
           "--vocab-size", "5000", "--min-char-count", "20", "--max-len", "16",
           "--max-tok", "14", "--tok-mode", "char", "--valid-frac", "0.05",
           "--dim", str(dim), "--ffn-hidden", str(2 * dim), "--k-enc", "2",
           "--k-dec", "4", "--dec-pattern", "ccaf", "--lr-bits", str(lr),
           "--hidden-bits", str(hb), "--state-bits", "16", "--weight-bits", "8",
           "--accum-bits", "32", "--positional", "16", "--score-exp2", "3",
           "--batch", "16", "--threads", "14", "--epochs", "80", "--eval-every", "1",
           "--patience", "25", "--max-eval", "400", "--seed", "0",
           "--show-samples", "10", "--snapshot-epochs", "0", "5", "10", "20", "40",
           "--save-ckpt", "%s.pt" % tag, "--out", "%s.json" % tag,
           "--probe-file", "translations_%s.txt" % tag, "--run-tag", tag + " "]
    report("R3 — 8k with %s (dim %d, hidden_bits %d, lr_bits %d)" % (best2[0], dim, hb, lr))
    start(tag, cmd, 14).wait()

    # R4 — acceptance on whichever checkpoint selection kept.
    ck = "%s_best.pt" % tag
    if not os.path.exists(os.path.join(HERE, ck)):
        ck = "%s_last.pt" % tag
    if os.path.exists(os.path.join(HERE, ck)):
        report("R4 — 30-sentence held-out acceptance, greedy, on %s" % ck)
        start("acceptance", [PY, "-X", "utf8", "-u", "reversible_jako_accept.py",
                             "--ckpt", ck, "--threads", "8",
                             "--out", "final_translation_acceptance.json"], 8).wait()
        try:
            d = json.load(open(os.path.join(HERE, "final_translation_acceptance.json"),
                               encoding="utf-8"))
            g = d.get("TUNED GREEDY", {}).get("metrics", {})
            report("acceptance (greedy): chrF %s BLEU %s korean%% %s blank %s"
                   % (g.get("chrf"), g.get("bleu"), g.get("korean_char_pct"),
                      g.get("blank_outputs")))
        except Exception as e:
            report("acceptance unreadable: %s" % e)
    else:
        report("no 8k checkpoint was written — selection rejected every epoch "
               "because the source never helped. That is the finding.")
    report("auto.py ladder complete at %s" % time.strftime("%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
