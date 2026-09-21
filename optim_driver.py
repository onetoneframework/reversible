#!/usr/bin/env python3
"""
Supervisor for the scale wall. Keeps launching the next experiment on its own.

Where things stand: with coupled base-2 attention the integer model memorises
32/64/128 pairs by greedy decoding and fails flat at 512 (0/512 at 7,200
steps); the float mirror with softmax + Adam memorised 128 in 1,000 steps. Four
512-pair runs are in flight to say whether the integer optimizer's arithmetic
is what breaks at scale:

  float  rung5   softmax + pow2, Adam                (control)
  float  rung8   softmax + pow2, BlockSGD replica    (the integer optimizer's
                                                      arithmetic on float weights)
  int    lr2     lr_bits 2
  int    hb24    hidden_bits 24
  int    mom3    momentum_bits 3                     (launched with this)

Branches, chosen from the results without waiting for anyone:

  A  Adam passes, replica fails   -> the optimizer is the wall. Sweep integer
                                     momentum 3/5/7 at 512; take the best to 8k.
  B  both pass                    -> arithmetic fine; the integer *gradient* is
                                     the difference. Sweep hidden_bits 20/24 and
                                     lr_bits 2 with momentum at 512.
  C  Adam fails                   -> the architecture cannot hold 512 at d96.
                                     Rerun Adam at d192; if it passes, int 512 at
                                     d192 + momentum.

A rung passes when greedy >= 75% of pairs. Anything that passes at 512 goes
straight to an 8k trainer run with the full validation/probe/anchor metrics.
Everything is appended to overnight_translation_report.md.
"""

import json
import os
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


def env(threads):
    return dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")


def popen(name, cmd, threads):
    lf = open(os.path.join(HERE, "_%s.log" % name), "w", encoding="utf-8")
    ef = open(os.path.join(HERE, "_%s.err" % name), "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=HERE, env=env(threads), stdout=lf, stderr=ef)
    log("launched %-16s pid %d" % (name, p.pid))
    return p


def int512(name, threads=3, **kw):
    extra = []
    for k, v in kw.items():
        extra += ["--" + k.replace("_", "-"), str(v)]
    cmd = [PY, "-X", "utf8", "-u", "reversible_jako_overfit32.py", "--pairs", "512",
           "--dim", str(kw.get("dim", 96)), "--ffn-hidden", str(2 * kw.get("dim", 96)),
           "--epochs", "900", "--report-every", "150", "--threads", str(threads),
           "--positional", "16", "--max-jaccard", "1.0", "--batch", "32", "--quick",
           "--score-exp2", "3", "--out", "i512_%s.json" % name]
    for k, v in kw.items():
        if k != "dim":
            cmd += ["--" + k.replace("_", "-"), str(v)]
    if "lr_bits" not in kw:
        cmd += ["--lr-bits", "1"]
    return popen("i512_" + name, cmd, threads), "i512_%s.json" % name


def float512(name, rung, dim=96, threads=4):
    cmd = [PY, "-X", "utf8", "-u", "reversible_jako_bisect.py", "--rung", str(rung),
           "--pairs", "512", "--steps", "12000", "--report-every", "1000",
           "--threads", str(threads), "--dim", str(dim), "--ffn-hidden", str(2 * dim)]
    return popen("bis512_" + name, cmd, threads), "bisect_r%d.json" % rung


def stage4(name, threads=6, **kw):
    cmd = [PY, "-X", "utf8", "-u", "reversible_jako_mt.py", "--mode", "train",
           "--corpus", "both", "--max-pairs", "0", "--base-pairs", "8000",
           "--vocab-size", "5000", "--min-char-count", "20", "--max-len", "16",
           "--max-tok", "14", "--tok-mode", "char", "--valid-frac", "0.05",
           "--dim", str(kw.get("dim", 96)), "--ffn-hidden", str(2 * kw.get("dim", 96)),
           "--k-enc", "2", "--k-dec", "4", "--dec-pattern", "ccaf",
           "--lr-bits", str(kw.get("lr_bits", 1)), "--hidden-bits", str(kw.get("hidden_bits", 16)),
           "--state-bits", "16", "--weight-bits", "8", "--accum-bits", "32",
           "--positional", "16", "--score-exp2", "3",
           "--momentum-bits", str(kw.get("momentum_bits", 0)),
           "--batch", "16", "--threads", str(threads), "--epochs", "80",
           "--eval-every", "1", "--patience", "25", "--max-eval", "400", "--seed", "0",
           "--show-samples", "10", "--snapshot-epochs", "0", "5", "10", "20", "40",
           "--save-ckpt", "jako_stage4_%s.pt" % name, "--out", "jako_stage4_%s.json" % name,
           "--probe-file", "jako_translations_stage4_%s.txt" % name,
           "--run-tag", "stage4-%s " % name]
    return popen("stage4_" + name, cmd, threads), "jako_stage4_%s.json" % name


def greedy_of(path, key="greedy_exact"):
    try:
        d = json.load(open(os.path.join(HERE, path), encoding="utf-8"))
    except Exception:
        return None
    h = d.get("history", [])
    if not h:
        return None
    return max(x.get(key, x.get("greedy_exact", 0)) for x in h)


def wait(files, poll=60):
    while not all(os.path.exists(os.path.join(HERE, f)) for f in files):
        time.sleep(poll)


def main():
    report("\n---\n\n## 11. optim_driver.py — supervising the 512-pair wall (%s)\n"
           % time.strftime("%Y-%m-%d %H:%M:%S"))
    wait(["bisect_r5.json", "bisect_r8.json"])
    adam = greedy_of("bisect_r5.json"); rep = greedy_of("bisect_r8.json")
    lr2 = greedy_of("i512_lr2.json"); hb24 = greedy_of("i512_hb24.json")
    mom3 = greedy_of("i512_mom3.json")
    line = ("512 pairs: float Adam %s/512, float BlockSGD-replica %s/512; integer "
            "lr2 %s, hb24 %s, mom3 %s" % (adam, rep, lr2, hb24, mom3))
    log(line); report(line)
    P = 384
    if adam is not None and adam < P:
        branch = "C"
    elif rep is not None and rep >= P:
        branch = "B"
    else:
        branch = "A"
    report("Branch **%s**." % branch)
    log("branch " + branch)

    if branch == "A":
        jobs = [int512("mom5", momentum_bits=5), int512("mom7", momentum_bits=7),
                int512("mom5_lr2", momentum_bits=5, lr_bits=2)]
    elif branch == "B":
        jobs = [int512("hb20_mom3", hidden_bits=20, momentum_bits=3),
                int512("hb24_mom3", hidden_bits=24, momentum_bits=3),
                int512("lr2_mom3", lr_bits=2, momentum_bits=3)]
    else:
        p, f = float512("adam_d192", 5, dim=192)
        p.wait()
        a192 = greedy_of(f)
        report("float Adam at d192: %s/512" % a192)
        jobs = [int512("d192_mom3", dim=192, momentum_bits=3),
                int512("d192_mom5", dim=192, momentum_bits=5)]
    for p, _f in jobs:
        p.wait()
    res = {n: greedy_of(f) for n, f in [(j[1].replace("i512_", "").replace(".json", ""), j[1]) for j in jobs]}
    if mom3 is not None:
        res["mom3"] = mom3
    report("512 results: " + json.dumps(res))
    log("512 results: " + json.dumps(res))
    best = max(res, key=lambda k: (res[k] or -1))
    report("best at 512: **%s** = %s/512" % (best, res[best]))
    kw = {}
    for tok in best.split("_"):
        if tok.startswith("mom"): kw["momentum_bits"] = int(tok[3:])
        if tok.startswith("hb"): kw["hidden_bits"] = int(tok[2:])
        if tok.startswith("lr"): kw["lr_bits"] = int(tok[2:])
        if tok.startswith("d") and tok[1:].isdigit(): kw["dim"] = int(tok[1:])
    if (res[best] or 0) >= P:
        report("512 passed; launching STAGE 4 (8k) with %s" % kw)
    else:
        report("512 not passed (best %s); launching STAGE 4 with the best setting "
               "anyway to measure, and continuing the ladder" % res[best])
    p, f = stage4(best, threads=10, **kw)
    p.wait()
    report("STAGE 4 %s finished; see jako_stage4_%s.json" % (best, best))
    log("done")


if __name__ == "__main__":
    sys.exit(main())
