#!/usr/bin/env python3
"""
Self-driving search for the attention formulation that makes generation work.

Where this stands. A float mirror of this model — same reversible coupling,
same positional tables, same hinge loss, same data — reproduces 128/128 pairs
by greedy decoding in 1000 Adam steps once softmax attention is used, and 0/128
without it. The integer path has neither softmax nor any other row
normalisation. Adding a power-of-two L1 row normalisation alone did not fix it
(0/128 at every mass target tried), so the remaining candidates are the other
properties softmax supplies — non-negative weights and score resolution — and
combinations of the three.

Each rung is a batch of variants run in parallel on a deliberately small
problem: 64 pairs, d64, quick evaluation. Small enough that a rung takes a
couple of minutes, large enough that the plain formulation is known to fail on
it. The driver reads the results, records them, and starts the next rung
itself; it never waits to be woken.

Pass condition for a rung: any variant reproducing most of the pairs under
greedy decoding, which is what the plain formulation cannot do.

  python attn_driver.py
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

PAIRS = 64
PASS_GREEDY = 48            # of 64
THREADS = 3


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def report(t):
    with open(REPORT, "a", encoding="utf-8") as fh:
        fh.write(t + "\n")


def launch(name, extra, epochs=600, every=100):
    out = os.path.join(HERE, "ad_%s.json" % name)
    cmd = [PY, "-X", "utf8", "-u", "reversible_jako_overfit32.py",
           "--pairs", str(PAIRS), "--dim", "64", "--ffn-hidden", "128",
           "--epochs", str(epochs), "--report-every", str(every),
           "--threads", str(THREADS), "--lr-bits", "1", "--positional", "16",
           "--max-jaccard", "1.0", "--batch", "32", "--quick",
           "--out", out] + extra
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS),
               MKL_NUM_THREADS=str(THREADS), PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1")
    lf = open(os.path.join(HERE, "_ad_%s.log" % name), "w", encoding="utf-8")
    ef = open(os.path.join(HERE, "_ad_%s.err" % name), "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=lf, stderr=ef)
    log("  %-18s pid %-6d %s" % (name, p.pid, " ".join(extra) or "(plain)"))
    return {"name": name, "proc": p, "out": out, "extra": extra}


def collect(jobs):
    for j in jobs:
        j["proc"].wait()
    res = []
    for j in jobs:
        try:
            d = json.load(open(j["out"], encoding="utf-8"))
            h = d.get("history", [])
            res.append({
                "name": j["name"], "extra": " ".join(j["extra"]) or "(plain)",
                "greedy": max([x.get("greedy_exact", 0) for x in h] or [0]),
                "tf": max([x.get("tf_content_acc", 0.0) for x in h] or [0.0]),
                "gap": max([x.get("gap_real_minus_shuffled", 0.0) for x in h] or [0.0]),
                "pos0": max([x.get("tf_pos0_acc", 0.0) for x in h] or [0.0]),
            })
        except Exception as e:
            res.append({"name": j["name"], "extra": " ".join(j["extra"]),
                        "greedy": -1, "tf": -1.0, "gap": 0.0, "pos0": 0.0,
                        "error": "%s: %s" % (type(e).__name__, e)})
    return res


def show(title, res):
    rows = ["", "### %s" % title, "",
            "| variant | args | greedy /%d | TF content | best gap | pos0 |" % PAIRS,
            "|---|---|---:|---:|---:|---:|"]
    for r in sorted(res, key=lambda x: (-x["greedy"], -x["tf"])):
        rows.append("| %s | `%s` | **%d** | %.2f%% | %+.2f | %.2f%% |"
                    % (r["name"], r["extra"], r["greedy"], r["tf"], r["gap"],
                       r["pos0"]))
    t = "\n".join(rows)
    report(t)
    print(t, flush=True)


LADDER = [
    # Non-negative weights, with and without the row normalisation. The leaky
    # score map lets negatives through, and negative weights cancel inside an
    # L1-normalised row, so the mass that was fixed is not the mass reaching V.
    ("rung1 — non-negative weights x row normalisation", [
        ("relu_plain", []),
        ("relu_mass12", ["--attn-mass-bits", "12"]),
        ("relu_mass14", ["--attn-mass-bits", "14"]),
        ("leaky_plain", ["--score-leak", "3"]),
        ("leaky_mass14", ["--score-leak", "3", "--attn-mass-bits", "14"]),
    ]),
    # Score resolution. pow2norm has always cut the scores to TB+2 = 7 bits
    # before the activation sees them: 128 levels for a whole row.
    ("rung2 — attention score resolution", [
        ("score10", ["--attn-score-bits", "10"]),
        ("score13", ["--attn-score-bits", "13"]),
        ("score16", ["--attn-score-bits", "16"]),
        ("score13_mass14", ["--attn-score-bits", "13", "--attn-mass-bits", "14"]),
        ("score16_leaky", ["--attn-score-bits", "16", "--score-leak", "3"]),
    ]),
    # Adjoint and state width, in case the loss of resolution is downstream of
    # the scores rather than in them.
    ("rung3 — adjoint and accumulator width", [
        ("hb24", ["--hidden-bits", "24"]),
        ("hb24_score13", ["--hidden-bits", "24", "--attn-score-bits", "13"]),
        ("lr2_score13", ["--lr-bits", "2", "--attn-score-bits", "13"]),
        ("hb12_score13", ["--hidden-bits", "12", "--attn-score-bits", "13"]),
    ]),
    # Longer budget on whatever looked least bad, in case it is only slow.
    ("rung4 — longer budget", [
        ("score13_long", ["--attn-score-bits", "13"]),
        ("score16_long", ["--attn-score-bits", "16"]),
        ("score13_mass14_long", ["--attn-score-bits", "13", "--attn-mass-bits", "14"]),
    ]),
]


def main():
    report("\n---\n\n## 9. Attention formulation search (attn_driver.py)\n\n"
           "Started %s. %d pairs, d64, quick evaluation; a rung takes a couple "
           "of minutes. Pass: greedy >= %d/%d, which the plain formulation "
           "cannot reach."
           % (time.strftime("%Y-%m-%d %H:%M:%S"), PAIRS, PASS_GREEDY, PAIRS))
    for i, (title, rung) in enumerate(LADDER):
        log("=== %s ===" % title)
        epochs = 2400 if "longer budget" in title else 600
        every = 400 if epochs > 600 else 100
        jobs = [launch(n, e, epochs, every) for n, e in rung]
        res = collect(jobs)
        show(title, res)
        good = [r for r in res if r["greedy"] >= PASS_GREEDY]
        if good:
            log("PASSED: %s" % [g["name"] for g in good])
            report("\n**Rung passed** by: %s. Next: scale this formulation to "
                   "128 and 512 pairs.\n" % ", ".join(g["name"] for g in good))
            return 0
        best = max(res, key=lambda r: (r["greedy"], r["tf"]))
        log("rung failed; best %s greedy %d tf %.2f%% gap %+.2f"
            % (best["name"], best["greedy"], best["tf"], best["gap"]))
        report("\nRung failed. Best: **%s** greedy %d/%d, TF content %.2f%%, "
               "gap %+.2f.\n" % (best["name"], best["greedy"], PAIRS, best["tf"],
                                 best["gap"]))
    report("\nLadder exhausted. Every integer attention variant tried still "
           "fails to generate what teacher forcing memorises, while the matched "
           "float model with softmax succeeds. The next separation to make is "
           "which single property of softmax is responsible, by adding them to "
           "the float mirror one at a time.\n")
    log("ladder exhausted")
    return 1


if __name__ == "__main__":
    sys.exit(main())
