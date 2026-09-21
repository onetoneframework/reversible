#!/usr/bin/env python3
"""
Self-driving experiment ladder for the JA->KO reversible-integer system.

Written after three hours of the machine sitting idle because the schedule that
was supposed to start the next experiment never fired. This does not depend on
anything waking up: it runs a batch of experiments in parallel, waits for them,
reads the results, decides, appends to the report, and starts the next batch
itself. If a batch passes its gate it advances; if not it drops to the next rung
of the ladder rather than stopping.

The gate is STAGE 3 from the mandate: 32-pair translation memorised well enough
to be reproduced by *greedy decoding*, which teacher-forced accuracy has already
been shown not to imply.

  python overnight_driver.py --threads-per-job 4
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python",
                  "Python313", "python.exe")
REPORT = os.path.join(HERE, "overnight_translation_report.md")

# STAGE 3 pass condition, from the mandate.
PASS_GREEDY = 30          # of 32
PASS_TF_CONTENT = 95.0


def log(msg):
    stamp = time.strftime("%H:%M:%S")
    print("[%s] %s" % (stamp, msg), flush=True)


def report(text):
    with open(REPORT, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


TAG = ""
SELFTEST = []


def launch(name, extra, threads, pairs=32, epochs=4000, every=250):
    """Start one 32-pair overfit variant. Returns (name, Popen, out_json)."""
    out = os.path.join(HERE, "of32_%s%s.json" % (TAG, name))
    if SELFTEST:
        pairs, epochs, every = 4, 2, 1
        extra = [x for x in extra if x not in ("--epochs", "--report-every")]
        extra = [e for i, e in enumerate(extra)
                 if not (i and extra[i - 1] in ("--epochs", "--report-every"))]
    cmd = [PY, "-X", "utf8", "-u", "reversible_jako_overfit32.py",
           "--pairs", str(pairs), "--epochs", str(epochs),
           "--report-every", str(every), "--threads", str(threads),
           "--lr-bits", "1", "--out", out] + extra
    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1")
    lf = open(os.path.join(HERE, "_of32_%s%s.log" % (TAG, name)), "w", encoding="utf-8")
    ef = open(os.path.join(HERE, "_of32_%s%s.err" % (TAG, name)), "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=lf, stderr=ef)
    log("launched %-14s pid %-6d %s" % (name, p.pid, " ".join(extra) or "(baseline)"))
    return {"name": name, "proc": p, "out": out, "extra": extra}


def workers_alive():
    """True while any overfit worker is still running.

    Read through the WMI provider rather than `tasklist`, whose output on this
    machine is CP949 and blows up subprocess's UTF-8 reader thread.
    """
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -like '*reversible_jako_overfit32*' } | "
          "Measure-Object | ForEach-Object { $_.Count }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, timeout=60)
        return int(r.stdout.decode("utf-8", "replace").strip() or 0) > 0
    except Exception as e:
        log("workers_alive failed (%s); assuming still running" % type(e).__name__)
        return True


def wait_all(jobs, poll=20):
    while any(j["proc"].poll() is None for j in jobs):
        time.sleep(poll)
    for j in jobs:
        j["rc"] = j["proc"].returncode


def read_result(j):
    try:
        with open(j["out"], encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:
        return {"name": j["name"], "error": "%s: %s" % (type(e).__name__, e),
                "greedy": -1, "tf": -1.0}
    hist = d.get("history", [])
    best_greedy = max([h.get("greedy_exact", 0) for h in hist] or [0])
    best_give1 = max([h.get("greedy_given1_exact", 0) for h in hist] or [0])
    best_tf = max([h.get("tf_content_acc", 0.0) for h in hist] or [0.0])
    last = hist[-1] if hist else {}
    return {"name": j["name"], "extra": " ".join(j["extra"]) or "(baseline)",
            "greedy": best_greedy, "give1": best_give1, "tf": best_tf,
            "real": last.get("real_content"), "shuf": last.get("shuffled_content"),
            "zero": last.get("zeroed_content"), "verdict": d.get("verdict", ""),
            "rc": j.get("rc")}


def summarise(batch_name, results):
    rows = ["", "### %s" % batch_name, "",
            "| variant | args | best greedy /32 | best give1 /32 | best TF content | real/shuf/zero (last) |",
            "|---|---|---:|---:|---:|---|"]
    for r in sorted(results, key=lambda x: -x["greedy"]):
        rows.append("| %s | `%s` | **%d** | %d | %.3f%% | %s / %s / %s |" % (
            r["name"], r.get("extra", ""), r["greedy"], r.get("give1", -1),
            r["tf"], r.get("real"), r.get("shuf"), r.get("zero")))
    txt = "\n".join(rows)
    report(txt)
    print(txt, flush=True)
    return txt


def passed(results):
    return [r for r in results
            if r["greedy"] >= PASS_GREEDY and r["tf"] >= PASS_TF_CONTENT]


# The ladder. Each rung is a list of (name, extra args). One variable changes
# per variant relative to the baseline unless the name says otherwise.
LADDER = [
    ("batch2 — combinations and stronger settings", [
        ("pos16_ss25", ["--positional", "16", "--scheduled-sampling", "25"]),
        ("pos16_pc50", ["--positional", "16", "--prefix-corrupt", "50"]),
        ("ss50", ["--scheduled-sampling", "50"]),
        ("pc75", ["--prefix-corrupt", "75"]),
    ]),
    ("batch3 — capacity and precision on the plain task", [
        ("hb20", ["--hidden-bits", "20"]),
        ("hb24", ["--hidden-bits", "24"]),
        ("lr2", ["--lr-bits", "2"]),
        ("pos16_lr2", ["--positional", "16", "--lr-bits", "2"]),
    ]),
    ("batch4 — longer budget on the two best plain settings", [
        ("base_long", ["--epochs", "12000", "--report-every", "1000"]),
        ("pos16_long", ["--positional", "16", "--epochs", "12000",
                        "--report-every", "1000"]),
        ("pos16_ss25_long", ["--positional", "16", "--scheduled-sampling", "25",
                             "--epochs", "12000", "--report-every", "1000"]),
        ("pos32_dec6", ["--positional", "16", "--k-dec", "6",
                        "--dec-pattern", "ccafca"]),
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads-per-job", type=int, default=4)
    ap.add_argument("--wait-for-running", action="store_true",
                    help="a batch is already in flight; wait for it and read it")
    ap.add_argument("--running-names", nargs="*",
                    default=["base", "pc50", "ss25", "pos16"])
    ap.add_argument("--tag", default="", help="prefix for output files")
    ap.add_argument("--self-test", action="store_true",
                    help="tiny/fast variants, to prove the ladder advances")
    a = ap.parse_args()
    global TAG, SELFTEST
    TAG = a.tag
    SELFTEST = [1] if a.self_test else []

    report("\n---\n\n## 6. Autonomous ladder (overnight_driver.py)\n\n"
           "Started %s. Gate: 32-pair greedy reproduction >= %d/32 with teacher-"
           "forced content >= %.0f%%. Batches run in parallel; the driver "
           "advances or drops to the next rung without waiting to be woken."
           % (time.strftime("%Y-%m-%d %H:%M:%S"), PASS_GREEDY, PASS_TF_CONTENT))

    if a.wait_for_running:
        log("waiting for the batch already in flight: %s" % a.running_names)
        while True:
            done = all(os.path.exists(os.path.join(HERE, "of32_%s.json" % n))
                       for n in a.running_names)
            if done:
                log("all result files present")
                break
            # Deadlock guard. A variant that crashes never writes its JSON, so
            # waiting on the files alone would hang forever. If no worker is
            # left running, stop waiting and read whatever landed; the missing
            # ones are reported as errors rather than silently skipped.
            if not workers_alive():
                log("no workers left; reading whatever finished")
                break
            time.sleep(30)
        results = []
        for n in a.running_names:
            results.append(read_result({"name": n, "out": os.path.join(
                HERE, "of32_%s.json" % n), "extra": []}))
        summarise("batch1 — one intervention each, stable optimizer", results)
        good = passed(results)
        if good:
            log("STAGE 3 PASSED by %s" % [g["name"] for g in good])
            report("\n**STAGE 3 PASSED** by: %s\n" % ", ".join(g["name"] for g in good))
            return 0

    for title, rung in LADDER:
        log("=== %s ===" % title)
        jobs = [launch(n, e, a.threads_per_job) for n, e in rung]
        wait_all(jobs)
        results = [read_result(j) for j in jobs]
        summarise(title, results)
        good = passed(results)
        if good:
            log("STAGE 3 PASSED by %s" % [g["name"] for g in good])
            report("\n**STAGE 3 PASSED** by: %s\n" % ", ".join(g["name"] for g in good))
            return 0
        best = max(results, key=lambda r: (r["greedy"], r["tf"]))
        log("rung failed; best was %s greedy %d/32 tf %.3f%%"
            % (best["name"], best["greedy"], best["tf"]))
        report("\nRung failed. Best: **%s**, greedy %d/32, TF content %.3f%%. "
               "Dropping to the next rung.\n"
               % (best["name"], best["greedy"], best["tf"]))

    report("\nLadder exhausted without passing STAGE 3. Next: float/autograd "
           "diagnostic to separate the integer training algorithm from the "
           "architecture and objective (mandate section 7).\n")
    log("ladder exhausted")
    return 1


if __name__ == "__main__":
    sys.exit(main())
