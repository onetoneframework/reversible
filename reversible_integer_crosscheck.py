"""Measure NumPy int8_reversible.py and PyTorch reversible_integer_final.py
on the same configurations and compare 5-seed means.

Both sides are re-measured here; no number is carried over from a report.
"""
import argparse, json, statistics, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

# name -> (groups, hidden, perm)
CONFIGS = {
    "dense g=1 H64":        (1, 64, "default"),
    "g=4 H64 transpose":    (4, 64, "transpose"),
    "g=8 H256 transpose":   (8, 256, "transpose"),
    "g=8 H64 default perm": (8, 64, "default"),
}


def numpy_cmd(seed, K, epochs, sbits, groups, hidden, perm):
    return [PY, os.path.join(HERE, "int8_reversible.py"),
            "--K", str(K), "--epochs", str(epochs), "--seed", str(seed),
            "--state-bits", str(sbits), "--weight-bits", "8",
            "--grad-bits", "8", "--accum-bits", "32",
            "--groups", str(groups), "--hidden", str(hidden), "--perm", perm]


def torch_cmd(seed, K, epochs, sbits, groups, hidden, perm):
    return [PY, os.path.join(HERE, "reversible_integer_final.py"),
            "--method", "reversible", "--K", str(K), "--epochs", str(epochs),
            "--seed", str(seed), "--state-bits", str(sbits), "--weight-bits", "8",
            "--hidden-bits", "8", "--accum-bits", "32",
            "--groups", str(groups), "--hidden-width", str(hidden), "--perm", perm,
            "--threads", "1"]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr.strip()[-400:]}
    return json.loads(p.stdout.strip().splitlines()[-1])


def sweep(impl, config_names, seeds, K, epochs, sbits, workers):
    jobs = []
    for name in config_names:
        g, h, perm = CONFIGS[name]
        for s in seeds:
            builder = numpy_cmd if impl == "numpy" else torch_cmd
            jobs.append((name, s, builder(s, K, epochs, sbits, g, h, perm)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda j: run(j[2]), jobs))
    out = {}
    for (name, seed, _cmd), res in zip(jobs, results):
        out.setdefault(name, []).append({"seed": seed, **res})
    return out


def summarize(runs):
    accs = [r["accuracy"] for r in runs if "accuracy" in r]
    errs = [r for r in runs if "accuracy" not in r]
    if not accs:
        return {"error": errs[0].get("error", "no runs") if errs else "no runs"}
    return {
        "n": len(accs),
        "mean_pct": round(100 * statistics.mean(accs), 2),
        "std_pct": round(100 * statistics.stdev(accs), 2) if len(accs) > 1 else 0.0,
        "min_pct": round(100 * min(accs), 2),
        "max_pct": round(100 * max(accs), 2),
        "per_seed_pct": [round(100 * a, 2) for a in accs],
        "wrap_frac": [r.get("state_wrap_frac") for r in runs if "accuracy" in r],
        "params": sorted({r.get("parameter_count") or r.get("nonzero_weights")
                          for r in runs if "accuracy" in r}),
        "errors": len(errs),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", nargs="+", default=["numpy", "torch"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--state-bits", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="_crosscheck.json")
    a = ap.parse_args()

    report = {"K": a.K, "epochs": a.epochs, "state_bits": a.state_bits,
              "seeds": a.seeds, "results": {}}
    for impl in a.impls:
        raw = sweep(impl, a.configs, a.seeds, a.K, a.epochs, a.state_bits, a.workers)
        report["results"][impl] = {k: summarize(v) for k, v in raw.items()}
        for name in a.configs:
            s = report["results"][impl][name]
            print(f"{impl:<6} sbits={a.state_bits:<3} {name:<22} "
                  + (f"{s['mean_pct']:>6.2f} +/- {s['std_pct']:<5.2f} "
                     f"params={s['params']} seeds={s['per_seed_pct']}"
                     if "mean_pct" in s else f"ERROR {s['error'][:120]}"), flush=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print("wrote", a.out)
