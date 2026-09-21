"""Run the Fashion-MNIST config x seed grid as a pool of subprocesses."""
import argparse, json, os, statistics, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def job(cfg, seed, a):
    cmd = [PY, os.path.join(HERE, "fashion_mnist_run.py"),
           "--config", cfg, "--K", str(a.K), "--epochs", str(a.epochs),
           "--seeds", str(seed), "--batch", str(a.batch), "--M", str(a.M),
           "--train-limit", str(a.train_limit), "--input-shift", str(a.input_shift),
           "--threads", str(a.threads_per_job)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"config": cfg, "seed": seed, "error": p.stderr.strip()[-500:]}
    return json.loads(p.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["dense", "g4", "g8_matched", "g8_default", "float"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--M", type=int, default=832)
    ap.add_argument("--train-limit", type=int, default=20000)
    ap.add_argument("--input-shift", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads-per-job", type=int, default=3)
    ap.add_argument("--out", default="fashion_mnist_results.json")
    a = ap.parse_args()

    jobs = [(c, s) for c in a.configs for s in a.seeds]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(lambda t: job(t[0], t[1], a), jobs))
    for r in results:
        print(json.dumps(r), flush=True)
    json.dump(results, open(a.out, "w"), indent=2)

    print("\n=== summary ===", flush=True)
    print(f"{'config':<12} {'n':>2} {'acc mean':>9} {'std':>6} {'params':>12} "
          f"{'MAC/sample/layer':>17} {'wrap':>8} {'inv':>5}")
    for c in a.configs:
        rs = [r for r in results if r.get("config") == c and "accuracy" in r]
        if not rs:
            errs = [r for r in results if r.get("config") == c]
            print(f"{c:<12} ERROR {errs[0].get('error','')[:100] if errs else ''}")
            continue
        acc = [r["accuracy"] * 100 for r in rs]
        p = rs[0].get("parameter_count")
        print(f"{c:<12} {len(acc):>2} {statistics.mean(acc):>8.2f}% "
              f"{(statistics.stdev(acc) if len(acc) > 1 else 0):>6.2f} "
              f"{(p or 0):>12,} {((p or 0)//a.K):>17,} "
              f"{str(rs[0].get('state_wrap_frac')):>8} "
              f"{str(rs[0].get('inverse_exact')):>5}")
