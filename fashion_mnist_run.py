#!/usr/bin/env python3
"""
Fashion-MNIST driver for the reversible integer network.

The input is zero-padded from 784 to M rather than projected, so v0 = [x_pad, 0]
and the whole network stays exactly reversible: there is no non-invertible
first layer whose activation would have to be stored.

  python fashion_mnist_run.py --config dense --seeds 0 1 2 3 4
  python fashion_mnist_run.py --config g8_matched --epochs 2
  python fashion_mnist_run.py --config float
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fashion_mnist_data as FM          # noqa: E402
import reversible_integer_final as R     # noqa: E402

# name -> (groups, hidden, perm_mode)
# hidden follows the two conditions carried over from digits: H >= M and
# H/g >= 16. g8_matched raises H so the MAC budget is half of dense, mirroring
# the digits "g=8 H256" point; g8_default is the dispersion control.
CONFIGS = {
    "dense":      (1, 832, "transpose"),
    "g4":         (4, 832, "transpose"),
    "g8_matched": (8, 3328, "transpose"),
    "g8_default": (8, 832, "default"),
    # Same g and H as g8_default, so identical MAC and parameters;
    # only the inter-layer permutation differs. This is the control
    # that isolates the dispersion condition from the MAC reduction.
    "g8_disp":    (8, 832, "transpose"),
    "float":      (1, 832, "transpose"),
}


def build(args):
    xtr, ytr, xte, yte, info = FM.load(
        M=args.M, input_shift=args.input_shift,
        train_limit=args.train_limit, test_limit=args.test_limit)
    R.set_dims(args.M)
    R.set_dataset(xtr, ytr, xte, yte, batch=args.batch)
    return info


def run_one(args, seed):
    groups, hidden, perm = CONFIGS[args.config]
    if args.hidden:
        hidden = args.hidden
    if args.config == "float":
        R.set_config(groups=1, hidden=hidden, perm_mode=perm)
        res = R.train_float(args.K, args.epochs, seed)
        res["config"] = args.config
        res["hidden"] = hidden
        return res

    R.set_config(state_bits=args.state_bits, weight_bits=args.weight_bits,
                 accum_bits=args.accum_bits, groups=groups, hidden=hidden,
                 perm_mode=perm, phase_step=args.phase_step)
    res = R.train_integer("reversible", args.K, args.epochs, seed,
                          lr_bits=args.lr_bits, hidden_bits=args.hidden_bits,
                          xscale=1, fused=True, step_mode="per-layer")
    res["config"] = args.config
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="dense")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--M", type=int, default=832)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=20000)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--input-shift", type=int, default=0)
    ap.add_argument("--state-bits", type=int, default=12)
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--accum-bits", type=int, default=32)
    ap.add_argument("--hidden-bits", type=int, default=8)
    ap.add_argument("--lr-bits", type=int, default=2)
    ap.add_argument("--phase-step", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 2)))
    info = build(args)

    rows = []
    for seed in args.seeds:
        t0 = time.time()
        res = run_one(args, seed)
        res["dataset"] = {k: info[k] for k in
                          ("pixels", "M", "pad", "input_shift", "train", "test")}
        res["batch"] = args.batch
        res["wall_seconds"] = round(time.time() - t0, 1)
        rows.append(res)
        print(json.dumps(res), flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
