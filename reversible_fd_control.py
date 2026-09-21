#!/usr/bin/env python3
"""
Control for the finite-difference gradient probe.

The same coordinate-wise probe is run against the MLP coupling, whose backward
is already established (it trains sklearn digits to ~89% and reproduces the
NumPy reference). If the MLP scores near chance too, the probe is measuring
quantisation noise in single coordinates rather than a defect in a backward.
"""

import argparse
import json

import numpy as np
import torch

import reversible_integer_final as R


def hinge_loss(v, y):
    out = v[:, R.M:R.M + R.NCLASS]
    n = len(y)
    masked = out.clone()
    masked[torch.arange(n), y] = torch.iinfo(torch.int64).min // 4
    j = masked.argmax(dim=1)
    margin = out[torch.arange(n), j] + 1 - out[torch.arange(n), y]
    return int(torch.clamp_min(margin, 0).sum().item())


def probe(K, seed, hidden_bits, deltas, probes, batch):
    layers = R.make_layers(K, seed)
    params = R.parameters(layers)
    xb, yb = R.XTR[:batch], R.YTR[:batch]
    grads, _v, _g = R.backward_reconstruct(
        R.forward_reversible(xb, layers, 1), yb, layers, hidden_bits, fused=True)

    out = {}
    for delta in deltas:
        rng = np.random.default_rng(seed)
        agree = total = 0
        for _ in range(probes):
            pi = int(rng.integers(len(params)))
            flat = params[pi].reshape(-1)
            ci = int(rng.integers(flat.numel()))
            gsign = int(torch.sign(grads[pi][0].reshape(-1)[ci]).item())
            if gsign == 0:
                continue
            orig = int(flat[ci].item())
            flat[ci] = orig + delta
            lp = hinge_loss(R.forward_reversible(xb, layers, 1), yb)
            flat[ci] = orig - delta
            lm = hinge_loss(R.forward_reversible(xb, layers, 1), yb)
            flat[ci] = orig
            if lp == lm:
                continue
            total += 1
            agree += int(np.sign(lp - lm) == gsign)
        out[f"delta{delta}"] = {"agreement": round(agree / total, 3) if total else None,
                                "probes_used": total}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-bits", type=int, default=8)
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 8])
    ap.add_argument("--probes", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--state-bits", type=int, default=12)
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--accum-bits", type=int, default=32)
    a = ap.parse_args()

    torch.set_num_threads(2)
    R.set_config(state_bits=a.state_bits, weight_bits=a.weight_bits,
                 accum_bits=a.accum_bits)
    rows = []
    for K in a.K:
        r = {"arch": "mlp", "K": K, **probe(K, a.seed, a.hidden_bits,
                                            a.deltas, a.probes, a.batch)}
        rows.append(r)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
