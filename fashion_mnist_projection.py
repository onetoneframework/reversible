#!/usr/bin/env python3
"""
Comparison group: 784 -> M integer projection instead of zero padding.

This is deliberately NOT the main configuration. The projection is not
invertible, so the network is reversible only from v0 onward; recovering v0 by
inversion no longer recovers the input. What that costs in practice is measured
here rather than argued: the bytes that must be retained across the backward
pass for the projection's own weight gradient are reported explicitly.

It answers a separate question from the padded run: can the input dimension be
reduced at all, and what does that buy in MAC.

  python fashion_mnist_projection.py --proj-M 128 --seeds 0 1 2 3 4
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fashion_mnist_data as FM          # noqa: E402
import reversible_integer_final as R     # noqa: E402


def train_projection(args, seed, xtr, ytr, xte, yte):
    pixels = xtr.shape[1]
    M = args.proj_M
    R.set_dims(M)
    R.set_config(state_bits=args.state_bits, weight_bits=args.weight_bits,
                 accum_bits=args.accum_bits, groups=args.groups,
                 hidden=args.hidden, perm_mode=args.perm)
    R.reset_stats()

    gen = torch.Generator().manual_seed(seed)
    lim = 8 if args.weight_bits is None else (1 << (args.weight_bits - 4))
    # Projection weight drawn first, then the coupling stack, so the two never
    # share a stream position with the padded run (they are different models).
    Wp = torch.randint(-lim, lim + 1, (M, pixels), dtype=torch.int64, generator=gen)
    layers = [R.Coupling(gen, R.CFG, phase=(k * R.CFG.phase_step) % R.CFG.groups)
              for k in range(args.K)]

    opt_layers = R.BlockSGD(R.parameters(layers), param_bits=12,
                            lr_bits=args.lr_bits, clip=1)
    opt_proj = R.BlockSGD([Wp], param_bits=12, lr_bits=args.lr_bits, clip=1)

    def project(x):
        raw = R.saturate(x @ Wp.T, R.CFG.accum_bits)
        z, sp = R.pow2norm(raw, R.TB + 2)
        return R.wrap_state(z, R.CFG.state_bits), sp

    def forward(x):
        z, sp = project(x)
        v = torch.cat([z, torch.zeros_like(z)], dim=1)
        for k, L in enumerate(layers):
            v = L.forward(v)
            if k < len(layers) - 1:
                v = v[:, R.perm_for(k)]
        return v, sp

    rng = np.random.default_rng(seed)
    t0 = time.time()
    retained = 0
    for _ep in range(args.epochs):
        idx = torch.tensor(rng.permutation(len(xtr)), dtype=torch.long)
        for st in range(0, len(xtr), args.batch):
            bi = idx[st:st + args.batch]
            xb, yb = xtr[bi], ytr[bi]
            final, sp = forward(xb)
            _g, _vin, g = R.backward_reconstruct(
                final, yb, layers, args.hidden_bits, fused=True,
                on_layer_grad=opt_layers.step_layer)

            # Backward through the projection: y = (x @ Wp.T) >> sp.
            # xb and sp must both still be alive here; that is the storage the
            # padded configuration does not need.
            retained = max(retained, xb.numel() * 8 + sp.numel() * 8)
            s = sp.reshape(-1).to(torch.int64)
            sample_e = g.e - s
            E = int(sample_e.max().item())
            aligned = R.rshift0(g.m[:, :M], (E - sample_e).reshape(-1, 1))
            gWp = R.saturate(aligned.T @ xb, R.CFG.accum_bits)
            opt_proj.step([R.norm_block(gWp, E, 30)])

    with torch.no_grad():
        vt, _ = forward(xte)
        acc = float((vt[:, M:M + R.NCLASS].argmax(1) == yte).float().mean().item())

    coupling_params = sum(w.numel() for w in R.parameters(layers))
    proj_params = Wp.numel()
    return {
        "config": f"proj{M}",
        "seed": seed,
        "K": args.K,
        "epochs": args.epochs,
        "accuracy": acc,
        "proj_M": M,
        "input_pixels": pixels,
        "hidden": args.hidden,
        "groups": args.groups,
        "parameter_count": coupling_params + proj_params,
        "coupling_parameter_count": coupling_params,
        "projection_parameter_count": proj_params,
        "mac_per_sample_total": coupling_params + proj_params,
        "state_wrap_frac": (round(R.STATS["state_wrap"] / R.STATS["state_elems"], 6)
                            if R.STATS["state_elems"] else 0.0),
        "accum_saturations": int(R.STATS["accum_saturations"]),
        "bytes_retained_for_projection_backward": int(retained),
        "reversible_from_v0_only": True,
        "wall_seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj-M", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--perm", default="transpose")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--train-limit", type=int, default=20000)
    ap.add_argument("--input-shift", type=int, default=0)
    ap.add_argument("--state-bits", type=int, default=12)
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--accum-bits", type=int, default=32)
    ap.add_argument("--hidden-bits", type=int, default=8)
    ap.add_argument("--lr-bits", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    xtr, ytr, xte, yte, _info = FM.load(
        M=784, input_shift=args.input_shift,
        train_limit=args.train_limit, test_limit=0)

    rows = []
    for seed in args.seeds:
        r = train_projection(args, seed, xtr, ytr, xte, yte)
        rows.append(r)
        print(json.dumps(r), flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
