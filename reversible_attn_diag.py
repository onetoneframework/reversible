#!/usr/bin/env python3
"""
Where the attention gradient dies.

Instruments the attention backward stage by stage, recording each tensor BEFORE
and AFTER normalisation, plus telemetry for every norm_block call. The traced
backward is asserted bit-exact against the production AttnCoupling.backward
first, so the numbers describe the real path and not a lookalike.

For each norm_block: input nonzero %, output nonzero %, max and median bit
length of the input, the chosen exponent, the applied right shift, and the
saturation count charged during the call.

Also computes, without re-running anything, the counterfactual zero fraction
each tensor would have under a per-row exponent instead of one tensor-shared
exponent. A large gap means the shared exponent spans too much dynamic range.

  python reversible_attn_diag.py --K 8
  python reversible_attn_diag.py --K 8 --hidden-bits 16
"""

import argparse
import json

import torch

import reversible_integer_final as R
import reversible_arch as A
from reversible_arch import bmm_backward_shared
from reversible_integer_final import SharedGrad, dense_linear_backward

RECORDS = []
STAGE = {"name": "?", "sub": 0}


def bitlen_t(x):
    return R.bitlen_arr(torch.abs(x))


def tensor_stats(name, m, e, bits):
    """Zero fraction now, and what it would be with a per-row exponent."""
    total = m.numel()
    nz = int((m != 0).sum().item())
    bl = bitlen_t(m)
    flat = m.reshape(-1, m.shape[-1])
    blf = bl.reshape(-1, m.shape[-1])
    # Tensor-shared exponent: one shift for everything.
    shared_shift = max(int(bl.max().item()) - bits, 0)
    zero_shared = int((bitlen_t(flat) <= shared_shift).sum().item())
    # Per-row exponent: each row keeps its own scale.
    row_shift = torch.clamp(blf.amax(dim=1, keepdim=True) - bits, min=0)
    zero_row = int((blf <= row_shift).sum().item())
    nzbl = bl[m != 0]
    return {
        "stage": name,
        "exponent": int(e),
        "nonzero_pct": round(100.0 * nz / total, 3),
        "elements": total,
        "max_bitlen": int(bl.max().item()),
        "median_bitlen": (int(nzbl.median().item()) if nzbl.numel() else 0),
        "shared_exponent_shift": shared_shift,
        "would_be_zero_pct_shared": round(100.0 * zero_shared / total, 3),
        "would_be_zero_pct_per_row": round(100.0 * zero_row / total, 3),
    }


_real_norm_block = R.norm_block


def traced_norm_block(m, e, bits):
    sat_before = R.STATS["accum_saturations"]
    bl = bitlen_t(m)
    nz_in = int((m != 0).sum().item())
    out_m, out_e = _real_norm_block(m, e, bits)
    nz_out = int((out_m != 0).sum().item())
    nzbl = bl[m != 0]
    RECORDS.append({
        "kind": "norm_block",
        "stage": f"{STAGE['name']}#{STAGE['sub']}",
        "bits": bits,
        "elements": m.numel(),
        "in_nonzero_pct": round(100.0 * nz_in / m.numel(), 3),
        "out_nonzero_pct": round(100.0 * nz_out / out_m.numel(), 3),
        "max_bitlen_in": int(bl.max().item()),
        "median_bitlen_in": (int(nzbl.median().item()) if nzbl.numel() else 0),
        "exponent_in": int(e),
        "exponent_out": int(out_e),
        "applied_right_shift": int(out_e) - int(e),
        "saturations": R.STATS["accum_saturations"] - sat_before,
    })
    STAGE["sub"] += 1
    return out_m, out_e


def mark(name):
    STAGE["name"] = name
    STAGE["sub"] = 0


def traced_attn_backward(layer, ga, gb, a, cached, hidden_bits, stages):
    """Mirror of AttnCoupling.backward with per-stage recording."""
    a2d, Q, K, V, S, Aw, O, o2d, sq, sk, sv, ss, so, sf = cached
    n = a.shape[0]
    ab = layer.cfg.accum_bits
    hb = hidden_bits

    mark("gWo / gO")
    gb2d = SharedGrad(gb.m.reshape(n * layer.t, layer.d), gb.e)
    stages.append(tensor_stats("gb (into block)", gb2d.m, gb2d.e, hb))
    gO2d, gWo = dense_linear_backward(gb2d, o2d, layer.Wo, sf, hb, 30, ab)
    gO = SharedGrad(gO2d.m.reshape(n, layer.t, layer.d), gO2d.e)
    stages.append(tensor_stats("gO (after norm)", gO.m, gO.e, hb))
    stages.append(tensor_stats("gW_o", gWo[0], gWo[1], 30))

    mark("gA / gV  (O = A@V)")
    gA, gV = bmm_backward_shared(gO, Aw, V, so, hb, ab)
    stages.append(tensor_stats("gA (after norm)", gA.m, gA.e, hb))
    stages.append(tensor_stats("gV (after norm)", gV.m, gV.e, hb))

    mark("ReLU mask on S")
    mask = (S > 0)
    surv = float(mask.sum().item()) / mask.numel()
    nz_before = int((gA.m != 0).sum().item())
    gS = SharedGrad(A.score_act_backward(gA.m, S), gA.e)
    nz_after = int((gS.m != 0).sum().item())
    stages.append({"stage": "ReLU backward", "mask_survival_pct": round(100 * surv, 3),
                   "gA_nonzero_pct": round(100.0 * nz_before / gA.m.numel(), 3),
                   "gS_nonzero_pct": round(100.0 * nz_after / gS.m.numel(), 3),
                   "killed_by_mask_pct": round(100.0 * (nz_before - nz_after)
                                               / max(1, nz_before), 3)})
    stages.append(tensor_stats("gS (post mask)", gS.m, gS.e, hb))

    mark("gQ / gK  (S = Q@K^T)")
    gQ, gKt = bmm_backward_shared(gS, Q, K.transpose(1, 2), ss, hb, ab)
    gK = SharedGrad(gKt.m.transpose(1, 2), gKt.e)
    stages.append(tensor_stats("gQ (after norm)", gQ.m, gQ.e, hb))
    stages.append(tensor_stats("gK (after norm)", gK.m, gK.e, hb))

    parts = []
    for nm, grad, W, shift in (("q", gQ, layer.Wq, sq), ("k", gK, layer.Wk, sk),
                               ("v", gV, layer.Wv, sv)):
        mark(f"gW_{nm}")
        g2d = SharedGrad(grad.m.reshape(n * layer.t, layer.d), grad.e)
        gin, gw = dense_linear_backward(g2d, a2d, W, shift, hb, 30, ab)
        stages.append(tensor_stats(f"gW_{nm}", gw[0], gw[1], 30))
        parts.append((gin, gw))

    mark("merge")
    da = parts[0][0]
    for gin, _gw in parts[1:]:
        m, e = R.align_shared(da.m, da.e, gin.m, gin.e, hb)
        da = SharedGrad(m, e)
    ga2d = SharedGrad(ga.m.reshape(n * layer.t, layer.d), ga.e)
    nm2, ne = R.align_shared(ga2d.m, ga2d.e, da.m, da.e, hb)
    nga = SharedGrad(nm2.reshape(n, layer.t, layer.d), ne)
    return nga, [parts[0][1], parts[1][1], parts[2][1], gWo]


def run(K, seed, hidden_bits, batch, state_bits, weight_bits, accum_bits):
    cfg = R.RunConfig(state_bits=state_bits, weight_bits=weight_bits,
                      accum_bits=accum_bits)
    R.CFG = cfg
    xtr, ytr, _a, _b = A.load_data("attn", 512, 1)
    xb, yb = xtr[:batch], ytr[:batch]
    net = A.make_net("attn", K, seed, cfg)

    # Real backward, for the faithfulness assertion.
    real_g, _ = net.backward(net.forward(xb), yb, hidden_bits)

    # Traced backward over the same inputs.
    RECORDS.clear()
    stages = []
    R.norm_block = traced_norm_block
    A.norm_block = traced_norm_block
    try:
        v = net.forward(xb)
        g = net.output_grad(v, yb, hidden_bits)
        collected = []
        for k in range(len(net.layers) - 1, -1, -1):
            if k < len(net.layers) - 1:
                v = net._permute(v, net.inv_perm)
                g = SharedGrad(net._permute(g.m, net.inv_perm), g.e)
            L = net.layers[k]
            a = v[:, :, :net.half]
            cached = L.parts(a)
            f, rest = cached[0], cached[1:]
            bp = v[:, :, net.half:]
            vin = torch.cat([a, R.wrap_state(bp - f, cfg.state_bits)], dim=2)
            ga, gb = net.split(g)
            per_layer = [] if k == len(net.layers) - 1 else None
            sink = stages if per_layer is not None else []
            nga, gws = traced_attn_backward(L, ga, gb, a, rest, hidden_bits, sink)
            g = net.join(nga, gb, hidden_bits)
            collected.append((k, gws))
            v = vin
        collected.sort(key=lambda t: t[0])
        traced_g = [w for _k, gws in collected for w in gws]
    finally:
        R.norm_block = _real_norm_block
        A.norm_block = _real_norm_block

    faithful = (len(real_g) == len(traced_g) and
                all(torch.equal(x, y) and int(ex) == int(ey)
                    for (x, ex), (y, ey) in zip(real_g, traced_g)))
    overall_nz = sum(int((w[0] != 0).sum().item()) for w in real_g)
    overall_tot = sum(w[0].numel() for w in real_g)
    return {
        "K": K, "seed": seed, "hidden_bits": hidden_bits, "batch": batch,
        "traced_equals_production_backward": bool(faithful),
        "overall_param_grad_nonzero_pct": round(100.0 * overall_nz / overall_tot, 3),
        "stages_last_layer": stages,
        "norm_block_calls_last_layer": [r for r in RECORDS][:40],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-bits", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--state-bits", type=int, default=12)
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--accum-bits", type=int, default=32)
    ap.add_argument("--score-leak", type=int, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(6)
    A.SCORE_LEAK = a.score_leak
    res = run(a.K, a.seed, a.hidden_bits, a.batch, a.state_bits,
              a.weight_bits, a.accum_bits)

    print(f"traced == production backward: {res['traced_equals_production_backward']}")
    print(f"overall parameter-gradient nonzero: {res['overall_param_grad_nonzero_pct']}%")
    print(f"\n--- stage tensors (last layer), hidden_bits={a.hidden_bits} ---")
    print(f"{'stage':<22} {'nz%':>8} {'exp':>6} {'maxbl':>6} {'medbl':>6} "
          f"{'shift':>6} {'zero%shared':>12} {'zero%perrow':>12}")
    for s in res["stages_last_layer"]:
        if "nonzero_pct" not in s:
            print(f"{s['stage']:<22} mask_survival={s['mask_survival_pct']}% "
                  f"gA_nz={s['gA_nonzero_pct']}% -> gS_nz={s['gS_nonzero_pct']}% "
                  f"(killed {s['killed_by_mask_pct']}%)")
            continue
        print(f"{s['stage']:<22} {s['nonzero_pct']:>8.3f} {s['exponent']:>6} "
              f"{s['max_bitlen']:>6} {s['median_bitlen']:>6} "
              f"{s['shared_exponent_shift']:>6} {s['would_be_zero_pct_shared']:>12.3f} "
              f"{s['would_be_zero_pct_per_row']:>12.3f}")
    print(f"\n--- norm_block telemetry (last layer) ---")
    print(f"{'stage':<20} {'bits':>5} {'in_nz%':>8} {'out_nz%':>8} {'maxbl':>6} "
          f"{'medbl':>6} {'e_in':>6} {'e_out':>6} {'shift':>6} {'sat':>5}")
    for r in res["norm_block_calls_last_layer"]:
        print(f"{r['stage']:<20} {r['bits']:>5} {r['in_nonzero_pct']:>8.3f} "
              f"{r['out_nonzero_pct']:>8.3f} {r['max_bitlen_in']:>6} "
              f"{r['median_bitlen_in']:>6} {r['exponent_in']:>6} {r['exponent_out']:>6} "
              f"{r['applied_right_shift']:>6} {r['saturations']:>5}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
