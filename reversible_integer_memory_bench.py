#!/usr/bin/env python3
"""
Memory and speed measurement for exact reversible integer training.

Variants compared. All use identical integer arithmetic and an identical
gradient representation; they differ only in how forward activations reach the
backward pass, and in when the optimizer consumes parameter gradients.

  cached                  forward stores (a, z, h, s1, s2) per layer, a kept as a
                          view of the layer input state; backward reads the tape
  cached_lean             same, but a is cloned to a contiguous 64-wide tensor so
                          the tape does not pin the full 128-wide state
  cached_perlayer         cached tape, optimizer steps inside the layer callback
  reconstructed           reference backward_reconstruct(fused=False): the
                          original two-call inverse-then-parts recomputation
  reconstructed_fused     reference backward_reconstruct(fused=True): one
                          parts() call per layer
  reconstructed_perlayer  fused reconstruction, optimizer steps inside the layer
                          callback so no O(K) parameter-gradient list is built

The reconstruction paths are the reference functions imported from
reversible_integer_final; this file holds no copy of them. Per-layer activation
sets are read out of the reference function's own live frame through the
on_layer_grad callback, so instrumentation never forks the logic under test.
The cached tape variants are this benchmark's own strict no-recompute baseline
and have no counterpart in the reference module.

Every reported number is measured. Nothing is estimated or filled in from
theory; a field that could not be measured is left empty in the CSV.

tracemalloc is deliberately not used for the tensor payloads: torch allocates
them through the C++ allocator, so they never appear in Python allocation
traces. Live bytes are tracked with a TorchDispatchMode that registers every
aten output storage and drops it when its last referencing tensor is collected.

  python reversible_integer_memory_bench.py --mode verify
  python reversible_integer_memory_bench.py --mode all --batches 60 --repeats 5
  python reversible_integer_memory_bench.py --mode rss --variants reconstructed_perlayer --depths 64
"""

import argparse
import csv
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.torch_memory import LiveByteTracker, dedup_bytes  # noqa: E402

import reversible_integer_final as REF  # noqa: E402
from reversible_integer_final import (  # noqa: E402
    BATCH,
    H,
    M,
    XTR,
    YTR,
    BlockSGD,
    SharedGrad,
    align_shared,
    concat_shared,
    backward_reconstruct,
    forward_reversible,
    invperm_for,
    linear_backward_shared,
    make_input,
    make_layers,
    multiclass_hinge_output_grad,
    parameters,
    perm_for,
    set_config,
    wrap_state,
)

DEPTHS = (8, 16, 32, 64)
HIDDEN_BITS = 9
XSCALE = 1

# Batch size actually used by this benchmark. Defaults to the constant the
# reference implementation trains with; --batch overrides it here only, so the
# reference module is never modified.
BENCH_BATCH = BATCH

# Names of the activation working set inside a reconstruction loop iteration.
ACTIVATION_LOCALS = ("v", "vin", "a", "z", "h", "s1", "s2", "f")


def use_problem(m, xtr, ytr, batch):
    """
    Point the benchmark at a different problem shape.

    M and the dataset are imported by value at module load, so switching the
    reference module's dimensions is not enough on its own; these names have to
    be rebound here too.
    """
    global M, XTR, YTR, BENCH_BATCH
    REF.set_dims(m)
    M = REF.M
    XTR = xtr.to(torch.int64)
    YTR = ytr.to(torch.long)
    BENCH_BATCH = int(batch)
    return {"M": M, "train": tuple(XTR.shape), "batch": BENCH_BATCH}


# -----------------------------------------------------------------------------
# Cached tape baseline (this benchmark's own; no reference counterpart)
# -----------------------------------------------------------------------------
def forward_cached_tape(x, layers, xscale=XSCALE, lean=False):
    """
    Forward that keeps (a, z, h, s1, s2) per layer.

    The per-layer arithmetic is Coupling.parts, the same call the reversible
    forward makes through Coupling.forward, so both paths run identical integer
    operations on identical inputs.

    lean=False stores a as a view of the 128-wide layer input, which costs no
    copy but pins that whole buffer. lean=True clones a to a contiguous 64-wide
    tensor instead.
    """
    v = make_input(x, xscale)
    tape = []
    for k, layer in enumerate(layers):
        a, b = v[:, :M], v[:, M:]
        f, z, h, s1, s2 = layer.parts(a)
        tape.append((a.clone() if lean else a, z, h, s1, s2))
        v = torch.cat([a, wrap_state(b + f, layer.cfg.state_bits)], dim=1)
        if k < len(layers) - 1:
            v = v[:, perm_for(k)]
    return v, tape


def backward_cached_tape(final_v, tape, y, layers, hidden_bits=HIDDEN_BITS,
                         on_layer_grad=None):
    """
    Backward that reads the tape. No inversion and no recomputation.

    Mirrors the reference backward_cached callback contract: when
    on_layer_grad is given, parameter gradients are handed over per layer and no
    O(K) list is built.
    """
    g = multiclass_hinge_output_grad(final_v, y, hidden_bits)
    grads = [] if on_layer_grad is None else None

    for k in range(len(layers) - 1, -1, -1):
        if k < len(layers) - 1:
            g.permute_(invperm_for(k))
        layer = layers[k]
        a, z, h, s1, s2 = tape[k]

        ga = SharedGrad(g.m[:, :M], g.e)
        gb = SharedGrad(g.m[:, M:], g.e)
        gh, gVs = linear_backward_shared(gb, h, layer, "V", s2, hidden_bits)
        gh.m = gh.m * (z > 0)
        da, gUs = linear_backward_shared(gh, a, layer, "U", s1, hidden_bits)
        nga, ne = align_shared(ga.m, ga.e, da.m, da.e, hidden_bits)
        fm, fe = concat_shared(nga, ne, gb.m, gb.e, hidden_bits)
        g = SharedGrad(fm, fe)

        if on_layer_grad is None:
            grads.append((k, gUs, gVs))
        else:
            on_layer_grad(k, gUs, gVs)

    if on_layer_grad is not None:
        return None, g
    grads.sort(key=lambda t: t[0])
    return [q for _, gus, gvs in grads for q in (*gus, *gvs)], g


# -----------------------------------------------------------------------------
# Variant table
# -----------------------------------------------------------------------------
# name -> (kind, lean, fused, per_layer)
VARIANT_SPEC = {
    "cached":                 ("tape",  False, None,  False),
    "cached_lean":            ("tape",  True,  None,  False),
    "cached_perlayer":        ("tape",  False, None,  True),
    "reconstructed":          ("recon", None,  False, False),
    "reconstructed_fused":    ("recon", None,  True,  False),
    "reconstructed_perlayer": ("recon", None,  True,  True),
}
ALL_VARIANTS = tuple(VARIANT_SPEC)


def run_forward(variant, xb, layers):
    kind, lean, _fused, _per = VARIANT_SPEC[variant]
    if kind == "tape":
        return forward_cached_tape(xb, layers, XSCALE, lean=lean)
    return forward_reversible(xb, layers, XSCALE), None


def run_backward(variant, final, tape, yb, layers, on_layer_grad=None):
    """Returns (grads_or_None, hidden_grad)."""
    kind, _lean, fused, _per = VARIANT_SPEC[variant]
    if kind == "tape":
        return backward_cached_tape(final, tape, yb, layers, HIDDEN_BITS,
                                    on_layer_grad=on_layer_grad)
    grads, _vin, g = backward_reconstruct(final, yb, layers, HIDDEN_BITS,
                                          fused=fused, on_layer_grad=on_layer_grad)
    return grads, g


def activation_set(variant, tape, frame):
    """
    The activation tensors alive at this point in the backward pass.

    For reconstruction this is read from the reference function's own frame; for
    the cached tape it is the whole tape, which stays live for the entire pass.
    """
    kind = VARIANT_SPEC[variant][0]
    if kind == "tape":
        return [t for entry in tape for t in entry]
    loc = frame.f_locals
    return [loc.get(name) for name in ACTIVATION_LOCALS]


# -----------------------------------------------------------------------------
# Required verification: every layer must match bit for bit
# -----------------------------------------------------------------------------
TRACE_TENSOR_FIELDS = ("a", "z", "h", "s1", "s2", "g_m")
TRACE_TENSOR_LIST_FIELDS = ("gU_m", "gV_m")
TRACE_SCALAR_FIELDS = ("gU_e", "gV_e", "g_e")


def _tensor_lists_equal(xs, ys):
    return len(xs) == len(ys) and all(torch.equal(x, y) for x, y in zip(xs, ys))


def collect_trace(variant, K, seed, xb, yb):
    """Per-layer snapshot, built entirely from the layer callback."""
    layers = make_layers(K, seed)
    kind = VARIANT_SPEC[variant][0]
    trace = []
    tape_ref = {}

    def cb(k, gUs, gVs):
        frame = sys._getframe(1)
        loc = frame.f_locals
        if kind == "tape":
            a, z, h, s1, s2 = tape_ref["tape"][k]
        else:
            a, z, h, s1, s2 = (loc["a"], loc["z"], loc["h"], loc["s1"], loc["s2"])
        trace.append({
            "k": k,
            "a": a.clone(), "z": z.clone(), "h": h.clone(),
            "s1": s1.clone(), "s2": s2.clone(),
            "gU_m": [m.clone() for m, _ in gUs], "gU_e": [int(e) for _, e in gUs],
            "gV_m": [m.clone() for m, _ in gVs], "gV_e": [int(e) for _, e in gVs],
            "g_m": loc["fm"].clone(), "g_e": int(loc["fe"]),
        })

    final, tape = run_forward(variant, xb, layers)
    tape_ref["tape"] = tape
    _grads, gfinal = run_backward(variant, final, tape, yb, layers, on_layer_grad=cb)
    return final, trace, gfinal


def verify_depth(K, seed=0):
    """
    One batch, one seed, all layers. Compares the cached tape against every
    other variant and reports the first layer that diverges.
    """
    xb, yb = XTR[:BENCH_BATCH], YTR[:BENCH_BATCH]
    final_c, trace_c, gfin_c = collect_trace("cached", K, seed, xb, yb)
    checks = []

    for variant in ALL_VARIANTS:
        if variant == "cached":
            continue
        final_r, trace_r, gfin_r = collect_trace(variant, K, seed, xb, yb)
        result = {
            "K": K,
            "seed": seed,
            "variant": variant,
            "final_state_equal": bool(torch.equal(final_c, final_r)),
            "layers_compared": len(trace_r),
            "first_divergent_layer": None,
            "first_divergent_field": None,
            "passed": False,
        }
        if len(trace_c) != len(trace_r):
            result["first_divergent_field"] = "trace_length"
            checks.append(result)
            continue

        diverged = False
        for sc, sr in zip(trace_c, trace_r):
            for field in TRACE_TENSOR_FIELDS:
                if not torch.equal(sc[field], sr[field]):
                    result["first_divergent_layer"] = int(sr["k"])
                    result["first_divergent_field"] = field
                    diverged = True
                    break
            if diverged:
                break
            for field in TRACE_TENSOR_LIST_FIELDS:
                if not _tensor_lists_equal(sc[field], sr[field]):
                    result["first_divergent_layer"] = int(sr["k"])
                    result["first_divergent_field"] = field
                    diverged = True
                    break
            if diverged:
                break
            for field in TRACE_SCALAR_FIELDS:
                if sc[field] != sr[field]:
                    result["first_divergent_layer"] = int(sr["k"])
                    result["first_divergent_field"] = field
                    diverged = True
                    break
            if diverged:
                break

        result["parameter_gradients_equal"] = not diverged
        result["input_gradient_mantissa_equal"] = bool(torch.equal(gfin_c.m, gfin_r.m))
        result["input_gradient_exponent_equal"] = int(gfin_c.e) == int(gfin_r.e)
        result["passed"] = bool(
            not diverged
            and result["final_state_equal"]
            and result["input_gradient_mantissa_equal"]
            and result["input_gradient_exponent_equal"]
        )
        checks.append(result)
    return checks


def verify_step_mode(K, seed=0, steps=12):
    """
    End-to-end check that per-layer stepping equals batched stepping.

    verify_depth compares gradients with the callback used as a collector, so it
    never exercises the stepping order. This runs real training both ways and
    compares every weight afterwards.
    """
    checks = []
    for variant in ALL_VARIANTS:
        kind, lean, fused, per_layer = VARIANT_SPEC[variant]
        if not per_layer:
            continue
        batched_name = "cached" if kind == "tape" else "reconstructed_fused"

        finals = {}
        for name in (batched_name, variant):
            is_per_layer = VARIANT_SPEC[name][3]
            layers = make_layers(K, seed)
            params = parameters(layers)
            opt = BlockSGD(params, param_bits=12, lr_bits=2, clip=1)
            cb = opt.step_layer if is_per_layer else None
            for bi in batch_indices(steps):
                xb, yb = XTR[bi], YTR[bi]
                final, tape = run_forward(name, xb, layers)
                grads, _g = run_backward(name, final, tape, yb, layers,
                                         on_layer_grad=cb)
                if not is_per_layer:
                    opt.step(grads)
            finals[name] = [w.clone() for w in params]

        a, b = finals[batched_name], finals[variant]
        mismatch = next((i for i, (x, y) in enumerate(zip(a, b))
                         if not torch.equal(x, y)), None)
        checks.append({
            "K": K, "seed": seed, "variant": variant,
            "check": "per_layer_step_equals_batched_step",
            "compared_against": batched_name,
            "training_steps": steps,
            "first_mismatched_parameter_index": mismatch,
            "first_mismatched_layer": None if mismatch is None else mismatch // 2,
            "passed": mismatch is None,
        })
    return checks


def run_verification(depths, seed):
    all_checks = []
    ok = True
    for K in depths:
        for check in verify_step_mode(K, seed):
            all_checks.append(check)
            status = "PASS" if check["passed"] else "FAIL"
            line = (f"[{status}] K={check['K']:>3} {check['variant']:<24} "
                    f"step-mode vs {check['compared_against']} over "
                    f"{check['training_steps']} steps: weights equal="
                    f"{check['passed']}")
            if not check["passed"]:
                ok = False
                line += (f"  DIVERGED at layer {check['first_mismatched_layer']} "
                         f"(param index {check['first_mismatched_parameter_index']})")
            print(line, flush=True)
        for check in verify_depth(K, seed):
            all_checks.append(check)
            status = "PASS" if check["passed"] else "FAIL"
            line = (
                f"[{status}] K={check['K']:>3} {check['variant']:<24} "
                f"layers={check['layers_compared']:>3} "
                f"final_state={check['final_state_equal']} "
                f"per_layer_grads={check.get('parameter_gradients_equal')} "
                f"input_grad_m={check.get('input_gradient_mantissa_equal')} "
                f"input_grad_e={check.get('input_gradient_exponent_equal')}"
            )
            if not check["passed"]:
                ok = False
                line += (
                    f"  DIVERGED at layer {check['first_divergent_layer']} "
                    f"field {check['first_divergent_field']}"
                )
            print(line, flush=True)
    return ok, all_checks


# -----------------------------------------------------------------------------
# Memory measurement
# -----------------------------------------------------------------------------
def measure_memory(variant, K, seed=0):
    per_layer = VARIANT_SPEC[variant][3]
    REF.reset_stats()
    layers = make_layers(K, seed)
    params = parameters(layers)
    opt = BlockSGD(params, param_bits=12, lr_bits=2, clip=1)
    xb, yb = XTR[:BENCH_BATCH].clone(), YTR[:BENCH_BATCH].clone()

    param_bytes = dedup_bytes(params)
    optimizer_bytes = dedup_bytes(opt.acc)

    activation_peak = {"v": 0}
    grad_buffer_peak = {"v": 0}
    held_grads = []          # batched: mirrors the list the reference would build
    tape_ref = {"tape": None}

    width_probe = {"elems": 0, "maxabs": 0}

    def probe(k, gUs, gVs):
        frame = sys._getframe(1)
        act_tensors = [t for t in activation_set(variant, tape_ref["tape"], frame)
                       if isinstance(t, torch.Tensor)]
        act = dedup_bytes(act_tensors)
        if act > activation_peak["v"]:
            activation_peak["v"] = act
            # Element count and observed magnitude of that same working set,
            # so a packed-width figure can be derived from measured values.
            seen, elems, mx = set(), 0, 0
            for t in act_tensors:
                ptr = t.untyped_storage().data_ptr()
                if ptr in seen:
                    continue
                seen.add(ptr)
                elems += t.untyped_storage().nbytes() // t.element_size()
                if t.numel():
                    mx = max(mx, int(torch.abs(t).max().item()))
            width_probe["elems"] = elems
            width_probe["maxabs"] = mx
        if per_layer:
            # Only this layer's blocks are live; the optimizer consumes them now.
            live = dedup_bytes([m for m, _ in gUs] + [m for m, _ in gVs])
            opt.step_layer(k, gUs, gVs)
        else:
            held_grads.append((k, gUs, gVs))
            live = dedup_bytes([m for _k, gus, gvs in held_grads
                                for m, _ in (*gus, *gvs)])
        if live > grad_buffer_peak["v"]:
            grad_buffer_peak["v"] = live

    tracker = LiveByteTracker()
    tracker.phase = "setup"
    tracker.adopt(params)
    tracker.adopt(opt.acc)
    tracker.adopt([xb, yb])
    setup_bytes = tracker.total

    with tracker:
        tracker.phase = "forward"
        final, tape = run_forward(variant, xb, layers)
        tape_ref["tape"] = tape
        gc.collect()
        activation_at_entry = tracker.bytes_in_phase("forward")
        total_at_entry = tracker.total

        tracker.phase = "backward"
        tracker.begin_region()
        _grads, gfinal = run_backward(variant, final, tape, yb, layers,
                                      on_layer_grad=probe)
        gc.collect()
        peak_backward = tracker.end_region()
        total_at_step = tracker.total

        tracker.phase = "step"
        if not per_layer:
            held_grads.sort(key=lambda t: t[0])
            opt.step([q for _k, gus, gvs in held_grads for q in (*gus, *gvs)])
        peak_total = tracker.peak
        ops = tracker.ops

    groups = layers[0].g
    exponents = [gfinal.e] + [int(e) for _k, gus, gvs in held_grads
                              for _m, e in (*gus, *gvs)]
    # Per layer only one layer's block exponents are ever live at once.
    exponent_count = (1 + 2 * groups) if per_layer else len(exponents)

    # Packed-width figures are DERIVED from the measured element count and the
    # measured magnitude of the live activation set; they are not allocations.
    state_bits = REF.CFG.state_bits
    observed_bits = int(width_probe["maxabs"]).bit_length() + 1
    declared_bits = state_bits if state_bits is not None else 64
    packed_bits = max(observed_bits, 1)

    return {
        "variant": variant,
        "K": K,
        "batch": BENCH_BATCH,
        "step_mode": "per_layer" if per_layer else "batched",
        "activation_peak_bytes_backward": int(activation_peak["v"]),
        "activation_bytes_at_backward_entry": int(activation_at_entry),
        "param_grad_buffer_peak_bytes": int(grad_buffer_peak["v"]),
        "peak_total_bytes": int(peak_total),
        "peak_backward_bytes": int(peak_backward),
        "total_bytes_at_backward_entry": int(total_at_entry),
        "param_bytes": int(param_bytes),
        "optimizer_bytes": int(optimizer_bytes),
        "setup_bytes": int(setup_bytes),
        "total_bytes_at_optimizer_step": int(total_at_step),
        "exponent_count": int(exponent_count),
        "exponent_bytes_as_int64_fields": int(exponent_count * 8),
        "aten_ops": int(ops),
        # Configuration in force for this row.
        "groups": int(groups),
        "hidden": int(REF.CFG.hidden),
        "perm_mode": REF.CFG.perm_mode,
        "state_bits": state_bits,
        "weight_bits": REF.CFG.weight_bits,
        "accum_bits": REF.CFG.accum_bits,
        "state_wrap_frac": (
            round(REF.STATS["state_wrap"] / REF.STATS["state_elems"], 6)
            if REF.STATS["state_elems"] else 0.0
        ),
        # Measured inputs to the packing derivation.
        "activation_peak_elements": int(width_probe["elems"]),
        "activation_peak_max_abs_value": int(width_probe["maxabs"]),
        "activation_observed_bits_needed": int(observed_bits),
        "activation_declared_state_bits": int(declared_bits),
        # Derived, not allocated: element count times the packed width.
        "activation_peak_bytes_packed_observed": int(
            width_probe["elems"] * ((packed_bits + 7) // 8)
        ),
        "activation_peak_bytes_packed_declared": int(
            width_probe["elems"] * ((declared_bits + 7) // 8)
        ),
    }


def validate_probe_free_path(variant, K, seed=0):
    """
    Cross-check: run the true reference call path with no probing callback and
    confirm the tracker reports the same total peak as the probed run.
    """
    per_layer = VARIANT_SPEC[variant][3]
    layers = make_layers(K, seed)
    params = parameters(layers)
    opt = BlockSGD(params, param_bits=12, lr_bits=2, clip=1)
    xb, yb = XTR[:BENCH_BATCH].clone(), YTR[:BENCH_BATCH].clone()

    tracker = LiveByteTracker()
    tracker.adopt(params)
    tracker.adopt(opt.acc)
    tracker.adopt([xb, yb])
    cb = opt.step_layer if per_layer else None
    with tracker:
        final, tape = run_forward(variant, xb, layers)
        grads, _g = run_backward(variant, final, tape, yb, layers, on_layer_grad=cb)
        if not per_layer:
            opt.step(grads)
        return int(tracker.peak), int(tracker.ops)


# -----------------------------------------------------------------------------
# Timing measurement
# -----------------------------------------------------------------------------
def batch_indices(count):
    """Deterministic batch order, cycling the training set as needed."""
    n = len(XTR)
    starts = list(range(0, n - BENCH_BATCH + 1, BENCH_BATCH))
    order = []
    while len(order) < count:
        order.extend(starts)
    return [torch.arange(s, s + BENCH_BATCH) for s in order[:count]]


def measure_time(variant, K, seed, warmup, batches, repeats):
    """
    Timed with the plain reference API and no instrumentation at all.

    For per-layer variants the optimizer runs inside the backward callback, so
    its cost lands in the backward interval and the separate step interval is
    left unmeasured. backward_plus_step_ms_per_batch is the comparable figure
    across both step modes.
    """
    per_layer = VARIANT_SPEC[variant][3]
    order = batch_indices(warmup + batches)
    best = None

    for _rep in range(repeats):
        layers = make_layers(K, seed)
        params = parameters(layers)
        opt = BlockSGD(params, param_bits=12, lr_bits=2, clip=1)
        cb = opt.step_layer if per_layer else None
        fwd_total = bwd_total = step_total = 0.0

        for i, bi in enumerate(order):
            xb, yb = XTR[bi], YTR[bi]

            t0 = time.perf_counter()
            final, tape = run_forward(variant, xb, layers)
            t1 = time.perf_counter()
            grads, _g = run_backward(variant, final, tape, yb, layers,
                                     on_layer_grad=cb)
            t2 = time.perf_counter()
            if not per_layer:
                opt.step(grads)
            t3 = time.perf_counter()

            if i >= warmup:
                fwd_total += t1 - t0
                bwd_total += t2 - t1
                step_total += t3 - t2

        combined = bwd_total + step_total
        if best is None or combined < best[0]:
            best = (combined, bwd_total, fwd_total, step_total)

    combined, bwd_total, fwd_total, step_total = best
    row = {
        "variant": variant,
        "K": K,
        "batch": BENCH_BATCH,
        "step_mode": "per_layer" if per_layer else "batched",
        "warmup_batches": warmup,
        "batches_measured": batches,
        "repeats": repeats,
        "backward_ms_per_batch": round(bwd_total / batches * 1000.0, 6),
        "forward_ms_per_batch": round(fwd_total / batches * 1000.0, 6),
        "backward_plus_step_ms_per_batch": round(combined / batches * 1000.0, 6),
        "forward_backward_step_ms_per_batch": round(
            (fwd_total + combined) / batches * 1000.0, 6
        ),
    }
    # A separate optimizer interval only exists for the batched step mode.
    row["optimizer_step_ms_per_batch"] = (
        "" if per_layer else round(step_total / batches * 1000.0, 6)
    )
    return row


# -----------------------------------------------------------------------------
# Process-level cross-check for the live-storage tracker
# -----------------------------------------------------------------------------
def measure_rss(variant, K, seed, iterations):
    """
    Runs one configuration with no dispatch-mode tracking and reports process
    peak working set, then repeats it with tracking and reports the tracker
    peak. Run this in separate processes per variant and compare the deltas;
    peak RSS is monotonic within a process.
    """
    try:
        import psutil
    except ImportError:
        return {"variant": variant, "K": K, "error": "psutil not installed"}

    proc = psutil.Process()
    per_layer = VARIANT_SPEC[variant][3]

    def rss_peak():
        info = proc.memory_info()
        return int(getattr(info, "peak_wset", getattr(info, "rss", 0)))

    gc.collect()
    baseline_peak = rss_peak()
    baseline_rss = int(proc.memory_info().rss)

    layers = make_layers(K, seed)
    params = parameters(layers)
    opt = BlockSGD(params, param_bits=12, lr_bits=2, clip=1)
    cb = opt.step_layer if per_layer else None
    order = batch_indices(iterations)
    for bi in order:
        xb, yb = XTR[bi], YTR[bi]
        final, tape = run_forward(variant, xb, layers)
        grads, _g = run_backward(variant, final, tape, yb, layers, on_layer_grad=cb)
        if not per_layer:
            opt.step(grads)
    gc.collect()

    after_peak = rss_peak()
    after_rss = int(proc.memory_info().rss)
    tracked_peak, tracked_ops = validate_probe_free_path(variant, K, seed)

    return {
        "variant": variant,
        "K": K,
        "batch": BENCH_BATCH,
        "iterations": iterations,
        "tracker_peak_bytes": tracked_peak,
        "tracker_aten_ops": tracked_ops,
        "process_peak_wset_baseline_bytes": baseline_peak,
        "process_peak_wset_after_bytes": after_peak,
        "process_peak_wset_delta_bytes": after_peak - baseline_peak,
        "process_rss_baseline_bytes": baseline_rss,
        "process_rss_after_bytes": after_rss,
        "process_rss_delta_bytes": after_rss - baseline_rss,
    }


# -----------------------------------------------------------------------------
# CSV output
# -----------------------------------------------------------------------------
ROW_FIELDS = [
    "variant", "K", "batch", "step_mode",
    "groups", "hidden", "perm_mode",
    "state_bits", "weight_bits", "accum_bits", "state_wrap_frac",
    "activation_peak_bytes_backward",
    "activation_peak_elements",
    "activation_peak_max_abs_value",
    "activation_observed_bits_needed",
    "activation_declared_state_bits",
    "activation_peak_bytes_packed_observed",
    "activation_peak_bytes_packed_declared",
    "activation_bytes_at_backward_entry",
    "param_grad_buffer_peak_bytes",
    "peak_total_bytes",
    "peak_backward_bytes",
    "total_bytes_at_backward_entry",
    "total_bytes_at_optimizer_step",
    "param_bytes", "optimizer_bytes", "setup_bytes",
    "exponent_count",
    "exponent_bytes_as_int64_fields",
    "aten_ops",
    "backward_ms_per_batch",
    "optimizer_step_ms_per_batch",
    "backward_plus_step_ms_per_batch",
    "forward_ms_per_batch",
    "forward_backward_step_ms_per_batch",
    "warmup_batches", "batches_measured", "repeats",
]

SCALING_METRICS = [
    "activation_peak_bytes_backward",
    "param_grad_buffer_peak_bytes",
    "peak_total_bytes",
    "peak_backward_bytes",
    "param_bytes",
    "optimizer_bytes",
    "exponent_bytes_as_int64_fields",
    "backward_plus_step_ms_per_batch",
]


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ROW_FIELDS})


def write_scaling(path, rows):
    """
    Ratio of each metric to its own smallest-K value, so constant versus linear
    growth in K is readable directly. Ratios come from measured values only.
    """
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], {})[row["K"]] = row

    depths = sorted({row["K"] for row in rows})
    fields = ["variant", "metric", "base_K", "base_value"]
    fields += [f"K{K}_value" for K in depths]
    fields += [f"K{K}_ratio_vs_base" for K in depths]

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for variant, per_k in by_variant.items():
            base_K = min(per_k)
            for metric in SCALING_METRICS:
                base = per_k[base_K].get(metric, "")
                if base in ("", None):
                    continue
                out = {"variant": variant, "metric": metric,
                       "base_K": base_K, "base_value": base}
                for K in depths:
                    value = per_k.get(K, {}).get(metric, "")
                    out[f"K{K}_value"] = value
                    out[f"K{K}_ratio_vs_base"] = (
                        "" if value in ("", None) or not base
                        else round(value / base, 4)
                    )
                writer.writerow(out)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def raise_priority():
    try:
        import psutil
    except ImportError:
        return "psutil not installed"
    try:
        proc = psutil.Process()
        if hasattr(psutil, "HIGH_PRIORITY_CLASS"):
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            proc.nice(-10)
        return "high"
    except Exception as exc:  # pragma: no cover - depends on OS permissions
        return f"failed: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode",
                    choices=["verify", "memory", "time", "rss", "all"],
                    default="all")
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    ap.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=30,
                    help="rss mode: training steps to run before sampling")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="batch size for this benchmark only; the reference "
                         "module is not modified")
    ap.add_argument("--validate-probe", action="store_true",
                    help="memory mode: also run the probe-free reference path "
                         "and compare tracker peaks")
    # Width and grouping options, forwarded to the reference module's config.
    ap.add_argument("--state-bits", type=int, default=None)
    ap.add_argument("--weight-bits", type=int, default=None)
    ap.add_argument("--accum-bits", type=int, default=None)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--phase-step", type=int, default=0)
    ap.add_argument("--hidden-width", type=int, default=H)
    ap.add_argument("--perm", choices=["default", "transpose"], default="default")
    ap.add_argument("--dataset", choices=["digits", "fashion"], default="digits")
    ap.add_argument("--M", type=int, default=0,
                    help="coupling half-width; required width for --dataset fashion")
    ap.add_argument("--train-limit", type=int, default=4096)
    ap.add_argument("--out-prefix", default="reversible_integer_memory_bench")
    args = ap.parse_args()

    global BENCH_BATCH
    BENCH_BATCH = args.batch
    if args.dataset == "fashion":
        import fashion_mnist_data
        m = args.M or 832
        xtr, ytr, _xte, _yte, _info = fashion_mnist_data.load(
            M=m, train_limit=max(args.train_limit, args.batch), test_limit=1)
        print("# problem: " + json.dumps(use_problem(m, xtr, ytr, args.batch)),
              flush=True)
    elif args.M:
        raise SystemExit("--M only applies to --dataset fashion")
    set_config(
        state_bits=args.state_bits,
        weight_bits=args.weight_bits,
        accum_bits=args.accum_bits,
        groups=args.groups,
        phase_step=args.phase_step,
        hidden=args.hidden_width,
        perm_mode=args.perm,
    )

    torch.set_num_threads(max(1, args.threads))
    priority = raise_priority()
    print(f"# threads={torch.get_num_threads()} priority={priority} "
          f"batch={BENCH_BATCH} seed={args.seed}", flush=True)

    if args.mode == "rss":
        for variant in args.variants:
            for K in args.depths:
                print("rss " + json.dumps(measure_rss(variant, K, args.seed,
                                                      args.iterations)), flush=True)
        return 0

    if args.mode in ("verify", "all"):
        print("# verification: cached tape vs every other variant, all layers",
              flush=True)
        ok, checks = run_verification(args.depths, args.seed)
        with open(f"{args.out_prefix}_verification.json", "w", encoding="utf-8") as fh:
            json.dump(checks, fh, indent=2)
        if not ok:
            print("# VERIFICATION FAILED - measurement skipped", flush=True)
            return 1
        print("# verification passed for every layer at every depth", flush=True)
        if args.mode == "verify":
            return 0

    rows = {}

    if args.mode in ("memory", "all"):
        for variant in args.variants:
            for K in args.depths:
                result = measure_memory(variant, K, args.seed)
                if args.validate_probe:
                    peak, ops = validate_probe_free_path(variant, K, args.seed)
                    result["probe_free_peak_total_bytes"] = peak
                    result["probe_free_aten_ops"] = ops
                    result["probe_matches_reference_path"] = (
                        peak == result["peak_total_bytes"]
                    )
                rows[(variant, K)] = dict(result)
                print("mem " + json.dumps(result), flush=True)

    if args.mode in ("time", "all"):
        for variant in args.variants:
            for K in args.depths:
                result = measure_time(variant, K, args.seed, args.warmup,
                                      args.batches, args.repeats)
                rows.setdefault((variant, K), {"variant": variant, "K": K})
                rows[(variant, K)].update(result)
                print("time " + json.dumps(result), flush=True)

    ordered = [rows[key] for key in sorted(rows, key=lambda t: (t[0], t[1]))]
    if ordered:
        main_csv = f"{args.out_prefix}.csv"
        scaling_csv = f"{args.out_prefix}_scaling.csv"
        write_rows(main_csv, ordered)
        write_scaling(scaling_csv, ordered)
        print(f"# wrote {main_csv} and {scaling_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
