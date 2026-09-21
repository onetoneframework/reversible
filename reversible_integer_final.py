"""
Final research prototype: exact reversible integer training with a tensor-shared
block exponent for the hidden adjoint.

What this file is meant to establish
------------------------------------
1) Integer-only reversible forward/inverse (bit-exact reconstruction).
2) Cache-free backward: forward activations are reconstructed, not stored.
3) Tensor-shared exponent for the backward adjoint (no per-row exponent needed
   on the current toy problem).
4) A cached control using the *same* arithmetic, for bit-equivalence testing.
5) Optional legacy and float baselines.

Important research status
-------------------------
- This is a research prototype, not a claim of novel BFP arithmetic.
- The strongest current result is on sklearn digits.
- Parameter updates use a conservative integer error-feedback optimizer that
  re-normalizes each parameter-gradient tensor before applying the update.

Example
-------
  python reversible_integer_final.py --method equiv --K 16 --seed 0
  python reversible_integer_final.py --method reversible --K 32 --epochs 4 --seed 0
  python reversible_integer_final.py --method cached --K 32 --epochs 4 --seed 0
  python reversible_integer_final.py --method legacy --K 16 --epochs 6 --seed 0
  python reversible_integer_final.py --method float --K 32 --epochs 4 --seed 0

Optional switches (both default ON; verified bit-identical to OFF):
  --no-fused              two-call inverse+parts recomputation (original form)
  --step-mode batched     collect all gradients then one optimizer step (original form)
  --method implequiv      verify fused == unfused
  --method stepequiv      verify per-layer step == batched step

Width and grouping options (all default to the original int64 dense behaviour;
a 34-entry behavioural fingerprint is byte-identical with them at their
defaults, and equiv/implequiv/stepequiv pass with them switched on):

  --state-bits N     two's complement wrap width of the reversible state.
                     Wraparound is exactly invertible, so a narrow state costs
                     accuracy but never breaks reversibility.
  --weight-bits N    saturating width of U and V; also sets the init range to
                     +/-2^(N-4), matching the NumPy reference.
  --accum-bits N     saturating width of the GEMM accumulators.
  --groups g         store each weight as g dense blocks of (H/g x M/g) wired
                     by residue class, so parameters and MACs are exactly 1/g.
  --phase-step s     layer k uses residue phase (k*s) mod g.
  --hidden-width H   coupling hidden width (distinct from --hidden-bits, which
                     is the gradient mantissa width).
  --perm transpose   table-free (M/g x g) grid transpose between layers, which
                     spreads every residue group across all g groups of the
                     next layer. Grouped coupling needs this; with the default
                     permutation, g=8 collapses to roughly chance.

The int8-style configuration measured against the NumPy reference is:
  --state-bits 12 --weight-bits 8 --accum-bits 32 --hidden-bits 8
"""

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Experiment constants
# -----------------------------------------------------------------------------
TB = 5
BATCH = 32
WID = 128
M = 64
H = 64
NCLASS = 10

# Fixed digits split used throughout the experiments.
_X, _y = load_digits(return_X_y=True)
_Xtr, _Xte, _ytr, _yte = train_test_split(
    _X, _y, test_size=0.25, random_state=0, stratify=_y
)
XTR = torch.tensor(_Xtr, dtype=torch.int64)
XTE = torch.tensor(_Xte, dtype=torch.int64)
YTR = torch.tensor(_ytr, dtype=torch.long)
YTE = torch.tensor(_yte, dtype=torch.long)

HALVES = torch.roll(torch.arange(WID, dtype=torch.long), M)
ODDEVEN = torch.arange(WID, dtype=torch.long) ^ 1
INV_HALVES = torch.argsort(HALVES)
INV_ODDEVEN = torch.argsort(ODDEVEN)



# -----------------------------------------------------------------------------
# Exact integer matmul
# -----------------------------------------------------------------------------
# PyTorch has no BLAS path for integer matmul on this CPU: int64 runs at about
# 5.5 GOP/s where float64 runs one to two orders of magnitude faster. Integers
# up to 2**53 are represented exactly by float64, so when
#     max|a| * max|b| * K < 2**53
# the float64 GEMM returns exactly the int64 result. The bound is checked per
# call and the int64 kernel is used whenever it is not met, so this is a pure
# implementation optimisation: the arithmetic being defined is unchanged.
USE_FLOAT64_GEMM = True
GEMM_STATS = {"float64": 0, "int64": 0}


def imatmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Integer matrix product, exact by construction."""
    if not USE_FLOAT64_GEMM:
        GEMM_STATS["int64"] += 1
        return a @ b
    k = a.shape[-1]
    ma = int(torch.abs(a).max().item()) if a.numel() else 0
    mb = int(torch.abs(b).max().item()) if b.numel() else 0
    if ma == 0 or mb == 0:
        GEMM_STATS["float64"] += 1
        return torch.zeros(a.shape[:-1] + b.shape[-1:], dtype=torch.int64)
    if ma * mb * k < (1 << 53):
        GEMM_STATS["float64"] += 1
        return torch.round(a.double() @ b.double()).to(torch.int64)
    GEMM_STATS["int64"] += 1
    return a @ b

# -----------------------------------------------------------------------------
# Optional width / grouping configuration
# -----------------------------------------------------------------------------
@dataclass
class RunConfig:
    """
    Every field defaults to the original int64 dense behaviour.

    state_bits / weight_bits / accum_bits at None mean "no width limit", which
    is what the original code did implicitly by living in int64. Setting them
    simulates narrow hardware while still storing values in int64.

    groups > 1 replaces each dense weight with g dense blocks of (H/g, M/g),
    connected by residue class, so both parameter count and MAC count are
    exactly 1/g of the dense layer.
    """
    state_bits: int = None       # two's complement wrap width of the state
    weight_bits: int = None      # saturating width of U, V
    accum_bits: int = None       # saturating width of GEMM accumulators
    groups: int = 1              # g: number of residue-connected dense blocks
    phase_step: int = 0          # layer k uses residue phase (k * phase_step) % g
    hidden: int = H              # coupling hidden width
    perm_mode: str = "default"   # "default" | "transpose"


CFG = RunConfig()

STATS = {"state_wrap": 0, "state_elems": 0, "accum_saturations": 0, "mac": 0}


def reset_stats():
    for key in STATS:
        STATS[key] = 0


def set_config(**kwargs):
    """
    Install a new active configuration; unknown or invalid keys raise.

    The configuration is process-global on purpose: perm_for, wrap_state,
    saturate, BlockSGD and the float baseline all read it, so there is exactly
    one source of truth. Validation happens before anything is mutated, so a
    rejected configuration leaves the previous one intact.
    """
    global CFG
    candidate = validate_config(RunConfig(**kwargs))
    CFG = candidate
    _PERM_CACHE.clear()
    return CFG


def validate_config(cfg: "RunConfig"):
    g = cfg.groups
    if g < 1:
        raise ValueError("groups must be >= 1")
    if M % g or cfg.hidden % g:
        raise ValueError(f"groups={g} must divide M={M} and hidden={cfg.hidden}")
    if cfg.perm_mode not in ("default", "transpose"):
        raise ValueError(f"unknown perm_mode {cfg.perm_mode}")
    if cfg.perm_mode == "transpose" and (M // g) % g:
        raise ValueError(
            f"transpose permutation needs (M/g) %% g == 0; M={M} g={g}"
        )
    for name in ("state_bits", "weight_bits", "accum_bits"):
        b = getattr(cfg, name)
        if b is not None and not (2 <= b <= 63):
            raise ValueError(f"{name} must be in [2, 63] or None")
    return cfg


def wrap_state(x: torch.Tensor, bits: int = None) -> torch.Tensor:
    """
    Two's complement wraparound to `bits`, counted.

    Wraparound is exactly invertible, so narrowing the state costs accuracy but
    never breaks reversibility: (b + f) then (- f) returns b for any width.
    """
    bits = CFG.state_bits if bits is None else bits
    if bits is None or bits >= 63:
        return x
    n = 1 << bits
    y = ((x + (n >> 1)) & (n - 1)) - (n >> 1)
    STATS["state_wrap"] += int(torch.count_nonzero(y != x).item())
    STATS["state_elems"] += int(x.numel())
    return y


def saturate(x: torch.Tensor, bits: int = None) -> torch.Tensor:
    """Saturating clip for accumulators and weights, counted."""
    bits = CFG.accum_bits if bits is None else bits
    if bits is None or bits >= 63:
        return x
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    STATS["accum_saturations"] += int(
        torch.count_nonzero((x < lo) | (x > hi)).item()
    )
    return torch.clamp(x, lo, hi)


_PERM_CACHE = {}


def _transpose_perm(g: int) -> torch.Tensor:
    """
    Table-free dispersing permutation: an (M/g x g) grid transpose inside each
    half. Every residue group is spread across all g residue groups of the next
    layer, which is the condition grouped coupling needs to keep mixing.
    """
    key = ("transpose", g)
    if key not in _PERM_CACHE:
        i = torch.arange(M, dtype=torch.long)
        p = (i // g) + (i % g) * (M // g)
        _PERM_CACHE[key] = torch.cat([p, p + M])
    return _PERM_CACHE[key]


def _inv(perm: torch.Tensor) -> torch.Tensor:
    key = ("inv", id(perm))
    if key not in _PERM_CACHE:
        _PERM_CACHE[key] = torch.argsort(perm)
    return _PERM_CACHE[key]


def perm_for(k: int) -> torch.Tensor:
    if CFG.perm_mode == "transpose":
        return HALVES if k % 2 == 0 else _transpose_perm(CFG.groups)
    return HALVES if k % 2 == 0 else ODDEVEN


def invperm_for(k: int) -> torch.Tensor:
    if CFG.perm_mode == "transpose":
        return INV_HALVES if k % 2 == 0 else _inv(_transpose_perm(CFG.groups))
    return INV_HALVES if k % 2 == 0 else INV_ODDEVEN


def group_start(c: int, g: int, phase: int) -> int:
    """First column index of residue group c, which then strides by g."""
    return (c - phase) % g


# -----------------------------------------------------------------------------
# Problem shape: coupling half-width, dataset, batch size
# -----------------------------------------------------------------------------
# M defaults to 64 and the dataset to sklearn digits, which is the configuration
# every earlier result was produced with. set_dims / set_dataset exist so a
# wider problem can reuse this module instead of forking it; neither is called
# unless a driver calls it, so the default path is untouched.

def set_dims(m: int, hidden: int = None):
    """
    Change the coupling half-width M and rebuild everything derived from it:
    the state width, both permutation tables and their inverses.

    The transpose permutation additionally needs (M/g) %% g == 0, which for
    g in {4, 8} reduces to M being a multiple of 64. validate_config enforces
    it for whichever g is active.
    """
    global M, WID, H, HALVES, ODDEVEN, INV_HALVES, INV_ODDEVEN
    if m < 2 or m % 2:
        raise ValueError("M must be a positive even number")
    M = m
    WID = 2 * m
    if hidden is not None:
        H = hidden
    HALVES = torch.roll(torch.arange(WID, dtype=torch.long), M)
    ODDEVEN = torch.arange(WID, dtype=torch.long) ^ 1
    INV_HALVES = torch.argsort(HALVES)
    INV_ODDEVEN = torch.argsort(ODDEVEN)
    _PERM_CACHE.clear()
    return {"M": M, "WID": WID, "H": H}


def set_dataset(xtr, ytr, xte, yte, batch: int = None):
    """
    Install a different dataset. Inputs must already be integer tensors of
    width exactly M (pad before calling, so the network stays fully
    reversible and no non-invertible projection layer is needed).
    """
    global XTR, YTR, XTE, YTE, BATCH
    for name, t in (("xtr", xtr), ("xte", xte)):
        if t.shape[1] != M:
            raise ValueError(f"{name} has width {t.shape[1]}, expected M={M}")
    XTR = xtr.to(torch.int64)
    XTE = xte.to(torch.int64)
    YTR = ytr.to(torch.long)
    YTE = yte.to(torch.long)
    if batch is not None:
        BATCH = int(batch)
    return {"train": tuple(XTR.shape), "test": tuple(XTE.shape), "batch": BATCH}


# -----------------------------------------------------------------------------
# Integer arithmetic helpers
# -----------------------------------------------------------------------------
def rshift0(x: torch.Tensor, sh) -> torch.Tensor:
    """Signed right shift with truncation toward zero."""
    sh = torch.as_tensor(sh, dtype=torch.int64, device=x.device)
    sh = torch.clamp(sh, min=0, max=63)
    return torch.sign(x) * (torch.abs(x) >> sh)


def bitlen_arr(x: torch.Tensor) -> torch.Tensor:
    """Elementwise integer bit length for int64 tensors."""
    z = torch.abs(x).to(torch.int64)
    b = torch.zeros_like(z)
    for k in (32, 16, 8, 4, 2, 1):
        q = (z >> k) > 0
        b += q.to(torch.int64) * k
        z = torch.where(q, z >> k, z)
    return b + (z > 0).to(torch.int64)


def block_bitlen(x: torch.Tensor) -> int:
    z = int(torch.abs(x).max().item()) if x.numel() else 0
    return z.bit_length()


def pow2norm(x: torch.Tensor, target_bits: int):
    """
    Per-sample power-of-two downscaling used in the integer forward path.
    Returns integer output and a per-row shift s.
    """
    mx = torch.abs(x).amax(dim=1, keepdim=True)
    s = torch.clamp(bitlen_arr(mx) - target_bits, min=0)
    return x >> s, s


def norm_block(m: torch.Tensor, e: int, bits: int):
    """
    Normalize one integer block to approximately `bits` of mantissa precision.
    The represented value is m * 2**e.
    """
    b = block_bitlen(m)
    if b == 0:
        return m, int(e)
    d = b - bits
    if d > 0:
        m = rshift0(m, d)
        e = int(e) + d
    elif d < 0:
        k = min(-d, max(bits - 1, 0))
        m = m << k
        e = int(e) - k
    return m, int(e)


def renorm_tensor(g: torch.Tensor, bits: int) -> torch.Tensor:
    """Mantissa-only block normalization used by the legacy/optimizer path."""
    b = block_bitlen(g)
    if b == 0:
        return g
    d = b - bits
    if d < 0:
        return g << (-d)
    if d > 0:
        return rshift0(g, d)
    return g


# -----------------------------------------------------------------------------
# Exact reversible integer coupling layer
# -----------------------------------------------------------------------------
class Coupling:
    """
    Additive coupling with optional residue grouping.

    groups == 1 stores one dense U (hidden, M) and V (M, hidden) and executes
    exactly the original expressions. groups == g stores g dense blocks of
    (hidden/g, M/g) and (M/g, hidden/g); residue class c of the input columns
    drives residue class c of the output columns, so parameters and MACs are
    both exactly 1/g of the dense layer.
    """

    def __init__(self, generator: torch.Generator, cfg: RunConfig = None, phase: int = 0):
        cfg = cfg if cfg is not None else CFG
        self.cfg = cfg
        self.g = cfg.groups
        self.phase = phase % cfg.groups
        hidden = cfg.hidden
        lim = 8 if cfg.weight_bits is None else (1 << (cfg.weight_bits - 4))
        if self.g == 1:
            # Original draw order and shapes, so the RNG stream is unchanged.
            self.Ug = [torch.randint(-lim, lim + 1, (hidden, M),
                                     dtype=torch.int64, generator=generator)]
            self.Vg = [torch.randint(-lim, lim + 1, (M, hidden),
                                     dtype=torch.int64, generator=generator)]
        else:
            g = self.g
            self.Ug = [torch.randint(-lim, lim + 1, (hidden // g, M // g),
                                     dtype=torch.int64, generator=generator)
                       for _ in range(g)]
            self.Vg = [torch.randint(-lim, lim + 1, (M // g, hidden // g),
                                     dtype=torch.int64, generator=generator)
                       for _ in range(g)]

    # Dense accessors; they exist only for the ungrouped configuration so that
    # a missed grouped call site fails loudly instead of silently going dense.
    @property
    def U(self):
        if self.g != 1:
            raise AttributeError("grouped coupling has no dense U; use .Ug")
        return self.Ug[0]

    @property
    def V(self):
        if self.g != 1:
            raise AttributeError("grouped coupling has no dense V; use .Vg")
        return self.Vg[0]

    def blocks(self):
        """Parameter order used by the optimizer: all U blocks, then all V."""
        return self.Ug + self.Vg

    def _gmm(self, x: torch.Tensor, blocks, n_out: int) -> torch.Tensor:
        """Grouped x @ block.T, writing each residue group's output columns."""
        if self.g == 1:
            return imatmul(x, blocks[0].T)
        g = self.g
        out = x.new_zeros((x.shape[0], n_out))
        for c in range(g):
            start = group_start(c, g, self.phase)
            out[:, start::g] = imatmul(x[:, start::g], blocks[c].T)
        return out

    def parts(self, a: torch.Tensor):
        cfg = self.cfg
        z, s1 = pow2norm(
            saturate(self._gmm(a, self.Ug, cfg.hidden), cfg.accum_bits), TB + 2
        )
        h = torch.clamp_min(z, 0)
        f, s2 = pow2norm(
            saturate(self._gmm(h, self.Vg, M), cfg.accum_bits), TB + 2
        )
        return f, z, h, s1, s2

    def forward(self, v: torch.Tensor):
        a, b = v[:, :M], v[:, M:]
        f, *_ = self.parts(a)
        return torch.cat([a, wrap_state(b + f, self.cfg.state_bits)], dim=1)

    def inverse(self, v: torch.Tensor):
        a, bout = v[:, :M], v[:, M:]
        f, *_ = self.parts(a)
        return torch.cat([a, wrap_state(bout - f, self.cfg.state_bits)], dim=1)


def make_layers(K: int, seed: int, cfg: RunConfig = None):
    cfg = cfg if cfg is not None else CFG
    validate_config(cfg)
    gen = torch.Generator().manual_seed(seed)
    return [Coupling(gen, cfg, phase=(k * cfg.phase_step) % cfg.groups)
            for k in range(K)]


def make_input(x: torch.Tensor, xscale: int = 1):
    return torch.cat([wrap_state(x * xscale), torch.zeros_like(x)], dim=1)


def forward_reversible(x: torch.Tensor, layers, xscale: int = 1):
    """Cache-free forward; returns only the final state."""
    v = make_input(x, xscale)
    for k, layer in enumerate(layers):
        v = layer.forward(v)
        if k < len(layers) - 1:
            v = v[:, perm_for(k)]
    return v


def forward_cached(x: torch.Tensor, layers, xscale: int = 1):
    """
    Forward control that stores each layer input. This is deliberately *not*
    the final method; it exists to verify that reconstruction changes nothing.
    """
    v = make_input(x, xscale)
    cache = []
    for k, layer in enumerate(layers):
        cache.append(v.clone())
        v = layer.forward(v)
        if k < len(layers) - 1:
            v = v[:, perm_for(k)]
    return v, cache


def inverse_network(v: torch.Tensor, layers):
    for k in range(len(layers) - 1, -1, -1):
        if k < len(layers) - 1:
            v = v[:, invperm_for(k)]
        v = layers[k].inverse(v)
    return v


# -----------------------------------------------------------------------------
# Tensor-shared exponent gradient representation
# -----------------------------------------------------------------------------
@dataclass
class SharedGrad:
    m: torch.Tensor
    e: int

    def permute_(self, p: torch.Tensor):
        self.m = self.m[:, p]
        return self


def align_shared(m1, e1: int, m2, e2: int, bits: int):
    E = max(int(e1), int(e2))
    d1, d2 = E - int(e1), E - int(e2)
    a = torch.zeros_like(m1) if d1 >= 63 else rshift0(m1, d1)
    b = torch.zeros_like(m2) if d2 >= 63 else rshift0(m2, d2)
    return norm_block(a + b, E, bits)


def concat_shared(m1, e1: int, m2, e2: int, bits: int):
    E = max(int(e1), int(e2))
    d1, d2 = E - int(e1), E - int(e2)
    a = torch.zeros_like(m1) if d1 >= 63 else rshift0(m1, d1)
    b = torch.zeros_like(m2) if d2 >= 63 else rshift0(m2, d2)
    return norm_block(torch.cat([a, b], dim=1), E, bits)


def dense_linear_backward(gy: SharedGrad, x: torch.Tensor, W: torch.Tensor,
                          s: torch.Tensor, hidden_bits: int,
                          param_bits: int = 30, accum_bits: int = None):
    """
    Backward through y = (x @ W.T) >> s with one shared exponent on gy and a
    per-row forward shift s. Returns (input gradient, (weight mantissa, exp)).

    Shared by the coupling stack and by the convolutional and attention blocks
    in reversible_arch, all of which reduce to this same shape.
    """
    s = s.reshape(-1).to(torch.int64)
    sample_e = gy.e - s
    E = int(sample_e.max().item()) if sample_e.numel() else int(gy.e)
    shift = (E - sample_e).reshape(-1, 1)

    gx = saturate(rshift0(imatmul(gy.m, W), shift), accum_bits)
    gxm, gxe = norm_block(gx, E, hidden_bits)
    gwm = saturate(imatmul(rshift0(gy.m, shift).T, x), accum_bits)
    gwm, gwe = norm_block(gwm, E, param_bits)
    return SharedGrad(gxm, gxe), (gwm, gwe)


def linear_backward_shared(
    gy: SharedGrad,
    x: torch.Tensor,
    layer: "Coupling",
    which: str,
    s: torch.Tensor,
    hidden_bits: int,
    param_bits: int = 30,
):
    """
    Backward through y ~= (x W^T) / 2**s using one shared exponent.

    gy represents gy.m * 2**gy.e.
    Each sample's local forward shift contributes an additional 2**(-s_i).
    Samples are aligned to one shared exponent before integer accumulation.

    Returns the input gradient and a list of per-block weight gradients, one
    entry per residue group (a single entry when groups == 1).
    """
    cfg = layer.cfg
    g = layer.g
    blocks = layer.Ug if which == "U" else layer.Vg
    n_in = M if which == "U" else cfg.hidden

    s = s.reshape(-1).to(torch.int64)
    sample_e = gy.e - s
    E = int(sample_e.max().item()) if sample_e.numel() else int(gy.e)
    shift = (E - sample_e).reshape(-1, 1)

    # Samples aligned to the shared exponent, used for the parameter gradient.
    gm_aligned = rshift0(gy.m, shift)

    if g == 1:
        gx, gw = dense_linear_backward(gy, x, blocks[0], s, hidden_bits,
                                       param_bits, cfg.accum_bits)
        return gx, [gw]

    raw = gy.m.new_zeros((gy.m.shape[0], n_in))
    weight_grads = []
    for c in range(g):
        start = group_start(c, g, layer.phase)
        raw[:, start::g] = imatmul(gy.m[:, start::g], blocks[c])
        gwm = saturate(imatmul(gm_aligned[:, start::g].T, x[:, start::g]), cfg.accum_bits)
        weight_grads.append(norm_block(gwm, E, param_bits))
    gx = saturate(rshift0(raw, shift), cfg.accum_bits)
    gxm, gxe = norm_block(gx, E, hidden_bits)
    return SharedGrad(gxm, gxe), weight_grads


# -----------------------------------------------------------------------------
# Loss gradient and backward passes
# -----------------------------------------------------------------------------
def multiclass_hinge_output_grad(v: torch.Tensor, y: torch.Tensor, hidden_bits: int):
    out = v[:, M : M + NCLASS]
    n = len(y)
    masked = out.clone()
    masked[torch.arange(n), y] = torch.iinfo(torch.int64).min // 4
    j = masked.argmax(dim=1)
    active = out[torch.arange(n), j] + 1 > out[torch.arange(n), y]

    gm = torch.zeros_like(v)
    ii = torch.arange(n)[active]
    gm[ii, M + j[active]] += 1
    gm[ii, M + y[active]] -= 1
    gm, ge = norm_block(gm, 0, hidden_bits)
    return SharedGrad(gm, ge)


def backward_reconstruct(
    final_v: torch.Tensor,
    y: torch.Tensor,
    layers,
    hidden_bits: int = 9,
    fused: bool = True,
    on_layer_grad=None,
):
    """
    Cache-free backward. Reconstructs each forward input by exact inversion.

    fused:
        True  -> call layer.parts(a) once per layer and derive the inverse from
                 the same f. Bit-identical to the unfused path (verified by
                 --method implequiv), ~1.4x faster backward.
        False -> original two-call form (layer.inverse, then layer.parts).
                 Kept for auditability.
    on_layer_grad:
        Optional callback (k, gU, gV) invoked as soon as layer k's parameter
        gradients exist. When given, gradients are NOT accumulated into a list
        (per-layer optimizer step; removes the O(K) parameter-gradient buffer).
        When None, the full list is returned as before.
    """
    v = final_v
    g = multiclass_hinge_output_grad(v, y, hidden_bits)
    grads = [] if on_layer_grad is None else None

    for k in range(len(layers) - 1, -1, -1):
        if k < len(layers) - 1:
            inv = invperm_for(k)
            v = v[:, inv]
            g.permute_(inv)

        layer = layers[k]
        if fused:
            # One recomputation serves both the inverse and the local gradient.
            a = v[:, :M]
            f, z, h, s1, s2 = layer.parts(a)
            vin = torch.cat(
                [a, wrap_state(v[:, M:] - f, layer.cfg.state_bits)], dim=1
            )
        else:
            # Exact layer inversion recovers the original layer input.
            vin = layer.inverse(v)
            a = vin[:, :M]
            f, z, h, s1, s2 = layer.parts(a)

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
            # Layer k's update only touches U_k, V_k. Layers < k have already
            # been reconstructed with the *pre-update* weights they need, so
            # stepping here is bit-identical to a batched step at the end.
            on_layer_grad(k, gUs, gVs)
        v = vin

    if on_layer_grad is not None:
        return None, v, g
    grads.sort(key=lambda t: t[0])
    return [q for _, gus, gvs in grads for q in (*gus, *gvs)], v, g


def backward_cached(
    final_v: torch.Tensor,
    cache,
    y: torch.Tensor,
    layers,
    hidden_bits: int = 9,
    on_layer_grad=None,
):
    """Cached correctness control using exactly the same gradient arithmetic."""
    g = multiclass_hinge_output_grad(final_v, y, hidden_bits)
    grads = [] if on_layer_grad is None else None

    for k in range(len(layers) - 1, -1, -1):
        if k < len(layers) - 1:
            g.permute_(invperm_for(k))

        layer = layers[k]
        vin = cache[k]
        a = vin[:, :M]
        f, z, h, s1, s2 = layer.parts(a)

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
# Integer error-feedback optimizer
# -----------------------------------------------------------------------------
class BlockSGD:
    """
    Conservative integer error-feedback update.

    Parameter-gradient exponents are intentionally not used as absolute learning
    rates here; each parameter-gradient mantissa is block-normalized before being
    accumulated. This was the most stable controlled optimizer in the toy tests.

    Stability condition, which is part of the algorithm and not a tuning choice:

        2**shift > max|renorm_tensor(g, param_bits)| = 2**param_bits - 1

    i.e. with shift = (param_bits - 1) + lr_bits, **lr_bits >= 1**.

    The accumulator gains up to 2**param_bits - 1 per step and drains exactly
    2**shift when it fires. If the drain is smaller than the largest possible
    gain, the clip to +-1 actually engages, the residual grows in one direction
    without bound, and the affected weights ramp until the width clamp stops
    them. Measured on a 32-way source->target lookup at param_bits=12: lr_bits=0
    (drain 2048) left |acc| wandering at 2366-5260 and accuracy oscillating
    between 6% and 25% over 4000 epochs, while lr_bits 1/2/3 held |acc| at
    exactly 2**shift - 1 and reached 32/32 by epoch 400. `self.stable` records
    which regime an instance is in.
    """

    def __init__(self, params, param_bits: int = 12, lr_bits: int = 2, clip: int = 1,
                 weight_bits: int = None, momentum_bits: int = 0):
        self.params = params
        self.acc = [torch.zeros_like(w) for w in params]
        # Integer momentum: m <- m - (m >> k) + g, an exponential average with
        # decay 1 - 2**-k, fed to the accumulator in place of the raw block.
        # The renormalised gradient of one batch is a +-2**11 vote from that
        # batch alone; with thousands of distinct sentences those votes cancel
        # step to step and the +-1 weight updates random-walk. Averaging over
        # ~2**k steps before quantising lets consistent directions accumulate.
        # Shift-only, per parameter, so per-layer == batched stepping still
        # holds. 0 disables and is bit-identical to the previous behaviour.
        self.momentum_bits = int(momentum_bits)
        self.mom = ([torch.zeros_like(w) for w in params] if self.momentum_bits else None)
        self.shift = (param_bits - 1) + lr_bits
        self.clip = clip
        self.param_bits = param_bits
        # None means "no width limit", which is the original behaviour.
        self.weight_bits = CFG.weight_bits if weight_bits is None else weight_bits
        # See the class docstring: below this the error-feedback residual grows
        # without bound instead of sawtoothing under 2**shift.
        self.stable = (1 << self.shift) > (1 << param_bits) - 1

    def _step_one(self, i: int, gm: torch.Tensor):
        w, acc = self.params[i], self.acc[i]
        # Controlled choice: use normalized mantissa, discard absolute e.
        gn = renorm_tensor(gm, self.param_bits)
        if self.momentum_bits:
            m = self.mom[i]
            m -= rshift0(m, self.momentum_bits)
            m += gn
            gn = renorm_tensor(m, self.param_bits)
        acc -= gn
        q = torch.sign(acc) * (torch.abs(acc) >> self.shift)
        q = torch.clamp(q, -self.clip, self.clip)
        w += q
        if self.weight_bits is not None:
            lo = -(1 << (self.weight_bits - 1))
            hi = (1 << (self.weight_bits - 1)) - 1
            torch.clamp(w, lo, hi, out=w)
        acc -= q << self.shift
        return int(torch.abs(q).max().item())

    def step(self, grad_pairs):
        """Batched step over all parameters (original behaviour)."""
        max_update = 0
        for i, (gm, ge) in enumerate(grad_pairs):
            max_update = max(max_update, self._step_one(i, gm))
        return max_update

    def step_layer(self, k: int, gUs, gVs):
        """
        Per-layer step: update layer k's blocks immediately. Each parameter's
        update depends only on its own accumulator and gradient, so this is
        bit-identical to step() applied to the full list.

        gUs / gVs are lists of (mantissa, exponent) pairs, one per residue
        group, matching the parameter order [U_0..U_{g-1}, V_0..V_{g-1}].
        """
        g = len(gUs)
        base = k * 2 * g
        largest = 0
        for c, (gm, _ge) in enumerate(gUs):
            largest = max(largest, self._step_one(base + c, gm))
        for c, (gm, _ge) in enumerate(gVs):
            largest = max(largest, self._step_one(base + g + c, gm))
        return largest


# -----------------------------------------------------------------------------
# Legacy baseline: no scale metadata in hidden gradient
# -----------------------------------------------------------------------------
def shifted_linear_backward_legacy(g, x, W, s):
    gx = (g @ W) >> s
    terms = (g[:, :, None] * x[:, None, :]) >> s[:, None, :]
    return gx, terms.sum(dim=0, dtype=torch.int64)


def backward_legacy(final_v, y, layers, grad_bits=9):
    if layers and layers[0].g != 1:
        raise ValueError(
            "legacy backward is a dense-only failure ablation; groups must be 1"
        )
    v = final_v
    out = v[:, M : M + NCLASS]
    n = len(y)
    masked = out.clone()
    masked[torch.arange(n), y] = torch.iinfo(torch.int64).min // 4
    j = masked.argmax(1)
    active = out[torch.arange(n), j] + 1 > out[torch.arange(n), y]
    g = torch.zeros_like(v)
    ii = torch.arange(n)[active]
    g[ii, M + j[active]] += 1
    g[ii, M + y[active]] -= 1
    g = renorm_tensor(g, grad_bits)

    grads = []
    for k in range(len(layers) - 1, -1, -1):
        if k < len(layers) - 1:
            inv = invperm_for(k)
            v = v[:, inv]
            g = renorm_tensor(g[:, inv], grad_bits)
        layer = layers[k]
        vin = layer.inverse(v)
        a = vin[:, :M]
        f, z, h, s1, s2 = layer.parts(a)
        ga, gb = g[:, :M], g[:, M:]
        gh, gV = shifted_linear_backward_legacy(gb, h, layer.V, s2)
        gz = gh * (z > 0)
        da, gU = shifted_linear_backward_legacy(gz, a, layer.U, s1)
        g = renorm_tensor(torch.cat([ga + da, gb], dim=1), grad_bits)
        grads.append((k, (gU, 0), (gV, 0)))
        v = vin

    grads.sort(key=lambda t: t[0])
    return [q for _, gu, gv in grads for q in (gu, gv)]


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------
def parameters(layers):
    """Flat parameter list in optimizer order: per layer, U blocks then V."""
    return [w for layer in layers for w in layer.blocks()]


def accuracy_int(layers, xscale=1):
    vt = forward_reversible(XTE, layers, xscale)
    pred = vt[:, M : M + NCLASS].argmax(dim=1)
    return float((pred == YTE).float().mean().item()), vt


def exact_inverse_check(layers, xscale=1, n=32):
    x = XTE[:n]
    final = forward_reversible(x, layers, xscale)
    rec = inverse_network(final, layers)
    return bool(torch.equal(rec, make_input(x, xscale)))


def cache_bytes(cache):
    return int(sum(t.numel() * t.element_size() for t in cache))


def train_integer(
    method: str,
    K: int,
    epochs: int,
    seed: int,
    lr_bits: int = 2,
    hidden_bits: int = 9,
    xscale: int = 1,
    fused: bool = True,
    step_mode: str = "per-layer",
):
    """
    step_mode:
        "per-layer" -> optimizer steps layer k right after its gradient is
                       computed; only one layer's (gU, gV) is ever alive.
        "batched"   -> collect all K layers' gradients, then one opt.step().
                       Keeps an O(K) parameter-gradient buffer (original form).
    Both modes are bit-identical in result (see --method stepequiv).
    Legacy always uses batched (it is a failure ablation, not a product path).
    """
    if method not in {"reversible", "cached", "legacy"}:
        raise ValueError(method)
    if step_mode not in {"per-layer", "batched"}:
        raise ValueError(step_mode)

    reset_stats()
    layers = make_layers(K, seed)
    params = parameters(layers)
    opt = BlockSGD(params, param_bits=12, lr_bits=lr_bits, clip=1)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    first_cache_bytes = None
    peak_grad_buffer_bytes = 0

    per_layer = step_mode == "per-layer" and method != "legacy"
    cb = (lambda k, gu, gv: opt.step_layer(k, gu, gv)) if per_layer else None

    for _ep in range(epochs):
        idx = torch.tensor(rng.permutation(len(XTR)), dtype=torch.long)
        for st in range(0, len(XTR), BATCH):
            bi = idx[st : st + BATCH]
            xb, yb = XTR[bi], YTR[bi]

            if method == "cached":
                final, cache = forward_cached(xb, layers, xscale)
                grad_pairs, _ = backward_cached(final, cache, yb, layers, hidden_bits, on_layer_grad=cb)
                if first_cache_bytes is None:
                    first_cache_bytes = cache_bytes(cache)
            else:
                final = forward_reversible(xb, layers, xscale)
                if method == "reversible":
                    grad_pairs, _, _ = backward_reconstruct(
                        final, yb, layers, hidden_bits, fused=fused, on_layer_grad=cb
                    )
                else:
                    grad_pairs = backward_legacy(final, yb, layers, hidden_bits)

            if grad_pairs is not None:
                peak_grad_buffer_bytes = max(
                    peak_grad_buffer_bytes,
                    sum(gm.numel() * gm.element_size() for gm, _ in grad_pairs),
                )
                opt.step(grad_pairs)
            else:
                # per-layer: at most one layer's blocks are alive at a time.
                peak_grad_buffer_bytes = max(
                    peak_grad_buffer_bytes,
                    sum(w.numel() for w in layers[0].blocks()) * 8,
                )

    acc, vt = accuracy_int(layers, xscale)
    return {
        "method": method,
        "K": K,
        "epochs": epochs,
        "seed": seed,
        "hidden_mantissa_bits": hidden_bits,
        "lr_bits": lr_bits,
        "fused_recompute": bool(fused) if method == "reversible" else None,
        "step_mode": "batched" if method == "legacy" else step_mode,
        "accuracy": acc,
        "inverse_exact": exact_inverse_check(layers, xscale),
        "max_activation": int(torch.abs(vt).max().item()),
        "max_weight": max(int(torch.abs(w).max().item()) for w in params),
        "stored_activation_cache_bytes_first_batch": int(first_cache_bytes or 0),
        "peak_parameter_gradient_buffer_bytes": int(peak_grad_buffer_bytes),
        # Configuration and width telemetry. MAC per sample per forward equals
        # the parameter count, since every weight is used exactly once.
        "groups": CFG.groups,
        "hidden": CFG.hidden,
        "perm_mode": CFG.perm_mode,
        "phase_step": CFG.phase_step,
        "state_bits": CFG.state_bits,
        "weight_bits": CFG.weight_bits,
        "accum_bits": CFG.accum_bits,
        "parameter_count": int(sum(w.numel() for w in params)),
        "state_wrap_frac": (
            round(STATS["state_wrap"] / STATS["state_elems"], 6)
            if STATS["state_elems"] else 0.0
        ),
        "accum_saturations": int(STATS["accum_saturations"]),
        "seconds": time.time() - t0,
    }


def implementation_equivalence_test(K: int, seed: int = 0, hidden_bits: int = 9, xscale: int = 1):
    """
    One-batch test: fused vs unfused reconstruction must give identical
    reconstructed inputs and identical parameter/input gradients.
    """
    layers = make_layers(K, seed)
    xb, yb = XTR[:BATCH], YTR[:BATCH]
    final = forward_reversible(xb, layers, xscale)

    g_f, in_f, fin_f = backward_reconstruct(final.clone(), yb, layers, hidden_bits, fused=True)
    g_u, in_u, fin_u = backward_reconstruct(final.clone(), yb, layers, hidden_bits, fused=False)

    grad_equal = all(torch.equal(a, b) and int(ea) == int(eb) for (a, ea), (b, eb) in zip(g_f, g_u))
    return {
        "method": "implequiv",
        "K": K,
        "seed": seed,
        "reconstructed_input_equal": bool(torch.equal(in_f, in_u)),
        "all_parameter_gradients_equal": bool(grad_equal),
        "final_input_gradient_equal": bool(torch.equal(fin_f.m, fin_u.m) and int(fin_f.e) == int(fin_u.e)),
    }


def step_mode_equivalence_test(K: int, seed: int = 0, hidden_bits: int = 9, xscale: int = 1, batches: int = 20):
    """
    Train two identical models for `batches` steps, one with per-layer stepping
    and one with batched stepping. Weights and optimizer accumulators must match
    exactly after every step.
    """
    lay_p = make_layers(K, seed)
    lay_b = make_layers(K, seed)
    opt_p = BlockSGD(parameters(lay_p), param_bits=12, lr_bits=2, clip=1)
    opt_b = BlockSGD(parameters(lay_b), param_bits=12, lr_bits=2, clip=1)
    rng = np.random.default_rng(seed)
    idx = torch.tensor(rng.permutation(len(XTR)), dtype=torch.long)

    all_equal = True
    first_divergence = None
    for step_i, st in enumerate(range(0, min(len(XTR), batches * BATCH), BATCH)):
        bi = idx[st : st + BATCH]
        xb, yb = XTR[bi], YTR[bi]

        fin_p = forward_reversible(xb, lay_p, xscale)
        backward_reconstruct(fin_p, yb, lay_p, hidden_bits, fused=True,
                             on_layer_grad=lambda k, gu, gv: opt_p.step_layer(k, gu, gv))

        fin_b = forward_reversible(xb, lay_b, xscale)
        gp, _, _ = backward_reconstruct(fin_b, yb, lay_b, hidden_bits, fused=True)
        opt_b.step(gp)

        w_eq = all(torch.equal(a, b) for a, b in zip(parameters(lay_p), parameters(lay_b)))
        a_eq = all(torch.equal(a, b) for a, b in zip(opt_p.acc, opt_b.acc))
        if not (w_eq and a_eq):
            all_equal = False
            first_divergence = step_i
            break

    return {
        "method": "stepequiv",
        "K": K,
        "seed": seed,
        "batches_compared": batches,
        "weights_and_accumulators_identical_every_step": bool(all_equal),
        "first_divergent_step": first_divergence,
    }


def equivalence_test(K: int, seed: int = 0, hidden_bits: int = 9, xscale: int = 1):
    """
    One-batch property test:
      cached activation == exact reconstructed activation
      cached gradients == reconstructed gradients (mantissa and exponent)
    """
    layers = make_layers(K, seed)
    xb, yb = XTR[:BATCH], YTR[:BATCH]
    final, cache = forward_cached(xb, layers, xscale)

    # Check every reconstructed layer input against the stored one.
    v = final.clone()
    layer_activation_equal = []
    for k in range(K - 1, -1, -1):
        if k < K - 1:
            v = v[:, invperm_for(k)]
        vin = layers[k].inverse(v)
        layer_activation_equal.append(bool(torch.equal(vin, cache[k])))
        v = vin

    gr, rec_input, final_gr = backward_reconstruct(final.clone(), yb, layers, hidden_bits)
    gc, final_gc = backward_cached(final.clone(), cache, yb, layers, hidden_bits)

    grad_equal = []
    exp_equal = []
    max_grad_abs_diff = 0
    for (mr, er), (mc, ec) in zip(gr, gc):
        grad_equal.append(bool(torch.equal(mr, mc)))
        exp_equal.append(int(er) == int(ec))
        max_grad_abs_diff = max(max_grad_abs_diff, int(torch.abs(mr - mc).max().item()))

    input_target = make_input(xb, xscale)
    return {
        "method": "equivalence",
        "K": K,
        "seed": seed,
        "hidden_mantissa_bits": hidden_bits,
        "network_inverse_exact": bool(torch.equal(inverse_network(final, layers), input_target)),
        "all_layer_inputs_equal": bool(all(layer_activation_equal)),
        "all_parameter_gradient_mantissas_equal": bool(all(grad_equal)),
        "all_parameter_gradient_exponents_equal": bool(all(exp_equal)),
        "max_parameter_gradient_mantissa_abs_diff": max_grad_abs_diff,
        "final_input_gradient_mantissa_equal": bool(torch.equal(final_gr.m, final_gc.m)),
        "final_input_gradient_exponent_equal": int(final_gr.e) == int(final_gc.e),
        "cached_activation_bytes": cache_bytes(cache),
        "reversible_activation_cache_bytes": 0,
    }


# -----------------------------------------------------------------------------
# Float baseline with same coupling topology
# -----------------------------------------------------------------------------
class FloatCoupling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.u = torch.nn.Linear(M, CFG.hidden, bias=False)
        self.v = torch.nn.Linear(CFG.hidden, M, bias=False)

    def forward(self, x):
        a, b = x[:, :M], x[:, M:]
        return torch.cat([a, b + self.v(torch.relu(self.u(a)))], dim=1)


class FloatNet(torch.nn.Module):
    def __init__(self, K):
        super().__init__()
        self.layers = torch.nn.ModuleList([FloatCoupling() for _ in range(K)])
        self.K = K

    def forward(self, x):
        v = torch.cat([x, torch.zeros_like(x)], dim=1)
        for k, layer in enumerate(self.layers):
            v = layer(v)
            if k < self.K - 1:
                v = v[:, perm_for(k)]
        return v[:, M : M + NCLASS]


def train_float(K: int, epochs: int, seed: int):
    torch.manual_seed(seed)
    net = FloatNet(K)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xf, xef = XTR.float(), XTE.float()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    saved_bytes = None

    for _ep in range(epochs):
        idx = torch.tensor(rng.permutation(len(xf)), dtype=torch.long)
        for st in range(0, len(xf), BATCH):
            bi = idx[st : st + BATCH]
            if saved_bytes is None:
                total = [0]

                def pack(t):
                    total[0] += t.numel() * t.element_size()
                    return t

                def unpack(t):
                    return t

                with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                    out = net(xf[bi])
                    loss = torch.nn.functional.cross_entropy(out, YTR[bi])
                    opt.zero_grad()
                    loss.backward()
                saved_bytes = total[0]
                opt.step()
            else:
                out = net(xf[bi])
                loss = torch.nn.functional.cross_entropy(out, YTR[bi])
                opt.zero_grad()
                loss.backward()
                opt.step()

    with torch.no_grad():
        acc = float((net(xef).argmax(dim=1) == YTE).float().mean().item())
    return {
        "method": "float",
        "K": K,
        "epochs": epochs,
        "seed": seed,
        "accuracy": acc,
        "saved_autograd_tensor_bytes_first_batch": int(saved_bytes or 0),
        "seconds": time.time() - t0,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--method",
        choices=["reversible", "cached", "legacy", "equiv", "implequiv", "stepequiv", "float"],
        default="reversible",
    )
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-bits", type=int, default=2)
    ap.add_argument("--hidden-bits", type=int, default=9)
    ap.add_argument("--xscale", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fused", dest="fused", action="store_true", default=True,
                    help="single recomputation per layer in reversible backward (default)")
    ap.add_argument("--no-fused", dest="fused", action="store_false",
                    help="original two-call inverse+parts form")
    ap.add_argument("--step-mode", choices=["per-layer", "batched"], default="per-layer",
                    help="per-layer: step each layer immediately (no O(K) grad buffer); "
                         "batched: collect all gradients then one step (original)")
    # Width and grouping options. Every default reproduces the original
    # int64 dense behaviour bit for bit.
    ap.add_argument("--state-bits", type=int, default=None,
                    help="two's complement wrap width of the reversible state "
                         "(default: none, i.e. full int64)")
    ap.add_argument("--weight-bits", type=int, default=None,
                    help="saturating width of U and V; also sets the init range "
                         "to +/-2^(bits-4) (default: none, init +/-8)")
    ap.add_argument("--accum-bits", type=int, default=None,
                    help="saturating width of GEMM accumulators (default: none)")
    ap.add_argument("--groups", type=int, default=1,
                    help="g residue-connected dense blocks per weight; "
                         "parameters and MACs are exactly 1/g (default: 1)")
    ap.add_argument("--phase-step", type=int, default=0,
                    help="layer k uses residue phase (k*step) %% g (default: 0)")
    ap.add_argument("--hidden-width", type=int, default=H,
                    help=f"coupling hidden width H (default: {H}); distinct from "
                         "--hidden-bits, which is the gradient mantissa width")
    ap.add_argument("--perm", choices=["default", "transpose"], default="default",
                    help="inter-layer permutation; 'transpose' is the "
                         "table-free dispersing grid transpose grouped coupling needs")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    set_config(
        state_bits=args.state_bits,
        weight_bits=args.weight_bits,
        accum_bits=args.accum_bits,
        groups=args.groups,
        phase_step=args.phase_step,
        hidden=args.hidden_width,
        perm_mode=args.perm,
    )

    if args.method == "equiv":
        result = equivalence_test(args.K, args.seed, args.hidden_bits, args.xscale)
    elif args.method == "implequiv":
        result = implementation_equivalence_test(args.K, args.seed, args.hidden_bits, args.xscale)
    elif args.method == "stepequiv":
        result = step_mode_equivalence_test(args.K, args.seed, args.hidden_bits, args.xscale)
    elif args.method == "float":
        result = train_float(args.K, args.epochs, args.seed)
    else:
        result = train_integer(
            args.method,
            args.K,
            args.epochs,
            args.seed,
            args.lr_bits,
            args.hidden_bits,
            args.xscale,
            fused=args.fused,
            step_mode=args.step_mode,
        )

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
