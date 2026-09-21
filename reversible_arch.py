#!/usr/bin/env python3
"""
Convolutional and attention coupling blocks for the reversible integer network.

Both reuse the machinery already verified for the MLP coupling: integer
arithmetic with power-of-two rescaling, a tensor-shared exponent on the
backward adjoint, exact additive coupling, and cache-free backward by exact
inversion. Only the function f inside the coupling changes.

  conv    v = [a, b] over (C, S, S); f(a) = conv3x3 -> relu -> conv3x3
          A 3x3 convolution is an im2col matrix times a weight matrix, so it
          reduces to the same dense linear backward the MLP uses; the only
          extra piece is folding the input gradient back from column form.

  attn    v = [a, b] over (T, D); f(a) = single-head attention -> out proj
          Softmax is replaced by ReLU plus a power-of-two rescale, so the whole
          block stays integer and deterministic. The two activation-by-
          activation matmuls (Q@K^T and A@V) need their own backward, which
          follows the same shared-exponent pattern.

Additive coupling is exactly invertible for ANY deterministic integer f, so
reversibility does not depend on what is inside the block.

  python reversible_arch.py --arch conv --mode verify
  python reversible_arch.py --arch attn --mode train --K 8 --epochs 2
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reversible_integer_final as R  # noqa: E402
from reversible_integer_final import (  # noqa: E402
    SharedGrad, BlockSGD, align_shared, concat_shared, dense_linear_backward,
    norm_block, pow2norm, rshift0, saturate, wrap_state,
)

TB = R.TB
NCLASS = 10
OFFSETS = [(di, dj) for di in range(3) for dj in range(3)]


# -----------------------------------------------------------------------------
# im2col / col2im for the 3x3 convolution
# -----------------------------------------------------------------------------
def im2col3x3(x):
    """(N, C, S, S) -> (N*S*S, 9*C), zero padded, offset-major per position."""
    n, c, s, _ = x.shape
    xp = F.pad(x, (1, 1, 1, 1))
    cols = torch.stack([xp[:, :, di:di + s, dj:dj + s] for di, dj in OFFSETS],
                       dim=2)                      # (N, C, 9, S, S)
    return cols.permute(0, 3, 4, 2, 1).reshape(n * s * s, 9 * c)


def col2im3x3(gcol, n, c, s):
    """Adjoint of im2col3x3: scatter-add columns back onto the image."""
    g = gcol.reshape(n, s, s, 9, c).permute(0, 4, 3, 1, 2)   # (N, C, 9, S, S)
    gp = torch.zeros((n, c, s + 2, s + 2), dtype=torch.int64)
    for idx, (di, dj) in enumerate(OFFSETS):
        gp[:, :, di:di + s, dj:dj + s] += g[:, :, idx]
    return gp[:, :, 1:1 + s, 1:1 + s]


# -----------------------------------------------------------------------------
# Convolutional coupling
# -----------------------------------------------------------------------------
class ConvCoupling:
    """f(a) = conv3x3(Ch -> hidden) -> relu -> conv3x3(hidden -> Ch)."""

    def __init__(self, generator, channels, hidden, side, cfg=None):
        self.cfg = cfg if cfg is not None else R.CFG
        self.c = channels          # channels per half
        self.h = hidden
        self.s = side
        lim = 8 if self.cfg.weight_bits is None else (1 << (self.cfg.weight_bits - 4))
        # Stored in im2col orientation: (out, 9*in).
        self.U = torch.randint(-lim, lim + 1, (hidden, 9 * channels),
                               dtype=torch.int64, generator=generator)
        self.V = torch.randint(-lim, lim + 1, (channels, 9 * hidden),
                               dtype=torch.int64, generator=generator)

    def blocks(self):
        return [self.U, self.V]

    def parts(self, a):
        """a is (N, C, S, S). Returns f and every intermediate backward needs."""
        n = a.shape[0]
        col1 = im2col3x3(a)
        z2d, s1 = pow2norm(saturate(col1 @ self.U.T, self.cfg.accum_bits), TB + 2)
        z = z2d.reshape(n, self.s, self.s, self.h).permute(0, 3, 1, 2)
        hpos = torch.clamp_min(z, 0)
        col2 = im2col3x3(hpos)
        f2d, s2 = pow2norm(saturate(col2 @ self.V.T, self.cfg.accum_bits), TB + 2)
        f = f2d.reshape(n, self.s, self.s, self.c).permute(0, 3, 1, 2)
        return f, z, hpos, s1, s2, col1, col2

    def forward(self, v):
        a, b = v[:, :self.c], v[:, self.c:]
        f = self.parts(a)[0]
        return torch.cat([a, wrap_state(b + f, self.cfg.state_bits)], dim=1)

    def inverse(self, v):
        a, bout = v[:, :self.c], v[:, self.c:]
        f = self.parts(a)[0]
        return torch.cat([a, wrap_state(bout - f, self.cfg.state_bits)], dim=1)

    def backward(self, ga, gb, a, cached, hidden_bits):
        """ga, gb are SharedGrad over (N, C, S, S) flattened per position."""
        z, hpos, s1, s2, col1, col2 = cached
        n = a.shape[0]
        cfg = self.cfg

        # gb enters f, which was produced per spatial position.
        gb2d = SharedGrad(
            gb.m.permute(0, 2, 3, 1).reshape(n * self.s * self.s, self.c), gb.e)
        gcol2, gV = dense_linear_backward(gb2d, col2, self.V, s2, hidden_bits,
                                          30, cfg.accum_bits)
        gh = col2im3x3(gcol2.m, n, self.h, self.s)
        gh = gh * (z > 0)
        gh2d = SharedGrad(
            gh.permute(0, 2, 3, 1).reshape(n * self.s * self.s, self.h), gcol2.e)
        gcol1, gU = dense_linear_backward(gh2d, col1, self.U, s1, hidden_bits,
                                          30, cfg.accum_bits)
        da = col2im3x3(gcol1.m, n, self.c, self.s)
        da2d = SharedGrad(da.permute(0, 2, 3, 1).reshape(n * self.s * self.s, self.c),
                          gcol1.e)

        ga2d = SharedGrad(
            ga.m.permute(0, 2, 3, 1).reshape(n * self.s * self.s, self.c), ga.e)
        nm, ne = align_shared(ga2d.m, ga2d.e, da2d.m, da2d.e, hidden_bits)
        nga = SharedGrad(
            nm.reshape(n, self.s, self.s, self.c).permute(0, 3, 1, 2), ne)
        return nga, [gU, gV]


# -----------------------------------------------------------------------------
# Attention coupling
# -----------------------------------------------------------------------------
def bmm_backward_shared(g, X, Y, s, hidden_bits, accum_bits):
    """
    Backward of Z = (X @ Y) >> s for batched activations, one shared exponent.

    s is per row of Z, i.e. per (batch, row) pair, exactly like the dense case.
    Returns gX and gY as SharedGrad.
    """
    nb, rows, _cols = g.m.shape
    s = s.reshape(nb, rows, 1).to(torch.int64)
    sample_e = g.e - s
    E = int(sample_e.max().item())
    shift = E - sample_e

    gx = saturate(rshift0(R.imatmul(g.m, Y.transpose(1, 2)), shift), accum_bits)
    gxm, gxe = norm_block(gx, E, hidden_bits)
    aligned = rshift0(g.m, shift)
    gy = saturate(R.imatmul(X.transpose(1, 2), aligned), accum_bits)
    gym, gye = norm_block(gy, E, hidden_bits)
    return SharedGrad(gxm, gxe), SharedGrad(gym, gye)


# Diagnostic knob for the score nonlinearity. None keeps the plain ReLU;
# an integer k makes it a power-of-two leaky form, negatives >> k. Both are
# deterministic integer functions, so exact reversibility is unaffected either
# way -- this only changes how much of the score gradient survives.
SCORE_LEAK = None


# Base-2 attention. The bisection that motivated it: a float mirror of this
# model reproduces 64/64 training pairs by greedy decoding with softmax, 64/64
# with softmax *and* every integer rescaling put back, and 0/64 with softmax
# removed and nothing else changed. So what was missing is not precision, not
# the +-1 updates, not the power-of-two rescaling -- it is the competition
# softmax creates, where a large score actually suppresses the small ones.
#
# Power-of-two normalisation cannot supply that: dividing a row by a constant
# leaves every ratio exactly as it was. This does, without an exponential or a
# division, because in base two the exponential *is* a shift:
#
#     w_i = 1 << ((s_i - max_j s_j) >> SCORE_EXP2 + CAP)
#
# The row maximum is subtracted first so the largest weight is 2**CAP and the
# rest fall off geometrically; anything more than CAP*2**SCORE_EXP2 below the
# maximum lands on zero, which is the same tail truncation softmax gives in
# fixed point. Row mass is then bounded by construction, so no separate
# normalisation is needed. Deterministic, integer, exactly reversible.
SCORE_EXP2 = None       # None keeps the linear/leaky map; an int enables base-2
SCORE_EXP2_CAP = 12     # weight of the row maximum, as a power of two


def score_act(s):
    if SCORE_EXP2 is not None:
        return s                       # handled in the attention block, needs the mask
    if SCORE_LEAK is None:
        return torch.clamp_min(s, 0)
    return torch.where(s > 0, s, rshift0(s, SCORE_LEAK))


def exp2_attend(s, mask, shift_bits=None, cap=None):
    """Base-2 softmax analogue: 2**((s - rowmax) >> shift), zero where masked.

    The subtraction of the row maximum is what makes this a competition rather
    than a rescaling, and it also keeps every exponent non-positive so the
    shift can never overflow. Masked entries take no part in the maximum, for
    the same reason they take no part in any other reduction here.
    """
    k = SCORE_EXP2 if shift_bits is None else shift_bits
    c = SCORE_EXP2_CAP if cap is None else cap
    valid = mask.to(torch.bool)
    neg = torch.iinfo(torch.int64).min // 4
    mx = torch.where(valid, s, torch.full_like(s, neg)).amax(dim=-1, keepdim=True)
    e = rshift0(s - mx, k) + c              # <= cap, and negative far from the max
    e = torch.clamp(e, min=0, max=40)
    return (torch.ones_like(s) << e) * mask


SCORE_EXP2_COUPLED = True   # False reverts to the per-entry (diagonal) adjoint


def exp2_attend_backward(gm, s, mask, shift_bits=None, cap=None):
    """Adjoint of exp2_attend, with softmax's row coupling.

    A float bisection settled what generation needs from attention. With the
    base-2 forward above and softmax's *backward* substituted in, a float
    mirror reproduces 64/64 training pairs in 1000 steps; with softmax's
    forward and a per-entry (diagonal) backward it reproduces 0/64. So the
    shape of the weights is not the issue -- the missing piece was the term
    through which raising one score lowers every other weight in the row:

        dL/ds_i  =  w_i * (g_i - gbar),     gbar = sum_j w_j g_j / sum_j w_j

    gbar is one integer per row: exact int64 products and sums (2**12 weights
    times 2**16 gradients over at most a few dozen keys stays under 2**33), then
    one floor division by the row mass. No float enters. The 1/2**k from the
    base-2 exponent and the ln 2 are folded into the right shift by k, as in
    every other power-of-two adjoint here, and the overall row-mass scale is
    carried by the caller's shared exponent. Entries the forward zeroed or
    clamped contribute nothing, exactly as a ReLU's do.
    """
    k = SCORE_EXP2 if shift_bits is None else shift_bits
    c = SCORE_EXP2_CAP if cap is None else cap
    valid = mask.to(torch.bool)
    neg = torch.iinfo(torch.int64).min // 4
    mx = torch.where(valid, s, torch.full_like(s, neg)).amax(dim=-1, keepdim=True)
    e = rshift0(s - mx, k) + c
    live = (e > 0) & (e < 40) & valid
    w = (torch.ones_like(s) << torch.clamp(e, min=0, max=40)) * mask
    if not SCORE_EXP2_COUPLED:
        return rshift0(gm * w, k) * live
    num = (w * gm).sum(dim=-1, keepdim=True)
    den = torch.clamp(w.sum(dim=-1, keepdim=True), min=1)
    gbar = torch.div(num, den, rounding_mode="floor")
    return rshift0(w * (gm - gbar), k) * live


def score_act_backward(gm, s):
    """Chain rule through score_act: 1 where s>0, else 2**-SCORE_LEAK."""
    if SCORE_EXP2 is not None:
        return gm                      # handled by exp2_attend_backward
    if SCORE_LEAK is None:
        return gm * (s > 0)
    return torch.where(s > 0, gm, rshift0(gm, SCORE_LEAK))


class AttnCoupling:
    """f(a) = single-head attention with a ReLU score map, then an out proj."""

    def __init__(self, generator, dim, tokens, cfg=None):
        self.cfg = cfg if cfg is not None else R.CFG
        self.d = dim            # channels per half
        self.t = tokens
        lim = 8 if self.cfg.weight_bits is None else (1 << (self.cfg.weight_bits - 4))
        mk = lambda: torch.randint(-lim, lim + 1, (dim, dim),
                                   dtype=torch.int64, generator=generator)
        self.Wq, self.Wk, self.Wv, self.Wo = mk(), mk(), mk(), mk()

    def blocks(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo]

    def parts(self, a):
        """a is (N, T, d)."""
        n = a.shape[0]
        a2d = a.reshape(n * self.t, self.d)
        cfg = self.cfg
        q2, sq = pow2norm(saturate(a2d @ self.Wq.T, cfg.accum_bits), TB + 2)
        k2, sk = pow2norm(saturate(a2d @ self.Wk.T, cfg.accum_bits), TB + 2)
        v2, sv = pow2norm(saturate(a2d @ self.Wv.T, cfg.accum_bits), TB + 2)
        Q = q2.reshape(n, self.t, self.d)
        K = k2.reshape(n, self.t, self.d)
        V = v2.reshape(n, self.t, self.d)

        raw_s = saturate(Q @ K.transpose(1, 2), cfg.accum_bits)
        s2d, ss = pow2norm(raw_s.reshape(n * self.t, self.t), TB + 2)
        S = s2d.reshape(n, self.t, self.t)
        A = score_act(S)                     # ReLU (or leaky) instead of softmax

        raw_o = saturate(A @ V, cfg.accum_bits)
        o2d, so = pow2norm(raw_o.reshape(n * self.t, self.d), TB + 2)
        O = o2d.reshape(n, self.t, self.d)

        f2d, sf = pow2norm(saturate(o2d @ self.Wo.T, cfg.accum_bits), TB + 2)
        f = f2d.reshape(n, self.t, self.d)
        return (f, a2d, Q, K, V, S, A, O, o2d, sq, sk, sv, ss, so, sf)

    def forward(self, v):
        a, b = v[:, :, :self.d], v[:, :, self.d:]
        f = self.parts(a)[0]
        return torch.cat([a, wrap_state(b + f, self.cfg.state_bits)], dim=2)

    def inverse(self, v):
        a, bout = v[:, :, :self.d], v[:, :, self.d:]
        f = self.parts(a)[0]
        return torch.cat([a, wrap_state(bout - f, self.cfg.state_bits)], dim=2)

    def backward(self, ga, gb, a, cached, hidden_bits):
        a2d, Q, K, V, S, A, O, o2d, sq, sk, sv, ss, so, sf = cached
        n = a.shape[0]
        cfg = self.cfg
        ab = cfg.accum_bits

        gb2d = SharedGrad(gb.m.reshape(n * self.t, self.d), gb.e)
        gO2d, gWo = dense_linear_backward(gb2d, o2d, self.Wo, sf, hidden_bits,
                                          30, ab)
        gO = SharedGrad(gO2d.m.reshape(n, self.t, self.d), gO2d.e)

        gA, gV = bmm_backward_shared(gO, A, V, so, hidden_bits, ab)
        gS = SharedGrad(score_act_backward(gA.m, S), gA.e)
        gQ, gKt = bmm_backward_shared(gS, Q, K.transpose(1, 2), ss, hidden_bits, ab)
        gK = SharedGrad(gKt.m.transpose(1, 2), gKt.e)

        parts = []
        for grad, W, shift in ((gQ, self.Wq, sq), (gK, self.Wk, sk),
                               (gV, self.Wv, sv)):
            g2d = SharedGrad(grad.m.reshape(n * self.t, self.d), grad.e)
            gin, gw = dense_linear_backward(g2d, a2d, W, shift, hidden_bits, 30, ab)
            parts.append((gin, gw))

        da = parts[0][0]
        for gin, _gw in parts[1:]:
            m, e = align_shared(da.m, da.e, gin.m, gin.e, hidden_bits)
            da = SharedGrad(m, e)

        ga2d = SharedGrad(ga.m.reshape(n * self.t, self.d), ga.e)
        nm, ne = align_shared(ga2d.m, ga2d.e, da.m, da.e, hidden_bits)
        nga = SharedGrad(nm.reshape(n, self.t, self.d), ne)
        gws = [parts[0][1], parts[1][1], parts[2][1], gWo]
        return nga, gws


# -----------------------------------------------------------------------------
# Network: forward, exact inverse, cache-free and cached backward
# -----------------------------------------------------------------------------
class RevNet:
    def __init__(self, arch, K, seed, cfg, channels=8, hidden=32, side=7,
                 tokens=16, dim=32):
        self.arch = arch
        self.cfg = cfg
        gen = torch.Generator().manual_seed(seed)
        if arch == "conv":
            self.layers = [ConvCoupling(gen, channels, hidden, side, cfg)
                           for _ in range(K)]
            self.half = channels
            self.perm = torch.roll(torch.arange(2 * channels), channels)
        else:
            self.layers = [AttnCoupling(gen, dim, tokens, cfg) for _ in range(K)]
            self.half = dim
            self.perm = torch.roll(torch.arange(2 * dim), dim)
        self.inv_perm = torch.argsort(self.perm)
        self.cdim = 1 if arch == "conv" else 2

    def parameters(self):
        return [w for L in self.layers for w in L.blocks()]

    def _permute(self, v, p):
        return v[:, p] if self.cdim == 1 else v[:, :, p]

    def forward(self, v):
        for k, L in enumerate(self.layers):
            v = L.forward(v)
            if k < len(self.layers) - 1:
                v = self._permute(v, self.perm)
        return v

    def inverse(self, v):
        for k in range(len(self.layers) - 1, -1, -1):
            if k < len(self.layers) - 1:
                v = self._permute(v, self.inv_perm)
            v = self.layers[k].inverse(v)
        return v

    def forward_cached(self, v):
        tape = []
        for k, L in enumerate(self.layers):
            a = v[:, :self.half] if self.cdim == 1 else v[:, :, :self.half]
            tape.append((a, L.parts(a)[1:]))
            v = L.forward(v)
            if k < len(self.layers) - 1:
                v = self._permute(v, self.perm)
        return v, tape

    def output_grad(self, v, y, hidden_bits):
        flat = v.reshape(v.shape[0], -1)
        out = flat[:, self.half_offset():self.half_offset() + NCLASS]
        n = len(y)
        masked = out.clone()
        masked[torch.arange(n), y] = torch.iinfo(torch.int64).min // 4
        j = masked.argmax(dim=1)
        active = out[torch.arange(n), j] + 1 > out[torch.arange(n), y]
        gm = torch.zeros_like(flat)
        ii = torch.arange(n)[active]
        gm[ii, self.half_offset() + j[active]] += 1
        gm[ii, self.half_offset() + y[active]] -= 1
        gm, ge = norm_block(gm, 0, hidden_bits)
        return SharedGrad(gm.reshape(v.shape), ge)

    def half_offset(self):
        return self.layers[0].c * self.layers[0].s ** 2 if self.arch == "conv" \
            else self.layers[0].d

    def logits(self, v):
        flat = v.reshape(v.shape[0], -1)
        o = self.half_offset()
        return flat[:, o:o + NCLASS]

    def split(self, g):
        if self.cdim == 1:
            return (SharedGrad(g.m[:, :self.half], g.e),
                    SharedGrad(g.m[:, self.half:], g.e))
        return (SharedGrad(g.m[:, :, :self.half], g.e),
                SharedGrad(g.m[:, :, self.half:], g.e))

    def join(self, nga, gb, hidden_bits):
        # concat_shared in the reference module concatenates on dim 1, which is
        # the channel axis for conv but the token axis for attention. Rejoin on
        # whichever axis this architecture splits on.
        E = max(int(nga.e), int(gb.e))
        a = rshift0(nga.m, E - int(nga.e))
        b = rshift0(gb.m, E - int(gb.e))
        m, e = norm_block(torch.cat([a, b], dim=self.cdim), E, hidden_bits)
        return SharedGrad(m, e)

    def backward(self, final_v, y, hidden_bits, tape=None, on_layer_grad=None):
        """tape=None -> cache-free: activations are recovered by exact inversion."""
        v = final_v
        g = self.output_grad(v, y, hidden_bits)
        collected = []
        for k in range(len(self.layers) - 1, -1, -1):
            if k < len(self.layers) - 1:
                v = self._permute(v, self.inv_perm)
                g = SharedGrad(self._permute(g.m, self.inv_perm), g.e)
            L = self.layers[k]
            if tape is None:
                a = v[:, :self.half] if self.cdim == 1 else v[:, :, :self.half]
                cached = L.parts(a)
                f, rest = cached[0], cached[1:]
                bpart = v[:, self.half:] if self.cdim == 1 else v[:, :, self.half:]
                vin = torch.cat([a, wrap_state(bpart - f, self.cfg.state_bits)],
                                dim=self.cdim)
            else:
                a, rest = tape[k]
                vin = None
            ga, gb = self.split(g)
            nga, gws = L.backward(ga, gb, a, rest, hidden_bits)
            g = self.join(nga, gb, hidden_bits)
            if on_layer_grad is None:
                collected.append((k, gws))
            else:
                on_layer_grad(k, gws)
            if tape is None:
                v = vin
        if on_layer_grad is not None:
            return None, g
        collected.sort(key=lambda t: t[0])
        return [w for _k, gws in collected for w in gws], g


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_data(arch, train_limit, test_limit=0):
    import fashion_mnist_data as FM
    if arch == "conv":
        xtr, ytr, xte, yte, _ = FM.load(M=784, train_limit=train_limit,
                                        test_limit=test_limit)
        # 28x28x1 -> 7x7x16 by space-to-depth, a pure permutation.
        def prep(x):
            v = x.reshape(-1, 1, 28, 28)
            v = F.pixel_unshuffle(v, 4)          # (N, 16, 7, 7)
            return v.contiguous()
        return prep(xtr), ytr, prep(xte), yte
    xtr, ytr, xte, yte, _ = FM.load(M=1024, train_limit=train_limit,
                                    test_limit=test_limit)
    return (xtr.reshape(-1, 16, 64), ytr, xte.reshape(-1, 16, 64), yte)


def make_net(arch, K, seed, cfg):
    if arch == "conv":
        return RevNet("conv", K, seed, cfg, channels=8, hidden=32, side=7)
    return RevNet("attn", K, seed, cfg, tokens=16, dim=32)


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
def hinge_loss(net, v, y):
    """Integer multiclass hinge, used only by the finite-difference probe."""
    out = net.logits(v)
    n = len(y)
    masked = out.clone()
    masked[torch.arange(n), y] = torch.iinfo(torch.int64).min // 4
    j = masked.argmax(dim=1)
    margin = out[torch.arange(n), j] + 1 - out[torch.arange(n), y]
    return int(torch.clamp_min(margin, 0).sum().item())


def step_all(opt, k, gws, nper):
    for c, (gm, _ge) in enumerate(gws):
        opt._step_one(k * nper + c, gm)


def verify(arch, K, seed, cfg, hidden_bits, xb, yb, probes=40, delta=8):
    net = make_net(arch, K, seed, cfg)
    nper = len(net.layers[0].blocks())

    final = net.forward(xb)
    inverse_exact = bool(torch.equal(net.inverse(final), xb))

    # Cache-free versus cached: identical arithmetic, different activation source.
    fin_c, tape = net.forward_cached(xb)
    gc, gfin_c = net.backward(fin_c, yb, hidden_bits, tape=tape)
    gr, gfin_r = net.backward(final.clone(), yb, hidden_bits, tape=None)
    grads_equal = (len(gc) == len(gr)
                   and all(torch.equal(a, b) and int(ea) == int(eb)
                           for (a, ea), (b, eb) in zip(gc, gr)))
    ingrad_equal = bool(torch.equal(gfin_c.m, gfin_r.m)
                        and int(gfin_c.e) == int(gfin_r.e))

    # Per-layer stepping versus batched stepping, over real training steps.
    def run(per_layer, steps=6, bs=16):
        nt = make_net(arch, K, seed, cfg)
        opt = BlockSGD(nt.parameters(), param_bits=12, lr_bits=2, clip=1)
        for i in range(steps):
            lo = (i * bs) % max(1, len(xb) - bs)
            sl = slice(lo, lo + bs)
            fv = nt.forward(xb[sl])
            if per_layer:
                nt.backward(fv, yb[sl], hidden_bits,
                            on_layer_grad=lambda k, gws: step_all(opt, k, gws, nper))
            else:
                gws, _ = nt.backward(fv, yb[sl], hidden_bits)
                opt.step(gws)
        return [w.clone() for w in nt.parameters()]
    step_equal = all(torch.equal(a, b) for a, b in zip(run(False), run(True)))

    # Finite difference: does the integer gradient actually point downhill?
    net2 = make_net(arch, K, seed, cfg)
    gws, _ = net2.backward(net2.forward(xb), yb, hidden_bits)
    params = net2.parameters()
    rng = np.random.default_rng(seed)
    agree = total = 0
    for _ in range(probes):
        pi = int(rng.integers(len(params)))
        flat = params[pi].reshape(-1)
        ci = int(rng.integers(flat.numel()))
        gsign = int(torch.sign(gws[pi][0].reshape(-1)[ci]).item())
        if gsign == 0:
            continue
        orig = int(flat[ci].item())
        flat[ci] = orig + delta
        lp = hinge_loss(net2, net2.forward(xb), yb)
        flat[ci] = orig - delta
        lm = hinge_loss(net2, net2.forward(xb), yb)
        flat[ci] = orig
        if lp == lm:
            continue
        total += 1
        agree += int(np.sign(lp - lm) == gsign)

    return {
        "arch": arch, "K": K, "seed": seed,
        "inverse_exact": inverse_exact,
        "cachefree_equals_cached_grads": bool(grads_equal),
        "cachefree_equals_cached_input_grad": ingrad_equal,
        "per_layer_step_equals_batched": bool(step_equal),
        "fd_sign_agreement": (round(agree / total, 3) if total else None),
        "fd_probes_used": total,
        "parameter_count": int(sum(w.numel() for w in net.parameters())),
    }


# -----------------------------------------------------------------------------
# Training and activation memory
# -----------------------------------------------------------------------------
def train(arch, K, seed, cfg, hidden_bits, epochs, batch, xtr, ytr, xte, yte,
          lr_bits=2):
    R.reset_stats()
    net = make_net(arch, K, seed, cfg)
    opt = BlockSGD(net.parameters(), param_bits=12, lr_bits=lr_bits, clip=1)
    nper = len(net.layers[0].blocks())
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for _ep in range(epochs):
        idx = torch.tensor(rng.permutation(len(xtr)), dtype=torch.long)
        for st in range(0, len(xtr), batch):
            bi = idx[st:st + batch]
            fv = net.forward(xtr[bi])
            net.backward(fv, ytr[bi], hidden_bits,
                         on_layer_grad=lambda k, gws: step_all(opt, k, gws, nper))
    with torch.no_grad():
        acc = float((net.logits(net.forward(xte)).argmax(1) == yte)
                    .float().mean().item())
    vt = net.forward(xte[:64])
    return {
        "arch": arch, "K": K, "seed": seed, "epochs": epochs, "batch": batch,
        "accuracy": acc,
        "inverse_exact": bool(torch.equal(net.inverse(vt), xte[:64])),
        "parameter_count": int(sum(w.numel() for w in net.parameters())),
        "max_activation": int(torch.abs(vt).max().item()),
        "state_wrap_frac": (round(R.STATS["state_wrap"] / R.STATS["state_elems"], 6)
                            if R.STATS["state_elems"] else 0.0),
        "accum_saturations": int(R.STATS["accum_saturations"]),
        "wall_seconds": round(time.time() - t0, 1),
    }


def measure_activation(arch, K, seed, cfg, hidden_bits, xb, yb):
    """Peak distinct activation bytes held during backward, cached vs cache-free."""
    from reversible_integer_memory_bench import dedup_bytes
    net = make_net(arch, K, seed, cfg)
    fin_free = net.forward(xb)
    _fin_cached, tape = net.forward_cached(xb)

    cached_bytes = dedup_bytes(
        [a for a, _r in tape]
        + [t for _a, rest in tape for t in rest if isinstance(t, torch.Tensor)])

    peak = 0
    v = fin_free
    g = net.output_grad(v, yb, hidden_bits)
    for k in range(len(net.layers) - 1, -1, -1):
        if k < len(net.layers) - 1:
            v = net._permute(v, net.inv_perm)
            g = SharedGrad(net._permute(g.m, net.inv_perm), g.e)
        L = net.layers[k]
        a = v[:, :net.half] if net.cdim == 1 else v[:, :, :net.half]
        cached = L.parts(a)
        f, rest = cached[0], cached[1:]
        bp = v[:, net.half:] if net.cdim == 1 else v[:, :, net.half:]
        vin = torch.cat([a, wrap_state(bp - f, cfg.state_bits)], dim=net.cdim)
        peak = max(peak, dedup_bytes(
            [v, vin, a, f] + [t for t in rest if isinstance(t, torch.Tensor)]))
        ga, gb = net.split(g)
        nga, _gws = L.backward(ga, gb, a, rest, hidden_bits)
        g = net.join(nga, gb, hidden_bits)
        v = vin
    return {"arch": arch, "K": K, "batch": int(xb.shape[0]),
            "activation_peak_bytes_cachefree": int(peak),
            "activation_bytes_cached": int(cached_bytes),
            "parameter_bytes": int(sum(w.numel() * 8 for w in net.parameters()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["conv", "attn"], default="conv")
    ap.add_argument("--mode", choices=["verify", "train", "memory"], default="verify")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--train-limit", type=int, default=20000)
    ap.add_argument("--state-bits", type=int, default=12)
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--accum-bits", type=int, default=32)
    ap.add_argument("--hidden-bits", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--score-leak", type=int, default=None,
                    help="attn only: power-of-two leaky score map, negatives >> k")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    torch.set_num_threads(a.threads or max(1, (os.cpu_count() or 2)))
    global SCORE_LEAK
    SCORE_LEAK = a.score_leak
    cfg = R.RunConfig(state_bits=a.state_bits, weight_bits=a.weight_bits,
                      accum_bits=a.accum_bits)
    R.CFG = cfg

    rows = []
    if a.mode == "train":
        xtr, ytr, xte, yte = load_data(a.arch, a.train_limit, 0)
        for s in a.seeds:
            r = train(a.arch, a.K, s, cfg, a.hidden_bits, a.epochs, a.batch,
                      xtr, ytr, xte, yte)
            rows.append(r); print(json.dumps(r), flush=True)
    else:
        xtr, ytr, _xte, _yte = load_data(a.arch, 512, 1)
        fn = verify if a.mode == "verify" else measure_activation
        for K in a.depths:
            r = fn(a.arch, K, a.seeds[0], cfg, a.hidden_bits, xtr[:32], ytr[:32])
            rows.append(r); print(json.dumps(r), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
