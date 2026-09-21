#!/usr/bin/env python3
"""
Tiny reversible-integer Japanese -> Korean sequence-to-sequence translator.

Every integer primitive is imported from the already-verified modules; none of
them is reimplemented here:

  reversible_integer_final : RunConfig/set_config, wrap_state, saturate, rshift0,
                             pow2norm, norm_block, SharedGrad, align_shared,
                             dense_linear_backward, BlockSGD (error-feedback,
                             per-layer and batched stepping)
  reversible_arch          : bmm_backward_shared, score_act, score_act_backward

Integer semantics, stated explicitly because the names are easy to misread:

  weight_bits = 8    SIGNED. Weights are clamped to [-2^7, 2^7-1] and initialised
                     uniform in [-2^(8-4), 2^(8-4)] = [-16, 16].
  state_bits = 16    SIGNED two's-complement WRAP width of the reversible state,
                     i.e. values live in [-2^15, 2^15). Wrap is exactly
                     invertible, so reversibility survives it.
  accum_bits = 32    SIGNED saturating width of GEMM accumulators.
  hidden_bits = 16   MAGNITUDE bits of the backward adjoint mantissa, on top of a
                     separate sign; norm_block targets this many bits of |m| and
                     carries the discarded scale in a per-tensor exponent. This
                     is the "grad-bits" knob; 16 is the Fashion-MNIST attention
                     finding carried over.

Structure. The reversible state is [a, b] over (B, T, 2*D); each block is an
additive coupling [a,b] -> [a, b + F(a)] followed by a half swap, so F may be
attention or an FFN without affecting invertibility. Encoder memory, masks and
token ids are deterministic external inputs to F, so decoder blocks invert with
b = b' - F(a, memory, mask).

Embedding tables and the output projection are NOT reversible and their
activations are stored. That cost is O(1) in reversible depth, which is the
claim under test; it is reported separately and never folded into "activation".

  python reversible_jako_mt.py --mode train --corpus tatoeba --max-pairs 512
  python reversible_jako_mt.py --mode generate --ckpt jako_ckpt.pt
"""

import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_integer_final as R          # noqa: E402
import reversible_arch as A                   # noqa: E402
import reversible_jako_data as D              # noqa: E402
from reversible_integer_final import (        # noqa: E402
    SharedGrad, BlockSGD, align_shared, dense_linear_backward,
    norm_block, pow2norm, rshift0, saturate, wrap_state,
)
from reversible_arch import (bmm_backward_shared, score_act,
                            score_act_backward, exp2_attend,
                            exp2_attend_backward)

TB = R.TB
PAD, BOS, EOS, UNK = D.PAD, D.BOS, D.EOS, D.UNK

# Cross-attention query anchoring.
#
# Measured on the ep6 checkpoint: the decoder state accumulates a shared
# component with depth (row cosine across target positions 0.31 -> 0.95, common
# mode 0.55 -> 0.98, mean |a| 8 -> 56), so every position projects to nearly the
# same query (Q row cosine 0.99) and every position reads the same weighted
# average of the source. Subtracting the BOS row before Wq removes the shared
# part: with a_t = c + r_t, a_t - a_0 = r_t - r_0 and c is gone.
#
# The anchor is a_0, not the sequence mean. The sequence mean would put future
# target rows into the position-0 query -- information that exists under teacher
# forcing and not under autoregressive decoding. a_0 depends only on BOS, so
# every query stays a function of tokens 0..t. reversible_jako_causalgate.py
# checks that, and position 0 keeps its own query rather than being zeroed.
CROSS_QUERY_ANCHOR = False


def corrupt_prefix(tin, pct, gen):
    """
    Replace a fraction of the gold decoder input with UNK, during training only.

    Measured on this model, the gold target prefix is worth 58.4 accuracy points
    and the source 2.2, so teacher forcing hands the decoder a shortcut that is
    an order of magnitude easier than translating. Corrupting part of the prefix
    makes that shortcut insufficient without touching the objective, the integer
    semantics or the reversible structure.

    UNK, not PAD: PAD carries length and mask meaning, and substituting it would
    change what the model is allowed to attend to and how long it thinks the
    sentence is. UNK is an ordinary vocabulary entry with no such role. BOS is
    left in place so position 0 keeps its sentinel, and PAD positions stay PAD.
    The draw is an integer threshold, so no float enters the path.
    """
    if pct <= 0:
        return tin
    r = torch.randint(0, 100, tin.shape, dtype=torch.int64, generator=gen)
    hit = (r < pct) & (tin != PAD) & (tin != BOS)
    return torch.where(hit, torch.full_like(tin, UNK), tin)


def scheduled_sample(model, src, tin, pct, gen):
    """
    Replace part of the gold decoder input with the model's own argmax.

    Discrete token substitution, so nothing float or stochastic-differentiable
    enters the path: one teacher-forced forward produces integer predictions,
    an integer threshold picks the positions, and the chosen positions are
    overwritten with those token ids.

    Causality holds. The prediction scored at target position t is a function of
    decoder inputs 0..t, and it is written to input position t+1, so a
    substituted token never depends on anything at or after the position it
    feeds. BOS and PAD are left alone for the same reasons as in
    `corrupt_prefix`.
    """
    if pct <= 0:
        return tin
    mem, _c, _i, _x = model.encode(src)
    dec, _c2, _i2, _x2 = model.decode(tin, mem, src)
    lg, _s = model.logits(dec)
    pred = lg.argmax(dim=2)                       # (B, T): prediction for pos t
    out = tin.clone()
    r = torch.randint(0, 100, tin.shape, dtype=torch.int64, generator=gen)
    hit = (r < pct) & (tin != PAD) & (tin != BOS)
    hit[:, 0] = False
    shifted = torch.zeros_like(tin)
    shifted[:, 1:] = pred[:, :-1]                 # input t+1 <- prediction at t
    return torch.where(hit, shifted, out)


# -----------------------------------------------------------------------------
# Weight helpers
# -----------------------------------------------------------------------------
def init_w(shape, cfg, gen):
    lim = 8 if cfg.weight_bits is None else (1 << (cfg.weight_bits - 4))
    return torch.randint(-lim, lim + 1, shape, dtype=torch.int64, generator=gen)


def lin_fwd(x2d, W, cfg):
    """y = (x @ W.T) >> s, per-row shift. x2d (N, in), W (out, in)."""
    return pow2norm(saturate(R.imatmul(x2d, W.T), cfg.accum_bits), TB + 2)


# Attention row-mass normalisation.
#
# A matched float mirror of this model -- same reversible coupling, same
# positional tables, same hinge objective, same data -- diverges without it:
# loss 4.8e10 at step 0 and still 8.1e4 at step 2000, greedy 0/128. Adding
# softmax alone took the same mirror to 128/128 exact greedy reproduction and
# 100% teacher-forced accuracy in 1000 steps. So what the integer path was
# missing is not the exponential; it is that nothing bounded the total mass of
# an attention row, and `pow2norm`'s max-based rescaling then re-quantised the
# relative importance of keys at every layer.
#
# This is the integer analogue: cap the row's L1 mass at 2**target_bits with a
# single power-of-two shift shared by the whole row, so the ordering of the
# weights inside a row is untouched. No exponential, no division, deterministic.
# The sum runs over the masked-in entries only -- a future key contributing to
# the shift would put the causality leak straight back.
ATTN_ROW_MASS_BITS = 0          # 0 disables; the pre-normalisation behaviour

# Resolution of the attention scores themselves. pow2norm has always brought
# them to TB+2 = 7 bits, which is 128 distinguishable levels for the whole row
# before the activation sees them. Softmax in the float mirror works at full
# precision, so this is a candidate for the same failure and is swept
# separately. 0 keeps the historical TB+2.
ATTN_SCORE_BITS = 0


def signed_shift(a, sh):
    """a >> sh where sh is positive, a << -sh where it is negative. Per row."""
    down = rshift0(a, torch.clamp(sh, min=0))
    up = a << torch.clamp(-sh, min=0, max=40)
    return torch.where(sh > 0, down, up)


def row_mass_norm(a, mask, target_bits, cfg=None):
    """
    Bring every row's L1 mass over its valid entries to about 2**target_bits.

    Capping alone does nothing here: the measured maximum row mass is ~600, so
    a 2**12 ceiling never fires. What softmax actually supplies is that *every*
    row carries the same total weight, so a row whose scores happen to be small
    still gets a full say. This equalises in both directions with one shift per
    row, which leaves the ordering inside the row untouched.
    """
    valid = mask.to(torch.bool)
    mass = torch.where(valid, a.abs(), torch.zeros_like(a)).sum(dim=-1, keepdim=True)
    sh = torch.where(mass > 0, R.bitlen_arr(mass) - target_bits,
                     torch.zeros_like(mass))
    out = signed_shift(a, sh) * mask
    if cfg is not None:
        out = saturate(out, cfg.accum_bits)
    return out, sh


def masked_pow2norm(x, mask, target_bits):
    """
    pow2norm whose per-row shift is taken over the masked-in entries only.

    pow2norm picks its shift from the row maximum. Applying the mask after it
    lets entries that are about to be discarded set the scale the surviving
    ones are divided by -- in decoder self-attention that is a future target
    key, in cross-attention a PAD source key. Either way an output depends on
    something the mask says it must not depend on, and in the decoder that is
    information teacher forcing has and autoregressive decoding does not.

    The reduction here runs over the valid entries alone; invalid ones are
    zeroed in the result so nothing outside the mask can survive. Shift
    semantics are untouched: same bit-length rule, same arithmetic `>>`, same
    clamp at zero. A row with no valid entry has a maximum of 0, hence shift 0
    and an all-zero output, which is the same answer pow2norm gives for an
    all-zero row.
    """
    valid = mask.to(torch.bool)
    mx = torch.where(valid, x.abs(), torch.zeros_like(x)).amax(dim=-1, keepdim=True)
    sh = torch.clamp(R.bitlen_arr(mx) - target_bits, min=0)
    return (x >> sh) * mask, sh


# -----------------------------------------------------------------------------
# Attention  (self or cross; kv_src may differ from the query source)
# -----------------------------------------------------------------------------
class AttnF:
    """F(a) = Wo . attn(Q(a), K(kv), V(kv)) with an integer score map."""

    kind = "attn"

    def __init__(self, d, cfg, gen, cross=False, d_kv=None):
        # Cross-attention reads the encoder memory, whose width is 2*d, so the
        # key/value projections take d_kv inputs while the query takes d.
        self.d, self.cfg, self.cross = d, cfg, cross
        self.dkv = d_kv if d_kv is not None else d
        self.Wq = init_w((d, d), cfg, gen)
        self.Wk = init_w((d, self.dkv), cfg, gen)
        self.Wv = init_w((d, self.dkv), cfg, gen)
        self.Wo = init_w((d, d), cfg, gen)

    def blocks(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo]

    def forward(self, a, ctx):
        cfg = self.cfg
        kv = ctx["memory"] if self.cross else a
        mask = ctx["cross_mask"] if self.cross else ctx["self_mask"]
        B, Tq, d = a.shape
        Tk = kv.shape[1]
        a2, k2 = a.reshape(B * Tq, d), kv.reshape(B * Tk, self.dkv)
        anchored = bool(self.cross and CROSS_QUERY_ANCHOR and Tq > 1)
        if anchored:
            # Exact: states are state_bits wide, so the difference needs one
            # more bit and cannot overflow the accumulator.
            aq = a - a[:, 0:1, :]
            aq[:, 0] = a[:, 0]
            a2 = aq.reshape(B * Tq, d)
        q, sq = lin_fwd(a2, self.Wq, cfg)
        k, sk = lin_fwd(k2, self.Wk, cfg)
        v, sv = lin_fwd(k2, self.Wv, cfg)
        Q, K, V = q.reshape(B, Tq, d), k.reshape(B, Tk, d), v.reshape(B, Tk, d)
        raw = saturate(R.imatmul(Q, K.transpose(1, 2)), cfg.accum_bits)
        S, ss = masked_pow2norm(raw, mask, ATTN_SCORE_BITS or (TB + 2))
        ss = ss.reshape(B * Tq, 1)
        if A.SCORE_EXP2 is not None:
            Aw = exp2_attend(S, mask)
        else:
            Aw = score_act(S) * mask
        if ATTN_ROW_MASS_BITS:
            Aw, sn = row_mass_norm(Aw, mask, ATTN_ROW_MASS_BITS, cfg)
        else:
            sn = None
        o2, so = pow2norm(saturate(R.imatmul(Aw, V), cfg.accum_bits).reshape(B * Tq, d), TB + 2)
        f2, sf = pow2norm(saturate(R.imatmul(o2, self.Wo.T), cfg.accum_bits), TB + 2)
        # a2 is the *query-side* input to Wq, anchored or not, which is what
        # the weight gradient needs.
        return f2.reshape(B, Tq, d), (a2, k2, Q, K, V, S, Aw, o2,
                                      sq, sk, sv, ss, so, sf, mask, B, Tq, Tk,
                                      anchored, sn)

    def backward(self, gf, cache, hb):
        """Returns (grad wrt query source, grad wrt kv source or None, weight grads)."""
        (a2, k2, Q, K, V, S, Aw, o2, sq, sk, sv, ss, so, sf,
         mask, B, Tq, Tk, anchored, sn) = cache
        cfg = self.cfg
        ab = cfg.accum_bits
        d = self.d

        g2 = SharedGrad(gf.m.reshape(B * Tq, d), gf.e)
        gO2, gWo = dense_linear_backward(g2, o2, self.Wo, sf, hb, 30, ab)
        gO = SharedGrad(gO2.m.reshape(B, Tq, d), gO2.e)

        gA, gV = bmm_backward_shared(gO, Aw, V, so, hb, ab)
        gAm = gA.m
        if sn is not None:
            # Forward scaled each row by 2**-sn, so the adjoint scales the
            # incoming gradient the same way.
            gAm = signed_shift(gAm, sn)
        if A.SCORE_EXP2 is not None:
            gS = SharedGrad(exp2_attend_backward(gAm, S, mask), gA.e)
        else:
            gS = SharedGrad(score_act_backward(gAm * mask, S), gA.e)
        gQ, gKt = bmm_backward_shared(gS, Q, K.transpose(1, 2), ss, hb, ab)
        gK = SharedGrad(gKt.m.transpose(1, 2), gKt.e)

        gq2, gWq = dense_linear_backward(
            SharedGrad(gQ.m.reshape(B * Tq, d), gQ.e), a2, self.Wq, sq, hb, 30, ab)
        gk2, gWk = dense_linear_backward(
            SharedGrad(gK.m.reshape(B * Tk, d), gK.e), k2, self.Wk, sk, hb, 30, ab)
        gv2, gWv = dense_linear_backward(
            SharedGrad(gV.m.reshape(B * Tk, d), gV.e), k2, self.Wv, sv, hb, 30, ab)

        gqm = gq2.m.reshape(B, Tq, d)
        gqe = gq2.e
        if anchored:
            # d/da of (a_t - a_0) for t>0 and a_0 for t=0: every tail row's
            # gradient also flows back into row 0, negated.
            gqm = gqm.clone()
            gqm[:, 0] = gqm[:, 0] - gqm[:, 1:].sum(dim=1)
            gqm, gqe = norm_block(gqm, gqe, hb)

        gws = [gWq, gWk, gWv, gWo]
        if self.cross:
            m, e = align_shared(gk2.m, gk2.e, gv2.m, gv2.e, hb)
            gkv = SharedGrad(m.reshape(B, Tk, self.dkv), e)
            return SharedGrad(gqm, gqe), gkv, gws
        # Self-attention: query and kv are the same tensor, so sum all three.
        m, e = align_shared(gqm.reshape(B * Tq, d), gqe, gk2.m, gk2.e, hb)
        m, e = align_shared(m, e, gv2.m, gv2.e, hb)
        return SharedGrad(m.reshape(B, Tq, d), e), None, gws


# -----------------------------------------------------------------------------
# Feed-forward
# -----------------------------------------------------------------------------
class FFNF:
    kind = "ffn"

    def __init__(self, d, hidden, cfg, gen):
        self.d, self.h, self.cfg = d, hidden, cfg
        self.W1 = init_w((hidden, d), cfg, gen)
        self.W2 = init_w((d, hidden), cfg, gen)

    def blocks(self):
        return [self.W1, self.W2]

    def forward(self, a, ctx):
        B, T, d = a.shape
        a2 = a.reshape(B * T, d)
        z, s1 = lin_fwd(a2, self.W1, self.cfg)
        hpos = torch.clamp_min(z, 0)
        f2, s2 = lin_fwd(hpos, self.W2, self.cfg)
        return f2.reshape(B, T, d), (a2, z, hpos, s1, s2, B, T)

    def backward(self, gf, cache, hb):
        a2, z, hpos, s1, s2, B, T = cache
        ab = self.cfg.accum_bits
        g2 = SharedGrad(gf.m.reshape(B * T, self.d), gf.e)
        gh, gW2 = dense_linear_backward(g2, hpos, self.W2, s2, hb, 30, ab)
        gz = SharedGrad(gh.m * (z > 0), gh.e)
        ga, gW1 = dense_linear_backward(gz, a2, self.W1, s1, hb, 30, ab)
        return SharedGrad(ga.m.reshape(B, T, self.d), ga.e), None, [gW1, gW2]


# -----------------------------------------------------------------------------
# Reversible stack of additive couplings
# -----------------------------------------------------------------------------
class RevStack:
    """[a,b] -> [a, b + F(a)] then swap halves. F is attn or ffn."""

    def __init__(self, d, hidden, cfg, gen, pattern):
        self.d, self.cfg = d, cfg
        self.fs = []
        for kind in pattern:
            if kind == "attn":
                self.fs.append(AttnF(d, cfg, gen, cross=False))
            elif kind == "cross":
                self.fs.append(AttnF(d, cfg, gen, cross=True, d_kv=2 * d))
            else:
                self.fs.append(FFNF(d, hidden, cfg, gen))

    def blocks(self):
        return [w for f in self.fs for w in f.blocks()]

    def block_offsets(self):
        """Start index of each block's weights inside blocks(); block sizes differ."""
        offs, run = [], 0
        for f in self.fs:
            offs.append(run)
            run += len(f.blocks())
        return offs

    def swap(self, v):
        return torch.cat([v[:, :, self.d:], v[:, :, :self.d]], dim=2)

    def forward(self, v, ctx):
        for f in self.fs:
            a, b = v[:, :, :self.d], v[:, :, self.d:]
            fa, _ = f.forward(a, ctx)
            v = self.swap(torch.cat([a, wrap_state(b + fa, self.cfg.state_bits)], dim=2))
        return v

    def forward_cached(self, v, ctx):
        tape = []
        for f in self.fs:
            a, b = v[:, :, :self.d], v[:, :, self.d:]
            fa, cache = f.forward(a, ctx)
            tape.append((a, cache))
            v = self.swap(torch.cat([a, wrap_state(b + fa, self.cfg.state_bits)], dim=2))
        return v, tape

    def inverse(self, v, ctx):
        for f in reversed(self.fs):
            v = self.swap(v)          # undo the swap (swap is its own inverse)
            a, bo = v[:, :, :self.d], v[:, :, self.d:]
            fa, _ = f.forward(a, ctx)
            v = torch.cat([a, wrap_state(bo - fa, self.cfg.state_bits)], dim=2)
        return v

    def backward(self, v_out, g_out, ctx, hb, tape=None, on_grad=None):
        """
        Cache-free by default: each layer input is recovered by exact inversion.
        Returns (grad wrt stack input, grad accumulated onto ctx['memory'], grads).
        """
        v, g = v_out, g_out
        gmem = None
        collected = []
        for idx in range(len(self.fs) - 1, -1, -1):
            f = self.fs[idx]
            v = self.swap(v)
            g = SharedGrad(self.swap(g.m), g.e)
            a = v[:, :, :self.d]
            if tape is None:
                fa, cache = f.forward(a, ctx)
                vin = torch.cat([a, wrap_state(v[:, :, self.d:] - fa,
                                               self.cfg.state_bits)], dim=2)
            else:
                a, cache = tape[idx]
                vin = None
            ga = SharedGrad(g.m[:, :, :self.d], g.e)
            gb = SharedGrad(g.m[:, :, self.d:], g.e)
            gfa, gkv, gws = f.backward(gb, cache, hb)
            m, e = align_shared(ga.m, ga.e, gfa.m, gfa.e, hb)
            g = SharedGrad(torch.cat([m, gb.m], dim=2), e) if e == gb.e else \
                self._join(m, e, gb, hb)
            if gkv is not None:
                gmem = gkv if gmem is None else self._acc(gmem, gkv, hb)
            if on_grad is None:
                collected.append((idx, gws))
            else:
                on_grad(idx, gws)
            if tape is None:
                v = vin
        if on_grad is None:
            collected.sort(key=lambda t: t[0])
            return g, gmem, [w for _i, gws in collected for w in gws]
        return g, gmem, None

    def _join(self, m, e, gb, hb):
        E = max(int(e), int(gb.e))
        x = rshift0(m, E - int(e))
        y = rshift0(gb.m, E - int(gb.e))
        mm, ee = norm_block(torch.cat([x, y], dim=2), E, hb)
        return SharedGrad(mm, ee)

    @staticmethod
    def _acc(g1, g2, hb):
        m, e = align_shared(g1.m, g1.e, g2.m, g2.e, hb)
        return SharedGrad(m, e)


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------
class JaKoMT:
    LETTER = {"a": "attn", "c": "cross", "f": "ffn"}

    def __init__(self, vocab, d=64, hidden=128, k_enc=4, k_dec=6, cfg=None, seed=0,
                 enc_pattern="", dec_pattern="", positional=0):
        cfg = cfg or R.RunConfig(state_bits=16, weight_bits=8, accum_bits=32)
        self.cfg, self.vocab, self.d = cfg, vocab, d
        gen = torch.Generator().manual_seed(seed)
        self.emb_src = init_w((vocab, d), cfg, gen)
        self.emb_tgt = init_w((vocab, d), cfg, gen)
        if enc_pattern:
            enc_pat = [self.LETTER[c] for c in enc_pattern]
        else:
            enc_pat = ["attn" if i % 2 == 0 else "ffn" for i in range(k_enc)]
        if dec_pattern:
            dec_pat = [self.LETTER[c] for c in dec_pattern]
        else:
            dec_pat = [("attn", "cross", "ffn")[i % 3] for i in range(k_dec)]
        self.enc = RevStack(d, hidden, cfg, gen, enc_pat)
        self.dec = RevStack(d, hidden, cfg, gen, dec_pat)
        self.Wout = init_w((vocab, 2 * d), cfg, gen)
        # Learned integer positional tables, appended AFTER Wout so that every
        # existing optimizer index keeps its meaning and old checkpoints still
        # load. positional=0 creates nothing at all, so a run without them is
        # bit-identical to the model before this option existed.
        self.max_pos = int(positional)
        if self.max_pos:
            self.pos_src = init_w((self.max_pos, d), cfg, gen)
            self.pos_tgt = init_w((self.max_pos, d), cfg, gen)
        else:
            self.pos_src = self.pos_tgt = None
        self.enc_pattern, self.dec_pattern = enc_pat, dec_pat

    # -- parameter ordering: embeddings, encoder, decoder, output head --------
    def parameters(self):
        ps = ([self.emb_src, self.emb_tgt] + self.enc.blocks()
              + self.dec.blocks() + [self.Wout])
        if self.max_pos:
            ps += [self.pos_src, self.pos_tgt]
        return ps

    def n_params(self):
        return int(sum(w.numel() for w in self.parameters()))

    def embed(self, ids, table, pos=None):
        e = table[ids]                                   # (B,T,d)
        if pos is not None:
            T = ids.shape[1]
            if T > pos.shape[0]:
                raise ValueError(
                    "sequence of %d exceeds the positional table (%d rows); "
                    "--positional must be at least --max-len" % (T, pos.shape[0]))
            # Same two's-complement wrap the reversible state uses, so the sum
            # stays inside state_bits and stays exactly invertible.
            e = wrap_state(e + pos[:T][None, :, :], self.cfg.state_bits)
        return torch.cat([e, torch.zeros_like(e)], dim=2)

    def masks(self, src_ids, tgt_ids):
        B, Ts = src_ids.shape
        Tt = tgt_ids.shape[1]
        src_ok = (src_ids != PAD).to(torch.int64)         # (B,Ts)
        enc_self = src_ok[:, None, :].expand(B, Ts, Ts)
        cross = src_ok[:, None, :].expand(B, Tt, Ts)
        causal = torch.tril(torch.ones(Tt, Tt, dtype=torch.int64))[None].expand(B, Tt, Tt)
        return enc_self.contiguous(), causal.contiguous(), cross.contiguous()

    def encode(self, src_ids, cached=False):
        enc_self, _c, _x = self.masks(src_ids, src_ids)
        ctx = {"self_mask": enc_self, "cross_mask": None, "memory": None}
        v = self.embed(src_ids, self.emb_src, self.pos_src)
        if cached:
            out, tape = self.enc.forward_cached(v, ctx)
            return out, ctx, v, tape
        return self.enc.forward(v, ctx), ctx, v, None

    def decode(self, tgt_in, memory, src_ids, cached=False):
        _e, causal, cross = self.masks(src_ids, tgt_in)
        ctx = {"self_mask": causal, "cross_mask": cross, "memory": memory}
        v = self.embed(tgt_in, self.emb_tgt, self.pos_tgt)
        if cached:
            out, tape = self.dec.forward_cached(v, ctx)
            return out, ctx, v, tape
        return self.dec.forward(v, ctx), ctx, v, None

    def logits(self, dec_state):
        B, T, W = dec_state.shape
        z, s = lin_fwd(dec_state.reshape(B * T, W), self.Wout, self.cfg)
        return z.reshape(B, T, self.vocab), s



# -----------------------------------------------------------------------------
# Chunked output head
# -----------------------------------------------------------------------------
# Materialising the whole B x T x V logit tensor, and then a second one for its
# gradient, is the largest allocation in a training step: 41 MB each at
# batch 32 / V 5000. The hinge needs only three numbers per position (the true
# logit, the largest wrong logit and its index), so the vocabulary can be walked
# in chunks with a running max instead.
#
# Two passes are required, not one. pow2norm picks its shift from the row max,
# so a single pass would put each chunk on its own scale; rescaling afterwards
# is exact for magnitudes but not for tie-breaking, because two logits that
# compare unequal at a chunk's scale can compare equal at the final one and the
# argmax would land on a different index. Pass one therefore takes the global
# row max, pass two applies the one shift the full head would have chosen.
#
# The backward needs no chunking at all: the hinge gradient has at most two
# nonzeros per row, both +-1, so both products against it collapse to gathers
# and scatter-adds. That removes the output-head backward GEMM outright.

def _out_chunks(model, flat, chunk):
    """Saturated accumulators for one vocabulary slice at a time."""
    for lo in range(0, model.vocab, chunk):
        hi = min(lo + chunk, model.vocab)
        yield lo, hi, saturate(R.imatmul(flat, model.Wout[lo:hi].T),
                               model.cfg.accum_bits)


def chunked_head_forward(model, dec_state, tgt_out, hb, chunk=512):
    """
    Hinge statistics over the full vocabulary without ever holding it.

    Returns (true_val, best_val, best_idx, shift, stats) where shift is exactly
    the per-row shift lin_fwd would have produced for the whole row.
    """
    B, T, W = dec_state.shape
    flat = dec_state.reshape(B * T, W)
    tgt = tgt_out.reshape(-1)
    n = flat.shape[0]
    NEG = torch.iinfo(torch.int64).min // 4
    notpad = tgt != PAD

    mx = torch.zeros((n, 1), dtype=torch.int64)
    for _lo, _hi, acc in _out_chunks(model, flat, chunk):
        mx = torch.maximum(mx, acc.abs().amax(dim=1, keepdim=True))
    shift = torch.clamp(R.bitlen_arr(mx) - (TB + 2), min=0)

    true_val = torch.zeros(n, dtype=torch.int64)
    best_val = torch.full((n,), NEG, dtype=torch.int64)
    best_idx = torch.zeros(n, dtype=torch.int64)
    top_val = torch.full((n,), NEG, dtype=torch.int64)
    top_idx = torch.zeros(n, dtype=torch.int64)
    ssum = ssq = 0.0
    cnt = 0
    for lo, hi, acc in _out_chunks(model, flat, chunk):
        z = acc >> shift
        zp = z[notpad].double()
        ssum += float(zp.sum())
        ssq += float((zp * zp).sum())
        cnt += zp.numel()
        cm = z.max(dim=1)
        upd = cm.values > top_val
        top_idx = torch.where(upd, cm.indices + lo, top_idx)
        top_val = torch.where(upd, cm.values, top_val)
        in_chunk = (tgt >= lo) & (tgt < hi)
        if bool(in_chunk.any()):
            ii = torch.nonzero(in_chunk).reshape(-1)
            true_val[ii] = z[ii, tgt[ii] - lo]
            z[ii, tgt[ii] - lo] = NEG
        cw = z.max(dim=1)
        upd = cw.values > best_val
        best_idx = torch.where(upd, cw.indices + lo, best_idx)
        best_val = torch.where(upd, cw.values, best_val)

    active = (best_val + 1 > true_val) & notpad
    mean = ssum / max(1, cnt)
    var = max(0.0, (ssq - cnt * mean * mean) / max(1, cnt - 1))
    stats = {
        "tokens": int(notpad.sum().item()),
        "active": int(active.sum().item()),
        "hinge_sum": int(torch.clamp_min((best_val + 1 - true_val)[notpad], 0).sum().item()),
        "correct": int(((top_idx == tgt) & notpad).sum().item()),
        "margin_mean": float((true_val - best_val)[notpad].double().mean().item()),
        "logit_std": var ** 0.5,
    }
    return true_val, best_val, best_idx, shift, stats


def sparse_head_backward(model, dec_state, tgt_out, true_val, best_val, best_idx,
                         shift, hb):
    """
    dense_linear_backward specialised to the hinge gradient.

    The hinge gradient is +1 at the best wrong class, -1 at the target and zero
    everywhere else, so g @ Wout is a gather of two rows and g.T @ x is a
    scatter-add of the state rows. Both are bit-identical to the GEMMs they
    replace: integer addition is exact, so accumulation order does not matter.

    norm_block does not leave the +-1 gradient alone -- with a one-bit block and
    `bits` of requested precision it shifts *left* by bits-1. The scale it picks
    is taken from norm_block itself here rather than re-derived, because the
    whole point of this path is to match the dense one exactly.
    """
    cfg = model.cfg
    B, T, W2 = dec_state.shape
    x = dec_state.reshape(B * T, W2)
    tgt = tgt_out.reshape(-1)
    n = x.shape[0]
    active = (best_val + 1 > true_val) & (tgt != PAD)
    ii = torch.nonzero(active).reshape(-1)

    if ii.numel():
        pm, e0 = norm_block(torch.ones(1, dtype=torch.int64), 0, hb)
        scale = int(pm[0].item())
    else:
        scale, e0 = 1, 0

    sm = shift.reshape(-1).to(torch.int64)
    sample_e = e0 - sm
    E = int(sample_e.max().item()) if sample_e.numel() else int(e0)
    align = (E - sample_e).reshape(-1, 1)

    gx = torch.zeros((n, W2), dtype=torch.int64)
    if ii.numel():
        gx[ii] = scale * (model.Wout[best_idx[ii]] - model.Wout[tgt[ii]])
    gx = saturate(rshift0(gx, align), cfg.accum_bits)
    gxm, gxe = norm_block(gx, E, hb)

    # rshift0(+-scale, align) is what actually reaches the weight gradient, so
    # rows carry a per-row coefficient that falls to zero once align exceeds the
    # scale's bit length.
    gw = torch.zeros_like(model.Wout)
    if ii.numel():
        coef = rshift0(torch.full((n,), scale, dtype=torch.int64),
                       align.reshape(-1)).reshape(-1, 1)
        sel = ii[coef.reshape(-1)[ii] != 0]
        if sel.numel():
            cx = coef[sel] * x[sel]
            gw.index_add_(0, best_idx[sel], cx)
            gw.index_add_(0, tgt[sel], -cx)
    gw = saturate(gw, cfg.accum_bits)
    gwm, gwe = norm_block(gw, E, 30)
    return SharedGrad(gxm, gxe), (gwm, gwe)


def chunked_head(model, dec_state, tgt_out, hb, chunk=512):
    """Forward statistics and backward gradients for the output layer, with no
    B x T x V tensor anywhere. Drop-in for logits + hinge_grad +
    dense_linear_backward."""
    tv, bv, bi, sh, stats = chunked_head_forward(model, dec_state, tgt_out, hb, chunk)
    g_dec_state, gWout = sparse_head_backward(
        model, dec_state, tgt_out, tv, bv, bi, sh, hb)
    return g_dec_state, gWout, stats


# -----------------------------------------------------------------------------
# Token-level integer hinge
# -----------------------------------------------------------------------------
def hinge_grad(logits, tgt_out, hb):
    """Per-token multiclass margin. PAD targets contribute nothing."""
    B, T, V = logits.shape
    flat = logits.reshape(B * T, V)
    tgt = tgt_out.reshape(B * T)
    n = flat.shape[0]
    rows = torch.arange(n)
    masked = flat.clone()
    masked[rows, tgt] = torch.iinfo(torch.int64).min // 4
    j = masked.argmax(dim=1)
    notpad = tgt != PAD
    true_l = flat[rows, tgt]
    best_w = flat[rows, j]
    active = (best_w + 1 > true_l) & notpad
    g = torch.zeros_like(flat)
    ii = rows[active]
    g[ii, j[active]] += 1
    g[ii, tgt[active]] -= 1
    gm, ge = norm_block(g, 0, hb)
    stats = {
        "tokens": int(notpad.sum().item()),
        "active": int(active.sum().item()),
        "hinge_sum": int(torch.clamp_min((best_w + 1 - true_l)[notpad], 0).sum().item()),
        "correct": int(((flat.argmax(1) == tgt) & notpad).sum().item()),
        "margin_mean": float((true_l - best_w)[notpad].double().mean().item()),
        "logit_std": float(flat[notpad].double().std().item()),
    }
    return SharedGrad(gm.reshape(B, T, V), ge), stats


def pos_grad(g, max_pos):
    """Adjoint of adding one row per position: sum the batch at each position."""
    B, T, d = g.m.shape
    out = torch.zeros((max_pos, d), dtype=torch.int64)
    out[:T] = g.m.sum(dim=0)
    return norm_block(out, g.e, 30)


def embed_grad(g, ids, vocab, hb):
    """Adjoint of a table lookup: scatter-add rows. g is over the first half."""
    B, T, d = g.m.shape
    flat = g.m.reshape(B * T, d)
    out = torch.zeros((vocab, d), dtype=torch.int64)
    out.index_add_(0, ids.reshape(-1), flat)
    return norm_block(out, g.e, 30)


# -----------------------------------------------------------------------------
# One training step: full manual backward, cache-free through both stacks
# -----------------------------------------------------------------------------
def param_index(model):
    """Global optimizer index of every parameter, in parameters() order."""
    idx = {"emb_src": 0, "emb_tgt": 1}
    off = 2
    idx["enc"] = off
    off += len(model.enc.blocks())
    idx["dec"] = off
    off += len(model.dec.blocks())
    idx["wout"] = off
    off += 1
    if getattr(model, "max_pos", 0):
        idx["pos_src"] = off
        idx["pos_tgt"] = off + 1
    return idx


def forward_backward(model, src, tin, tout, hb, cached=False, opt=None,
                     telemetry=None, chunk_head=0):
    """
    cached=False  -> reversible stacks reconstruct their activations (the method)
    cached=True   -> stacks keep a tape (control, for Gate 3)
    opt is not None -> per-layer stepping; otherwise gradients are returned.
    chunk_head > 0 -> output layer walks the vocabulary in chunks of that size
                      and takes the sparse hinge backward; bit-identical.
    """
    cfg = model.cfg
    idx = param_index(model)
    grads = {} if opt is None else None

    def emit(i, pair):
        if opt is None:
            grads[i] = pair
        else:
            opt._step_one(i, pair[0])

    mem, enc_ctx, enc_in, enc_tape = model.encode(src, cached=cached)
    dec_out, dec_ctx, dec_in, dec_tape = model.decode(tin, mem, src, cached=cached)

    B, T = tout.shape
    W2 = 2 * model.d
    if chunk_head:
        g_dec_state, gWout, stats = chunked_head(model, dec_out, tout, hb, chunk_head)
    else:
        logits, s_out = model.logits(dec_out)
        g_log, stats = hinge_grad(logits, tout, hb)
        V = logits.shape[2]
        g2 = SharedGrad(g_log.m.reshape(B * T, V), g_log.e)
        g_dec_state, gWout = dense_linear_backward(
            g2, dec_out.reshape(B * T, W2), model.Wout, s_out, hb, 30, cfg.accum_bits)
    emit(idx["wout"], gWout)
    g_dec = SharedGrad(g_dec_state.m.reshape(B, T, W2), g_dec_state.e)

    dec_off = model.dec.block_offsets()
    on = None if opt is None else (
        lambda i, gws: [opt._step_one(idx["dec"] + dec_off[i] + c, g[0])
                        for c, g in enumerate(gws)])
    g_dec_in, g_mem, dec_grads = model.dec.backward(
        dec_out, g_dec, dec_ctx, hb, tape=dec_tape, on_grad=on)
    if dec_grads is not None:
        for c, g in enumerate(dec_grads):
            emit(idx["dec"] + c, g)

    g_tgt_emb = SharedGrad(g_dec_in.m[:, :, :model.d], g_dec_in.e)
    emit(idx["emb_tgt"], embed_grad(g_tgt_emb, tin, model.vocab, hb))
    if model.max_pos:
        emit(idx["pos_tgt"], pos_grad(g_tgt_emb, model.max_pos))

    if g_mem is None:
        g_mem = SharedGrad(torch.zeros_like(mem), 0)
    enc_off = model.enc.block_offsets()
    on_e = None if opt is None else (
        lambda i, gws: [opt._step_one(idx["enc"] + enc_off[i] + c, g[0])
                        for c, g in enumerate(gws)])
    g_enc_in, _none, enc_grads = model.enc.backward(
        mem, g_mem, enc_ctx, hb, tape=enc_tape, on_grad=on_e)
    if enc_grads is not None:
        for c, g in enumerate(enc_grads):
            emit(idx["enc"] + c, g)

    g_src_emb = SharedGrad(g_enc_in.m[:, :, :model.d], g_enc_in.e)
    emit(idx["emb_src"], embed_grad(g_src_emb, src, model.vocab, hb))
    if model.max_pos:
        emit(idx["pos_src"], pos_grad(g_src_emb, model.max_pos))

    if telemetry is not None:
        telemetry["attn_grad_nonzero"] = attn_nonzero(model, grads, idx)
    if opt is None:
        return [grads[i] for i in range(len(model.parameters()))], stats
    return None, stats


def attn_nonzero(model, grads, idx):
    """Per-projection gradient nonzero ratio for every attention block."""
    if grads is None:
        return {}
    out = {}
    for tag, stack, base in (("enc", model.enc, idx["enc"]), ("dec", model.dec, idx["dec"])):
        off = base
        for bi, f in enumerate(stack.fs):
            names = ["Wq", "Wk", "Wv", "Wo"] if f.kind == "attn" else ["W1", "W2"]
            for c, nm in enumerate(names):
                g = grads.get(off + c)
                if g is not None and f.kind == "attn":
                    out[f"{tag}{bi}_{nm}"] = round(
                        float((g[0] != 0).double().mean().item()) * 100, 3)
            off += len(names)
    return out


# -----------------------------------------------------------------------------
# Greedy decoding
# -----------------------------------------------------------------------------
def greedy(model, src_ids, max_len):
    B = src_ids.shape[0]
    mem, _c, _i, _t = model.encode(src_ids)
    out = torch.full((B, max_len), PAD, dtype=torch.int64)
    cur = torch.full((B, max_len), PAD, dtype=torch.int64)
    cur[:, 0] = BOS
    done = torch.zeros(B, dtype=torch.bool)
    for t in range(max_len - 1):
        dec, _c2, _i2, _t2 = model.decode(cur, mem, src_ids)
        logits, _s = model.logits(dec)
        nxt = logits[:, t].argmax(dim=1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        out[:, t] = nxt
        done = done | (nxt == EOS)
        if bool(done.all()):
            break
        cur[:, t + 1] = nxt
    return out


def ids_to_text(tok, ids):
    keep = []
    for i in ids.tolist():
        if i in (EOS, PAD):
            break
        if i != BOS:
            keep.append(i)
    return tok.decode(keep)


# -----------------------------------------------------------------------------
# Metrics (no sacrebleu here, so chrF and BLEU are implemented directly)
# -----------------------------------------------------------------------------
def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def chrf(hyps, refs, n=6, beta=2.0):
    from collections import Counter
    tp = [0.0] * n; hl = [0.0] * n; rl = [0.0] * n
    for h, r in zip(hyps, refs):
        hc, rc = h.replace(" ", ""), r.replace(" ", "")
        for k in range(1, n + 1):
            hn, rn = Counter(ngrams(hc, k)), Counter(ngrams(rc, k))
            tp[k - 1] += sum((hn & rn).values())
            hl[k - 1] += sum(hn.values()); rl[k - 1] += sum(rn.values())
    ps = [tp[i] / hl[i] for i in range(n) if hl[i] > 0]
    rs = [tp[i] / rl[i] for i in range(n) if rl[i] > 0]
    if not ps or not rs:
        return 0.0
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def bleu(hyps, refs, n=4):
    from collections import Counter
    tp = [0] * n; tot = [0] * n; hlen = rlen = 0
    for h, r in zip(hyps, refs):
        ht, rt = list(h.replace(" ", "")), list(r.replace(" ", ""))
        hlen += len(ht); rlen += len(rt)
        for k in range(1, n + 1):
            hn, rn = Counter(ngrams(ht, k)), Counter(ngrams(rt, k))
            tp[k - 1] += sum((hn & rn).values()); tot[k - 1] += max(0, len(ht) - k + 1)
    if min(tot) == 0:
        return 0.0
    precs = [(tp[i] + 1.0) / (tot[i] + 1.0) for i in range(n)]      # add-1 smoothing
    logp = sum(math.log(p) for p in precs) / n
    bp = 1.0 if hlen > rlen else math.exp(1 - rlen / max(1, hlen))
    return 100.0 * bp * math.exp(logp)


# -----------------------------------------------------------------------------
# Evaluation and telemetry
# -----------------------------------------------------------------------------
def proc_mem_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2 ** 20, 1)
    except Exception:
        return None


def evaluate(model, data, hb, batch, max_eval=None, decode=False, tok=None, pairs=None):
    src_all, tin_all, tout_all = data
    n = len(src_all) if max_eval is None else min(len(src_all), max_eval)
    tot = dict(tokens=0, active=0, hinge_sum=0, correct=0)
    logit_stds, margins = [], []
    for st in range(0, n, batch):
        src = torch.tensor(src_all[st:st + batch], dtype=torch.int64)
        tin = torch.tensor(tin_all[st:st + batch], dtype=torch.int64)
        tout = torch.tensor(tout_all[st:st + batch], dtype=torch.int64)
        mem, _c, _i, _t = model.encode(src)
        dec, _c2, _i2, _t2 = model.decode(tin, mem, src)
        logits, _s = model.logits(dec)
        _g, st_ = hinge_grad(logits, tout, hb)
        for k in tot:
            tot[k] += st_[k]
        logit_stds.append(st_["logit_std"])
        margins.append(st_["margin_mean"])
    out = {
        "tokens": tot["tokens"],
        "hinge_per_token": round(tot["hinge_sum"] / max(1, tot["tokens"]), 4),
        "token_acc": round(100.0 * tot["correct"] / max(1, tot["tokens"]), 3),
        "margin_active_frac": round(tot["active"] / max(1, tot["tokens"]), 4),
        "logit_std": round(sum(logit_stds) / max(1, len(logit_stds)), 3),
        "margin_true_minus_best_wrong": round(sum(margins) / max(1, len(margins)), 3),
    }
    if decode and tok is not None and pairs is not None:
        hyps, refs, exact = [], [], 0
        m = min(len(pairs), n)
        for st in range(0, m, batch):
            src = torch.tensor(src_all[st:st + batch], dtype=torch.int64)
            gen = greedy(model, src, len(src_all[0]))
            for bi in range(gen.shape[0]):
                if st + bi >= m:
                    break
                h = ids_to_text(tok, gen[bi])
                r = pairs[st + bi][1]
                hyps.append(h)
                refs.append(r)
                exact += int(h.strip() == r.strip())
        out["chrf"] = round(chrf(hyps, refs), 3)
        out["bleu"] = round(bleu(hyps, refs), 3)
        out["seq_exact_match"] = round(100.0 * exact / max(1, len(hyps)), 3)
        out["samples"] = [[pairs[i][0], refs[i], hyps[i]] for i in range(len(hyps))]
    return out


def activation_report(model, src, tin, tout, hb):
    """Peak distinct activation bytes: reversible stacks vs their cached tape."""
    from reversible_integer_memory_bench import dedup_bytes
    mem, ectx, enc_in, _t = model.encode(src, cached=False)
    dec_out, dctx, dec_in, _t2 = model.decode(tin, mem, src, cached=False)

    def cached_bytes(stack, v, ctx):
        _o, tape = stack.forward_cached(v, ctx)
        ts = []
        for a, cache in tape:
            ts.append(a)
            ts += [x for x in cache if isinstance(x, torch.Tensor)]
        return dedup_bytes(ts)

    def free_peak(stack, v_out, ctx):
        peak = 0
        v = v_out
        for i in range(len(stack.fs) - 1, -1, -1):
            v = stack.swap(v)
            a = v[:, :, :stack.d]
            fa, cache = stack.fs[i].forward(a, ctx)
            live = [v, a, fa] + [x for x in cache if isinstance(x, torch.Tensor)]
            peak = max(peak, dedup_bytes(live))
            v = torch.cat([a, wrap_state(v[:, :, stack.d:] - fa,
                                         stack.cfg.state_bits)], dim=2)
        return peak

    return {
        "encoder_cached_activation_bytes": cached_bytes(model.enc, enc_in, ectx),
        "encoder_cachefree_peak_bytes": free_peak(model.enc, mem, ectx),
        "decoder_cached_activation_bytes": cached_bytes(model.dec, dec_in, dctx),
        "decoder_cachefree_peak_bytes": free_peak(model.dec, dec_out, dctx),
        "parameter_bytes": int(sum(w.numel() * 8 for w in model.parameters())),
        "encoder_memory_bytes": int(mem.numel() * 8),
        "token_buffer_bytes": int((src.numel() + tin.numel() + tout.numel()) * 8),
    }



# -----------------------------------------------------------------------------
# Checkpoints and interactive translation
# -----------------------------------------------------------------------------
def save_ckpt(path, model, args, tok_path, opt=None, epoch=-1):
    torch.save({"weights": [w.clone() for w in model.parameters()],
                "opt_acc": ([a.clone() for a in opt.acc] if opt is not None else None),
                "epoch": epoch, "lr_bits": getattr(args, "lr_bits", 2),
                "hidden_bits": getattr(args, "hidden_bits", 16),
                "vocab_size_arg": getattr(args, "vocab_size", 0),
                "tok_mode": getattr(args, "tok_mode", "char"),
                "score_leak": getattr(args, "score_leak", None),
                "score_exp2": getattr(args, "score_exp2", None),
                "score_exp2_cap": getattr(args, "score_exp2_cap", 12),
                "exp2_coupled": not getattr(args, "exp2_diag", False),
                "momentum_bits": getattr(args, "momentum_bits", 0),
                "query_anchor": getattr(args, "query_anchor", False),
                "prefix_corrupt": getattr(args, "prefix_corrupt", 0),
                "scheduled_sampling": getattr(args, "scheduled_sampling", 0),
                "positional": getattr(args, "positional", 0),
                "attn_mass_bits": getattr(args, "attn_mass_bits", 0),
                "attn_score_bits": getattr(args, "attn_score_bits", 0),
                "max_tok": getattr(args, "max_tok", 0),
                "corpus": getattr(args, "corpus", ""),
                "vocab": model.vocab, "dim": model.d,
                "ffn_hidden": args.ffn_hidden, "k_enc": args.k_enc,
                "k_dec": args.k_dec, "max_len": args.max_len,
                "enc_pattern": getattr(args, "enc_pattern", ""),
                "dec_pattern": getattr(args, "dec_pattern", ""),
                "state_bits": args.state_bits, "weight_bits": args.weight_bits,
                "accum_bits": args.accum_bits, "tokenizer_path": tok_path}, path)


def load_ckpt(path):
    ck = torch.load(path, weights_only=False)
    cfg = R.RunConfig(state_bits=ck["state_bits"], weight_bits=ck["weight_bits"],
                      accum_bits=ck["accum_bits"])
    R.CFG = cfg
    model = JaKoMT(ck["vocab"], ck["dim"], ck["ffn_hidden"],
                   ck["k_enc"], ck["k_dec"], cfg, 0,
                   enc_pattern=ck.get("enc_pattern", ""),
                   dec_pattern=ck.get("dec_pattern", ""),
                   positional=ck.get("positional", 0))
    for dst, src in zip(model.parameters(), ck["weights"]):
        dst.copy_(src)
    A.SCORE_LEAK = ck.get("score_leak", None)
    A.SCORE_EXP2 = ck.get("score_exp2", None)
    A.SCORE_EXP2_CAP = ck.get("score_exp2_cap", 12)
    A.SCORE_EXP2_COUPLED = bool(ck.get("exp2_coupled", True))
    global ATTN_ROW_MASS_BITS
    ATTN_ROW_MASS_BITS = int(ck.get("attn_mass_bits", 0))
    global ATTN_SCORE_BITS
    ATTN_SCORE_BITS = int(ck.get("attn_score_bits", 0))
    global CROSS_QUERY_ANCHOR
    CROSS_QUERY_ANCHOR = bool(ck.get("query_anchor", False))
    tok = D.load_tokenizer(ck["tokenizer_path"])
    return model, tok, ck


def translate(model, tok, sentences, max_len):
    ids = []
    for s in sentences:
        e = tok.encode(s).ids[:max_len - 1] + [EOS]
        ids.append(e + [PAD] * (max_len - len(e)))
    src = torch.tensor(ids, dtype=torch.int64)
    out = greedy(model, src, max_len)
    return [ids_to_text(tok, out[i]) for i in range(len(sentences))]

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(args):
    torch.set_num_threads(args.threads)
    A.SCORE_LEAK = getattr(args, 'score_leak', None)
    A.SCORE_EXP2 = getattr(args, 'score_exp2', None)
    A.SCORE_EXP2_CAP = getattr(args, 'score_exp2_cap', 12)
    A.SCORE_EXP2_COUPLED = not getattr(args, 'exp2_diag', False)
    global CROSS_QUERY_ANCHOR, ATTN_ROW_MASS_BITS
    CROSS_QUERY_ANCHOR = bool(getattr(args, "query_anchor", False))
    ATTN_ROW_MASS_BITS = int(getattr(args, "attn_mass_bits", 0))
    global ATTN_SCORE_BITS
    ATTN_SCORE_BITS = int(getattr(args, "attn_score_bits", 0))
    cfg = R.RunConfig(state_bits=args.state_bits, weight_bits=args.weight_bits,
                      accum_bits=args.accum_bits)
    R.CFG = cfg
    ds = D.prepare(args.corpus, args.max_pairs, args.vocab_size, args.max_len,
                   valid_frac=args.valid_frac, max_tok=getattr(args, 'max_tok', 0),
                   tok_mode=getattr(args, 'tok_mode', 'char'),
                   subs_pairs=getattr(args, 'subs_pairs', 25000),
                   subs_pool=getattr(args, 'subs_pool', 0),
                   base_pairs=getattr(args, 'base_pairs', 25000),
                   base_pool=getattr(args, 'base_pool', 0),
                   min_char_count=getattr(args, 'min_char_count', 1))
    tok = ds["tokenizer"]
    vocab = ds["info"]["vocab_size"]
    model = JaKoMT(vocab, args.dim, args.ffn_hidden, args.k_enc, args.k_dec,
                   cfg, args.seed, enc_pattern=getattr(args, 'enc_pattern', ''),
                   dec_pattern=getattr(args, 'dec_pattern', ''),
                   positional=getattr(args, 'positional', 0))
    opt = BlockSGD(model.parameters(), param_bits=12, lr_bits=args.lr_bits, clip=1,
                   momentum_bits=getattr(args, 'momentum_bits', 0))
    hb = args.hidden_bits

    try:
        import reversible_jako_metrics as MET
        klass = MET.build_classifier(tok, ds["valid"][2])
    except Exception as e:                       # never let metrics stop a run
        print("metrics unavailable: %s: %s" % (type(e).__name__, e), flush=True)
        MET, klass = None, None

    src_all, tin_all, tout_all = ds["train"]
    ntr = len(src_all)
    print(json.dumps({"corpus": ds["info"]["corpus"], "train_pairs": ntr,
                      "train_by_corpus": ds["info"].get("train_by_corpus", {}),
                      "valid_by_corpus": ds["info"].get("valid_by_corpus", {}),
                      "valid_pairs": len(ds["valid"][0]), "vocab": vocab,
                      "params": model.n_params(), "dim": args.dim,
                      "k_enc": args.k_enc, "k_dec": args.k_dec,
                      "state_bits": args.state_bits, "weight_bits": args.weight_bits,
                      "accum_bits": args.accum_bits, "hidden_bits": hb,
                      "lr_bits": args.lr_bits, "optimizer_stable": opt.stable,
                      "train_prefix_corruption": getattr(args, "prefix_corrupt", 0) / 100.0,
                      "train_scheduled_sampling": getattr(args, "scheduled_sampling", 0) / 100.0,
                      "eval_prefix_corruption": 0.0,
                      "query_anchor": getattr(args, "query_anchor", False),
                      "positional_encoding": getattr(args, "positional", 0),
                      "attn_row_mass_bits": getattr(args, "attn_mass_bits", 0),
                      "score_exp2": getattr(args, "score_exp2", None),
                      "exp2_coupled": not getattr(args, "exp2_diag", False),
                "momentum_bits": getattr(args, "momentum_bits", 0),
                      "threads": args.threads}, ensure_ascii=False), flush=True)

    before = None
    if args.show_samples:
        before = evaluate(model, ds["valid"], hb, args.batch, args.show_samples,
                          decode=True, tok=tok, pairs=ds["valid_pairs"])

    if getattr(args, "resume", ""):
        rc = torch.load(args.resume, weights_only=False)
        for dst, srcw in zip(model.parameters(), rc["weights"]):
            dst.copy_(srcw)
        if rc.get("opt_acc") is not None:
            for dst, srca in zip(opt.acc, rc["opt_acc"]):
                dst.copy_(srca)
        print("resumed from " + args.resume + " (epoch " + str(rc.get("epoch")) + ")",
              flush=True)
    # Views of the validation set restricted to one source corpus each, so the
    # mixed run can tell "more data helped everywhere" from "domain match
    # helped only the subtitles".
    vtags = ds.get("valid_tags") or []
    vsplit = {}
    if vtags and len(set(vtags)) > 1:
        for tag in sorted(set(vtags)):
            idx = [i for i, t in enumerate(vtags) if t == tag]
            vsplit[tag] = tuple([[col[i] for i in idx] for col in ds["valid"]])

    init_snapshot = [w.clone() for w in model.parameters()]
    best_acc = [-1.0, float('inf')]
    bad = 0
    rng = np.random.default_rng(args.seed)
    corrupt_gen = torch.Generator().manual_seed(args.seed + 7919)
    hist = []
    t0 = time.time()
    tok_seen = 0
    for ep in range(args.epochs):
        R.reset_stats()
        order = rng.permutation(ntr)
        ep_stats = dict(tokens=0, active=0, hinge_sum=0, correct=0)
        nz = tot = 0
        exps = [0]
        tele = {}
        for st in range(0, ntr, args.batch):
            bi = order[st:st + args.batch]
            src = torch.tensor([src_all[i] for i in bi], dtype=torch.int64)
            tin = torch.tensor([tin_all[i] for i in bi], dtype=torch.int64)
            if getattr(args, "scheduled_sampling", 0):
                tin = scheduled_sample(model, src, tin, args.scheduled_sampling,
                                       corrupt_gen)
            if getattr(args, "prefix_corrupt", 0):
                tin = corrupt_prefix(tin, args.prefix_corrupt, corrupt_gen)
            tout = torch.tensor([tout_all[i] for i in bi], dtype=torch.int64)
            want = {} if st == 0 else None
            grads, stt = forward_backward(model, src, tin, tout, hb, cached=False,
                                          telemetry=want,
                                          chunk_head=getattr(args, "chunk_head", 0))
            if st == 0:
                nz = sum(int((g[0] != 0).sum()) for g in grads)
                tot = sum(g[0].numel() for g in grads)
                exps = [int(g[1]) for g in grads]
                tele = want or {}
            opt.step(grads)
            for k in ep_stats:
                ep_stats[k] += stt[k]
            tok_seen += stt["tokens"]
        moved = sum(int((a != b).sum()) for a, b in zip(init_snapshot, model.parameters()))
        ptot = model.n_params()
        row = {
            "epoch": ep,
            "train_hinge_per_token": round(ep_stats["hinge_sum"] / max(1, ep_stats["tokens"]), 4),
            "train_token_acc": round(100.0 * ep_stats["correct"] / max(1, ep_stats["tokens"]), 3),
            "train_margin_active_frac": round(ep_stats["active"] / max(1, ep_stats["tokens"]), 4),
            "param_grad_nonzero_pct": round(100.0 * nz / max(1, tot), 3),
            "weights_moved_since_init_pct": round(100.0 * moved / ptot, 3),
            "grad_exponent_min": min(exps), "grad_exponent_max": max(exps),
            "state_wrap_frac": (round(R.STATS["state_wrap"] / R.STATS["state_elems"], 8)
                                if R.STATS["state_elems"] else 0.0),
            "accum_saturations": int(R.STATS["accum_saturations"]),
            "max_weight": max(int(w.abs().max()) for w in model.parameters()),
            "attn_grad_nonzero_pct": tele.get("attn_grad_nonzero", {}),
            "tokens_per_sec": round(tok_seen / max(1e-9, time.time() - t0), 1),
            "wall_seconds": round(time.time() - t0, 1),
            "process_rss_mb": proc_mem_mb(),
        }
        stop = False
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            if MET is not None:
                rep = MET.epoch_report(model, ds["valid"], hb, args.batch, tok,
                                       klass, ds.get("valid_tags"), args.max_len,
                                       max_eval=args.max_eval)
                row["metrics"] = rep
                print("  ep%-3d %s" % (ep, MET.one_line(rep)), flush=True)
            v = evaluate(model, ds["valid"], hb, args.batch, args.max_eval)
            row.update({"valid_" + k: v[k] for k in
                        ("hinge_per_token", "token_acc", "margin_active_frac",
                         "logit_std", "margin_true_minus_best_wrong")})
            for tag, view in vsplit.items():
                vt = evaluate(model, view, hb, args.batch, args.max_eval)
                row["valid_" + tag] = {"token_acc": vt["token_acc"],
                                       "hinge_per_token": vt["hinge_per_token"],
                                       "tokens": vt["tokens"]}
            # Selection metric. Pooled token accuracy is dominated by EOS -- it
            # was 83% of every correct token in run1 -- so it rewards models
            # that stop early, and that is what it picked. Primary is balanced
            # non-EOS content accuracy, averaged over source corpora so the
            # larger one cannot carry it; content hinge breaks ties. The probe
            # sentences never take part in selection.
            if "metrics" in row:
                tf_ = row["metrics"]["teacher_forced"]
                sc_ = row["metrics"]["source_conditioning"]
                # Content accuracy alone rewards a target-side language model:
                # the 512-pair momentum runs reached 67% teacher-forced while
                # scoring *higher* with the encoder memory zeroed. Selection
                # takes content accuracy only to the extent the source earned
                # it, and refuses a checkpoint the source does not help.
                gap = sc_["content_acc_gap_real_minus_shuffled"]
                real, zero = sc_["real"]["content_acc"], sc_["zeroed"]["content_acc"]
                score = tf_["balanced_content_acc"] if (gap > 0 and real > zero) else 0.0
                row["select_gap"] = gap
                row["select_real_minus_zeroed"] = round(real - zero, 4)
                tie = tf_["balanced_content_hinge"]
            else:
                score, tie = v["token_acc"], v["hinge_per_token"]
            row["select_score"] = score
            row["select_tiebreak"] = tie
            better = (score > best_acc[0] or (score == best_acc[0] and tie < best_acc[1]))
            if getattr(args, "save_ckpt", "") and better:
                best_acc[0] = score
                best_acc[1] = tie
                import os as _os
                save_ckpt(args.save_ckpt.replace(".pt", "_best.pt"), model, args,
                          _os.path.join(D.CACHE, ds["info"]["tokenizer"]), opt, ep)
                row["saved_best"] = True
            # Early stop: `patience` consecutive validations with no new best.
            bad = 0 if better else bad + 1
            row["validations_since_best"] = bad
            if args.patience and bad >= args.patience:
                row["early_stop"] = True
                stop = True
        if getattr(args, "save_ckpt", ""):
            # Every epoch, not just improving ones: an epoch that fails to
            # improve is the one you most want to be able to decode later.
            import os as _os
            tp = _os.path.join(D.CACHE, ds["info"]["tokenizer"])
            save_ckpt(args.save_ckpt.replace(".pt", "_last.pt"), model, args, tp, opt, ep)
            if ep in set(getattr(args, "snapshot_epochs", []) or []):
                save_ckpt(args.save_ckpt.replace(".pt", "_ep%d.pt" % ep), model,
                          args, tp, opt, ep)
        hist.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if getattr(args, "probe_file", ""):
            try:
                epoch_probe(model, tok, args.max_len, args.probe_file, ep,
                            getattr(args, "run_tag", ""))
            except Exception as e:
                print("probe failed: %s: %s" % (type(e).__name__, e), flush=True)
        if stop:
            print("early stop: %d validations without improvement" % bad, flush=True)
            break

    after = evaluate(model, ds["valid"], hb, args.batch, args.show_samples or 10,
                     decode=True, tok=tok, pairs=ds["valid_pairs"])
    src0 = torch.tensor(ds["train"][0][:args.batch], dtype=torch.int64)
    tin0 = torch.tensor(ds["train"][1][:args.batch], dtype=torch.int64)
    tout0 = torch.tensor(ds["train"][2][:args.batch], dtype=torch.int64)
    act = activation_report(model, src0, tin0, tout0, hb)
    result = {"config": vars(args), "data": ds["info"], "history": hist,
              "before": before, "after": after, "activation": act,
              "params": model.n_params(),
              "total_wall_seconds": round(time.time() - t0, 1)}
    if args.out:
        json.dump(result, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    if getattr(args, "save_ckpt", ""):
        import os as _os
        tp = _os.path.join(D.CACHE, ds["info"]["tokenizer"])
        save_ckpt(args.save_ckpt, model, args, tp, opt, args.epochs - 1)
        print("saved checkpoint: " + args.save_ckpt, flush=True)
    return result, model, ds


def epoch_probe(model, tok, max_len, path, epoch, tag=""):
    """
    Decode a fixed held-out probe set and append it to a transcript.

    Called once per epoch from inside training, so it does not depend on a
    checkpoint having been written -- an epoch that fails to improve still gets
    its translations recorded, which is the epoch you most want to look at.
    """
    import reversible_jako_probe as P
    import reversible_jako_decode as DEC
    ids = []
    for ja, _ref in P.PROBE:
        e = tok.encode(ja).ids[:max_len - 1] + [EOS]
        ids.append(e + [PAD] * (max_len - len(e)))
    src = torch.tensor(ids, dtype=torch.int64)
    gy = DEC.decode(model, src, max_len, "greedy", 0, False, 1, 1)
    rp = DEC.decode(model, src, max_len, "greedy-repeat", 8, True, 2, 1)
    gs, gh = DEC.stats(tok, gy)
    rs, rh = DEC.stats(tok, rp)
    lines = ["=" * 76,
             "[%s] %sepoch %d (in-training probe)"
             % (time.strftime("%Y-%m-%d %H:%M:%S"), tag, epoch),
             "=" * 76]
    for i, (ja, ref) in enumerate(P.PROBE):
        lines += ["JA      " + ja, "REF     " + ref,
                  "GREEDY  " + gh[i], "REPEAT  " + rh[i], ""]
    for name, st in (("greedy", gs), ("repeat", rs)):
        lines.append("  %-7s korean%% %-6s repl%% %-5s rep_uni%% %-6s rep_bi%% %-6s distinct %s"
                     % (name, st["korean_char_pct"], st["replacement_char_pct"],
                        st["repeat_unigram_pct"], st["repeat_bigram_pct"],
                        st["distinct_outputs"]))
    lines.append("")
    text = "\n".join(lines)
    print(text, flush=True)
    if path:
        with io.open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def print_samples(title, ev, limit=10):
    print("\n=== " + title + " ===")
    for ja, ref, pred in (ev.get("samples") or [])[:limit]:
        print("JA:   " + ja)
        print("REF:  " + ref)
        print("PRED: " + pred)
        print()
    for k in ("chrf", "bleu", "seq_exact_match", "token_acc", "hinge_per_token"):
        if k in ev:
            print("  " + k + ": " + str(ev[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "memory", "translate"], default="train")
    ap.add_argument("--corpus", default="tatoeba",
                    help="tatoeba | ted2020 | both | opensubtitles | mixed")
    ap.add_argument("--max-pairs", type=int, default=512)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ffn-hidden", type=int, default=128)
    ap.add_argument("--k-enc", type=int, default=4)
    ap.add_argument("--k-dec", type=int, default=6)
    ap.add_argument("--enc-pattern", default="")
    ap.add_argument("--dec-pattern", default="",
                    help="block letters: a=self-attn c=cross-attn f=ffn, e.g. ccaf")
    ap.add_argument("--tok-mode", choices=["char","byte"], default="char")
    ap.add_argument("--score-leak", type=int, default=None,
                    help="power-of-two leaky score map; without it the encoder's "
                         "gradient collapses to zero during training")
    ap.add_argument("--max-tok", type=int, default=0,
                    help="keep only pairs whose both sides fit in this many tokens")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr-bits", type=int, default=2)
    ap.add_argument("--state-bits", type=int, default=16)
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--accum-bits", type=int, default=32)
    ap.add_argument("--hidden-bits", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many validations with no new best "
                         "(0 disables early stopping)")
    ap.add_argument("--subs-pairs", type=int, default=25000,
                    help="corpus=mixed: OpenSubtitles pairs kept after max-tok")
    ap.add_argument("--subs-pool", type=int, default=0,
                    help="corpus=mixed: filtered subtitle pool the tokenizer sees")
    ap.add_argument("--base-pairs", type=int, default=25000,
                    help="corpus=mixed: Tatoeba+TED pairs kept after max-tok")
    ap.add_argument("--base-pool", type=int, default=0,
                    help="corpus=mixed: base pool the tokenizer sees")
    ap.add_argument("--min-char-count", type=int, default=1,
                    help="drop characters rarer than this from the forced "
                         "tokenizer alphabet, freeing vocabulary for merges")
    ap.add_argument("--max-eval", type=int, default=256)
    ap.add_argument("--momentum-bits", type=int, default=0,
                    help="integer momentum in BlockSGD: decay 1-2**-k; 0 = off")
    ap.add_argument("--score-exp2", type=int, default=None,
                    help="base-2 attention: w = 2**((s - rowmax) >> N + cap). "
                         "The integer analogue of softmax's competition, which "
                         "a bisection showed is the one thing its absence "
                         "costs. Unset keeps the linear/leaky map")
    ap.add_argument("--score-exp2-cap", type=int, default=12)
    ap.add_argument("--exp2-diag", action="store_true",
                    help="per-entry adjoint for base-2 attention (diagnostic "
                         "only; the coupled adjoint is what makes generation work)")
    ap.add_argument("--attn-score-bits", type=int, default=0,
                    help="bits kept in the attention scores; 0 keeps TB+2 = 7")
    ap.add_argument("--attn-mass-bits", type=int, default=0,
                    help="cap each attention row's L1 mass at 2**N with one "
                         "power-of-two shift; 0 keeps the old max-based "
                         "behaviour")
    ap.add_argument("--scheduled-sampling", type=int, default=0,
                    help="percent of gold decoder-input tokens replaced by the "
                         "model's own integer argmax during training")
    ap.add_argument("--positional", type=int, default=0,
                    help="length of the learned integer positional tables; 0 "
                         "creates none and is bit-identical to no option")
    ap.add_argument("--prefix-corrupt", type=int, default=0,
                    help="percent of gold decoder-input tokens replaced by UNK "
                         "during training, to make the target-prefix shortcut "
                         "insufficient; validation is never corrupted")
    ap.add_argument("--query-anchor", action="store_true",
                    help="cross-attention BOS query anchoring (run3 knob; off here)")
    ap.add_argument("--snapshot-epochs", type=int, nargs="*", default=[],
                    help="also save <ckpt>_ep<N>.pt at these epochs, so the "
                         "query-collapse diagnostics can be re-measured later")
    ap.add_argument("--probe-file", default="",
                    help="append a held-out probe translation after every epoch")
    ap.add_argument("--run-tag", default="",
                    help="label written into the probe transcript")
    ap.add_argument("--chunk-head", type=int, default=0,
                    help="walk the output vocabulary in chunks of this size and "
                         "take the sparse hinge backward; bit-identical to the "
                         "full-tensor path (reversible_jako_headgate.py)")
    ap.add_argument("--show-samples", type=int, default=10)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--out", default="jako_results.json")
    ap.add_argument("--save-ckpt", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--ckpt", default="jako_model.pt")
    ap.add_argument("--text", nargs="*", default=[])
    args = ap.parse_args()

    if args.mode == "translate":
        torch.set_num_threads(args.threads)
        model, tok, ck = load_ckpt(args.ckpt)
        sents = args.text or [
            "今日はとても寒いです。",
            "私は学生です。",
            "ありがとうございます。",
        ]
        for s_, p_ in zip(sents, translate(model, tok, sents, ck["max_len"])):
            print("JA:   " + s_)
            print("PRED: " + p_)
            print()
        return

    if args.mode == "memory":
        torch.set_num_threads(args.threads)
        cfg = R.RunConfig(state_bits=args.state_bits, weight_bits=args.weight_bits,
                          accum_bits=args.accum_bits)
        R.CFG = cfg
        ds = D.prepare(args.corpus, 256, args.vocab_size, args.max_len,
                       tok_mode=getattr(args, "tok_mode", "char"))
        vocab = ds["info"]["vocab_size"]
        rows = []
        for K in args.depths:
            model = JaKoMT(vocab, args.dim, args.ffn_hidden, K, K, cfg, args.seed)
            src = torch.tensor(ds["train"][0][:args.batch], dtype=torch.int64)
            tin = torch.tensor(ds["train"][1][:args.batch], dtype=torch.int64)
            tout = torch.tensor(ds["train"][2][:args.batch], dtype=torch.int64)
            r = {"K_enc": K, "K_dec": K,
                 **activation_report(model, src, tin, tout, args.hidden_bits)}
            rows.append(r)
            print(json.dumps(r), flush=True)
        json.dump(rows, open("jako_memory.json", "w"), indent=2)
        return

    result, model, ds = train(args)
    if result.get("before"):
        print_samples("BEFORE TRAINING", result["before"])
    print_samples("AFTER TRAINING", result["after"])
    print("\nactivation: " + json.dumps(result["activation"]))
    print("total wall seconds: " + str(result["total_wall_seconds"]))


if __name__ == "__main__":
    main()
