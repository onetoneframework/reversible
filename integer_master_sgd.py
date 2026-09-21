"""IntegerMasterSGD: int32 sub-grid master weights driven by SharedGrad mantissa AND exponent.

The integer model keeps its int8-range forward weights W. Behind each W this
optimizer holds a master M = W << FRAC_BITS in int32, updated directly from the
backward's SharedGrad(m, e), whose value is m * 2**e (reversible_integer_final.
norm_block). With learning rate 2**-LR_SHIFT the master-space step is

    delta_M = m * 2**(e + FRAC_BITS - LR_SHIFT)

realised as a left shift when the exponent is non-negative and rshift0 (the
codebase's signed power-of-two shift, truncation toward zero) when it is negative.
Descent is M -= delta_M, the same orientation BlockSGD uses (acc -= gn, w += sign).

Nothing is renormalised per tensor, no exponent is discarded, no sign-only step,
no firing threshold, no accumulator, no momentum. The forward weight is
W = clamp(rshift0(M, FRAC_BITS), -128, 127), the same clamp range BlockSGD applies.
M is never overwritten from W; it keeps its sub-grid state.

All optimizer arithmetic is int64 on int32 storage. Any int32 overflow raises;
there is no saturation and no wrap.
"""

import torch

import reversible_integer_final as R

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


def mant_exp(g):
    """(mantissa, exponent) of a gradient entry.

    reversible_jako_mt.forward_backward emits plain (m, e) tuples (its emit()
    stores the pair verbatim, and BlockSGD consumes only pair[0], discarding the
    exponent); hinge_grad builds a SharedGrad dataclass. Accept either form.
    """
    if hasattr(g, "m"):
        return g.m, int(g.e)
    return g[0], int(g[1])


def shift_signed(m: torch.Tensor, s: int) -> torch.Tensor:
    """m * 2**s in integers: left shift for s >= 0, rshift0 for s < 0."""
    s = int(s)
    if s >= 0:
        return m.to(torch.int64) << s
    return R.rshift0(m.to(torch.int64), -s)


def prospective_bits(grads, frac_bits: int):
    """floor(log2|m|) + e + frac_bits for every nonzero SharedGrad element (LR_SHIFT = 0)."""
    out = []
    for g in grads:
        m, e = mant_exp(g)
        nz = m.reshape(-1)
        nz = nz[nz != 0]
        if nz.numel():
            out.append(R.bitlen_arr(nz) - 1 + int(e) + frac_bits)
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.int64)


def percentile_order_statistic(values: torch.Tensor, q: float) -> int:
    """Deterministic order statistic: the ceil(q*n)-th smallest value."""
    v = torch.sort(values)[0]
    n = v.numel()
    idx = max(0, min(n - 1, int(-(-q * n // 1)) - 1))
    return int(v[idx])


class IntegerMasterSGD:
    def __init__(self, params, lr_shift: int, frac_bits: int = 12, weight_bits: int = 8):
        self.params = list(params)
        self.frac_bits = int(frac_bits)
        self.lr_shift = int(lr_shift)
        self.lo, self.hi = -(1 << (weight_bits - 1)), (1 << (weight_bits - 1)) - 1
        self.masters = []
        for w in self.params:
            if w.dtype != torch.int64:
                raise TypeError("integer parameters expected")
            M = (w.to(torch.int64) << self.frac_bits)
            if int(M.max()) > INT32_MAX or int(M.min()) < INT32_MIN:
                raise OverflowError("master initialisation does not fit int32")
            self.masters.append(M.to(torch.int32))
        for w, M in zip(self.params, self.masters):
            if not torch.equal(self.quantize(M), w):
                raise RuntimeError("quantize(M) != W at initialisation")
        self.steps = 0
        self.unique_changed = [torch.zeros(w.shape, dtype=torch.bool) for w in self.params]
        self.cum_master_changed = 0.0
        self.cum_w_changed = 0.0
        self.cum_abs_delta = 0.0
        self.cum_nonzero = 0
        self.max_master_abs = 0
        self.clip_events = 0
        self.total = sum(w.numel() for w in self.params)

    def quantize(self, M: torch.Tensor) -> torch.Tensor:
        return torch.clamp(R.rshift0(M.to(torch.int64), self.frac_bits), self.lo, self.hi)

    def deltas(self, grads):
        out = []
        for g in grads:
            m, e = mant_exp(g)
            if m.dtype != torch.int64:
                raise TypeError("SharedGrad mantissa must be int64")
            out.append(shift_signed(m, int(e) + self.frac_bits - self.lr_shift))
        return out

    def dry_run(self, grads):
        """Deltas and what one step would change, without touching any state."""
        ds = self.deltas(grads)
        m_changed = w_changed = 0
        overflow = False
        for w, M, d in zip(self.params, self.masters, ds):
            n = M.to(torch.int64) - d
            if int(n.max()) > INT32_MAX or int(n.min()) < INT32_MIN:
                overflow = True
            m_changed += int((n != M.to(torch.int64)).sum())
            w_changed += int((self.quantize(n) != w).sum())
        return ds, {"master_changed_fraction": m_changed / self.total,
                    "quantized_changed_fraction": w_changed / self.total,
                    "int32_overflow": overflow}

    def step(self, grads):
        ds = self.deltas(grads)
        new = []
        for M, d in zip(self.masters, ds):
            n = M.to(torch.int64) - d
            if int(n.max()) > INT32_MAX or int(n.min()) < INT32_MIN:
                raise OverflowError("int32 master overflow at step %d" % (self.steps + 1))
            new.append(n)
        m_changed = w_changed = clip = 0
        abs_sum = 0
        nz = 0
        mmax = 0
        for i, (w, M, n, d) in enumerate(zip(self.params, self.masters, new, ds)):
            m_changed += int((n != M.to(torch.int64)).sum())
            self.masters[i] = n.to(torch.int32)
            raw = R.rshift0(n, self.frac_bits)
            clip += int(((raw < self.lo) | (raw > self.hi)).sum())
            Wq = torch.clamp(raw, self.lo, self.hi)
            ch = Wq != w
            w_changed += int(ch.sum())
            self.unique_changed[i] |= ch
            w.copy_(Wq)
            abs_sum += int(d.abs().sum())
            nz += int((d != 0).sum())
            mmax = max(mmax, int(n.abs().max()))
        self.steps += 1
        self.cum_master_changed += m_changed / self.total
        self.cum_w_changed += w_changed / self.total
        self.cum_abs_delta += abs_sum
        self.cum_nonzero += nz
        self.max_master_abs = max(self.max_master_abs, mmax)
        self.clip_events += clip
        return {"master_changed": m_changed, "quantized_changed": w_changed,
                "clip": clip, "master_max_abs": mmax}

    def telemetry(self):
        steps = max(1, self.steps)
        return {"steps": self.steps,
                "mean_master_changed_fraction_per_step": self.cum_master_changed / steps,
                "mean_quantized_changed_fraction_per_step": self.cum_w_changed / steps,
                "cumulative_unique_quantized_changed": int(sum(int(u.sum()) for u in self.unique_changed)),
                "cumulative_unique_quantized_changed_fraction":
                    sum(int(u.sum()) for u in self.unique_changed) / self.total,
                "mean_abs_delta_M_all": self.cum_abs_delta / max(1, self.total * steps),
                "mean_abs_delta_M_nonzero": self.cum_abs_delta / max(1, self.cum_nonzero),
                "master_int32_max_abs": self.max_master_abs,
                "master_int32_headroom_bits": (31 - (self.max_master_abs.bit_length() if self.max_master_abs else 0)),
                "clip_events_total": self.clip_events,
                "lr_shift": self.lr_shift, "frac_bits": self.frac_bits}
