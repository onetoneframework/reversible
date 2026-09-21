"""
REFERENCE ONLY - superseded by reversible_integer_final.py.

Every feature below has been ported into the PyTorch mainline and cross-checked
against this file on sklearn digits (K=16, 4 epochs, seeds 0-9, state 12 bit /
weight 8 bit / grad 8 bit / accum 32 bit):

    feature                     mainline option
    width limits                --state-bits / --weight-bits / --accum-bits
    grouped compressed GEMM     --groups g  (+ --hidden-width, --phase-step)
    dispersing permutation      --perm transpose
    wrap / saturation counters  reported as state_wrap_frac, accum_saturations

Measured agreement (both sides re-measured here, nothing carried over):
    dense g=1 H64          numpy 87.82 +/- 1.75   torch 88.02 +/- 2.32  (10 seeds)
    g=4 H64 transpose      numpy 84.67 +/- 3.18   torch 83.20 +/- 1.28  (5 seeds)
    g=8 H256 transpose     numpy 87.20 +/- 1.41   torch 86.53 +/- 3.07  (5 seeds)
    g=8 H64 default perm   numpy 39.18 +/- 2.77   torch 38.33 +/- 2.80  (10 seeds)

Keep this file for auditability of the original NumPy derivation. New work,
and anything that needs the reversible/cached/legacy/float baselines or the
equiv / implequiv / stepequiv gates, belongs in reversible_integer_final.py.

Note: this file's --state-bits default is 8, which collapses to chance. The
results above and in the project notes were produced with --state-bits 12.

int8 variant of the exact reversible integer network.

Width policy (simulated, stored in int64 but range-enforced):
  state v      : WBITS (default 8)  -- two's complement wraparound
  weights U,V  : WEIGHT_BITS (default 8)
  GEMM accum   : int32
  grad mantissa: GBITS (default 8) with one shared exponent per tensor

Wraparound note: two's complement add/sub is exactly invertible even on
overflow, so bit-exact reversibility survives narrow state. Overflow costs
accuracy, not correctness. We count it.
"""
import argparse, json, time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

TB = 5
BATCH = 32
M = 64
H = 64
NCLASS = 10
WID = 2 * M

_X, _y = load_digits(return_X_y=True)
_Xtr, _Xte, _ytr, _yte = train_test_split(_X, _y, test_size=0.25, random_state=0, stratify=_y)
XTR = _Xtr.astype(np.int64); XTE = _Xte.astype(np.int64)
YTR = _ytr.astype(np.int64); YTE = _yte.astype(np.int64)

HALVES = np.roll(np.arange(WID), M)
ODDEVEN = np.arange(WID) ^ 1


def outward_fold_perm(width=WID, expand=2):
    """Scatter state into an empty 2W buffer with the middle left empty:
    first half to the left outer region (even slots), second half to the right
    outer region (descending). Fold with a column-wise add, out[j]=buf[j]+buf[j+W].
    Exactly one nonzero lands per output column, so the fold is lossless and the
    whole operation is a permutation -- reversibility is preserved."""
    B = width * expand
    pos = np.zeros(width, dtype=np.int64)
    half = width // 2
    for i in range(half):
        pos[i] = 2 * i
    for i in range(half, width):
        pos[i] = B - 1 - 2 * (i - half)
    src = np.full(width, -1, dtype=np.int64)
    for i, p in enumerate(pos):
        j = p % width
        assert src[j] == -1
        src[j] = i
    return pos, src


SCATTER_POS, OUTWARD = outward_fold_perm()
INV_H = np.argsort(HALVES); INV_O = np.argsort(ODDEVEN)
INV_OUT = np.argsort(OUTWARD)
PERM_MODE = "default"
PHASE_STEP = 0
GROUPS = 1


def _transpose_perm(g):
    """Fixed, table-free permutation satisfying the dispersion condition:
    grid transpose (M/g x g) inside each half, so every residue group is
    spread across all g residue groups of the next layer. Requires M/g % g == 0.
    Alternated with HALVES so the two coupling halves also cross."""
    i = np.arange(M)
    p = (i // g) + (i % g) * (M // g)
    return np.concatenate([p, p + M])


def _bitshuffle_perm(g, rot=1):
    """Table-free, full dispersion, non-involution. Index bits = (within | group).
    New group bits = top bits of old within (every old group -> every new group);
    new within bits = remaining old within bits + old group bits rotated by `rot`.
    Order 6 for M=64, g in {4, 8}; the plain grid transpose has order 2 at g=8."""
    gb = int(np.log2(g)); wb = int(np.log2(M)) - gb
    i = np.arange(M); low = i & (g - 1); high = i >> gb
    low_rot = ((low << rot) | (low >> (gb - rot))) & (g - 1) if gb > 1 else low
    top = high >> (wb - gb); rest = high & ((1 << (wb - gb)) - 1)
    p = (((rest << gb) | low_rot) << gb) | top
    return np.concatenate([p, p + M])


_RP = {}
def _rand_perm(k):
    if k not in _RP:
        _RP[k] = np.random.default_rng(1000 + k).permutation(WID)
    return _RP[k]


def perm_for(k):
    if PERM_MODE == "random":
        return _rand_perm(k)
    if PERM_MODE == "transpose":
        return HALVES if k % 2 == 0 else _transpose_perm(GROUPS)
    if PERM_MODE == "bitshuffle":
        return HALVES if k % 2 == 0 else _bitshuffle_perm(GROUPS)
    if PERM_MODE == "outward":
        return OUTWARD
    if PERM_MODE == "mixed":
        return OUTWARD if k % 2 == 0 else ODDEVEN
    return HALVES if k % 2 == 0 else ODDEVEN


def invperm_for(k):
    if PERM_MODE == "random":
        return np.argsort(_rand_perm(k))
    if PERM_MODE == "transpose":
        return INV_H if k % 2 == 0 else np.argsort(_transpose_perm(GROUPS))
    if PERM_MODE == "bitshuffle":
        return INV_H if k % 2 == 0 else np.argsort(_bitshuffle_perm(GROUPS))
    if PERM_MODE == "outward":
        return INV_OUT
    if PERM_MODE == "mixed":
        return INV_OUT if k % 2 == 0 else INV_O
    return INV_H if k % 2 == 0 else INV_O


def apply_outward(v):
    """Literal form: build the empty expanded buffer, push values outward,
    add column-wise. Equivalent to v[:, OUTWARD] but counts the real ops."""
    n = v.shape[0]
    buf = np.zeros((n, WID * 2), dtype=np.int64)
    buf[:, SCATTER_POS] = v
    STATS["scatter"] += v.size
    STATS["add"] += WID * n
    return buf[:, :WID] + buf[:, WID:]

STATS = {"state_wrap": 0, "state_elems": 0, "acc_sat": 0,
         "mac": 0, "shift": 0, "cmp": 0, "add": 0, "scatter": 0}


def checkerboard(rows, cols, phase=0):
    """Fixed alternating mask: n0n0 / 0n0n. No index storage needed."""
    i = np.arange(rows)[:, None]; j = np.arange(cols)[None, :]
    return ((i + j + phase) % 2 == 0).astype(np.int64)


def gemm(x, W, mask_density=1.0):
    """Integer matmul with MAC accounting. mask_density scales counted MACs."""
    STATS["mac"] += int(x.shape[0] * x.shape[1] * W.shape[1] * mask_density)
    return x @ W


def wrap(x, bits):
    """Two's complement wraparound to `bits`. Exactly invertible."""
    if bits >= 63:
        return x
    n = np.int64(1) << np.int64(bits)
    y = ((x + (n >> 1)) & (n - 1)) - (n >> 1)
    STATS["state_wrap"] += int(np.count_nonzero(y != x))
    STATS["state_elems"] += int(np.size(x))
    return y


def sat(x, bits):
    """Saturating clip (for accumulators / weights)."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    n = int(np.count_nonzero((x < lo) | (x > hi)))
    STATS["acc_sat"] += n
    return np.clip(x, lo, hi)


def rshift0(x, sh):
    return np.sign(x) * (np.abs(x) >> np.clip(sh, 0, 62))


def bitlen(x):
    z = np.abs(np.asarray(x, dtype=np.int64)); b = np.zeros_like(z)
    for k in (32, 16, 8, 4, 2, 1):
        q = (z >> k) > 0
        b += q * k
        z = np.where(q, z >> k, z)
    return b + (z > 0)


def pow2norm(x, tb):
    s = np.maximum(bitlen(np.abs(x).max(1, keepdims=True)) - tb, 0)
    STATS["shift"] += x.size
    return x >> s, s


def block_bitlen(m):
    return int(np.max(np.abs(m))).bit_length() if np.size(m) else 0


def norm_block(m, e, bits):
    b = block_bitlen(m)
    if b == 0:
        return m, int(e)
    d = b - bits
    if d > 0:
        return rshift0(m, d), int(e) + d
    if d < 0:
        k = min(-d, max(bits - 1, 0))
        return m << k, int(e) - k
    return m, int(e)


def residue_perm(n, g, phase=0):
    """Placement array: index i belongs to residue group (i+phase) % g.
    Rows/cols with the same residue are connected; the rest are structural zeros.
    g=1 is dense, g=2 is the checkerboard."""
    return (np.arange(n) + phase) % g


class Coup:
    def __init__(self, rng, wbits, sbits, abits, groups=1, phase=0):
        self.wbits, self.sbits, self.abits, self.g = wbits, sbits, abits, groups
        g = groups
        assert M % g == 0 and H % g == 0
        lim = 1 << (wbits - 4)
        # compact storage: g dense blocks of (H/g, M/g) and (M/g, H/g)
        self.Ug = [rng.integers(-lim, lim + 1, (H // g, M // g)).astype(np.int64) for _ in range(g)]
        self.Vg = [rng.integers(-lim, lim + 1, (M // g, H // g)).astype(np.int64) for _ in range(g)]
        self.rU = residue_perm(H, g, phase); self.cU = residue_perm(M, g, phase)
        self.rV = residue_perm(M, g, phase); self.cV = residue_perm(H, g, phase)
        self.density = 1.0 / g

    def params(self):
        return self.Ug + self.Vg

    def gmm(self, x, blocks, rows_res, cols_res, transpose):
        """Grouped matmul via outward compaction: for each residue c, take the
        input columns in group c, multiply by the dense block, write the output
        columns in group c. Vertical merge is a strided scatter -- one block per
        output column, so it is lossless."""
        g = self.g
        n_out = len(rows_res)
        out = np.zeros((x.shape[0], n_out), dtype=np.int64)
        for c in range(g):
            xi = x[:, cols_res == c]
            W = blocks[c].T if transpose else blocks[c]
            out[:, rows_res == c] = gemm(xi, W, 1.0)
        return out

    def parts(self, a):
        z, s1 = pow2norm(sat(self.gmm(a, self.Ug, self.rU, self.cU, True), self.abits), TB + 2)
        STATS["cmp"] += z.size
        h = np.maximum(z, 0)
        f, s2 = pow2norm(sat(self.gmm(h, self.Vg, self.rV, self.cV, True), self.abits), TB + 2)
        return f, z, h, s1, s2

    def forward(self, v):
        a, b = v[:, :M], v[:, M:]
        f, *_ = self.parts(a)
        STATS["add"] += f.size
        return np.concatenate([a, wrap(b + f, self.sbits)], 1)

    def inverse(self, v):
        a, bo = v[:, :M], v[:, M:]
        f, *_ = self.parts(a)
        STATS["add"] += f.size
        return np.concatenate([a, wrap(bo - f, self.sbits)], 1)


def fwd(x, layers, sbits):
    v = np.concatenate([wrap(x, sbits), np.zeros_like(x)], 1)
    for k, L in enumerate(layers):
        v = L.forward(v)
        if k < len(layers) - 1:
            v = v[:, perm_for(k)]
    return v


def inv_net(v, layers):
    for k in range(len(layers) - 1, -1, -1):
        if k < len(layers) - 1:
            v = v[:, invperm_for(k)]
        v = layers[k].inverse(v)
    return v


def out_grad(v, y, gbits):
    out = v[:, M:M + NCLASS]; n = len(y)
    mk = out.copy(); mk[np.arange(n), y] = np.iinfo(np.int64).min // 4
    j = mk.argmax(1)
    act = out[np.arange(n), j] + 1 > out[np.arange(n), y]
    g = np.zeros_like(v); ii = np.arange(n)[act]
    g[ii, M + j[act]] += 1
    g[ii, M + y[act]] -= 1
    return norm_block(g, 0, gbits)


def lin_bwd(gm, ge, x, L, which, s, gbits, abits):
    """Backward of y = grouped(x, W) >> s with one shared exponent.
    Returns input-grad (mantissa, exp) and a list of per-block weight grads."""
    s = s.reshape(-1)
    se = ge - s
    E = int(se.max())
    d = (E - se).reshape(-1, 1)
    STATS["shift"] += gm.size
    blocks, rr, cc = (L.Ug, L.rU, L.cU) if which == "U" else (L.Vg, L.rV, L.cV)
    # gx = gm @ W  -> grouped: input col group c gets gm[:, out group c] @ block[c]
    gx = np.zeros((gm.shape[0], len(cc)), dtype=np.int64)
    gws = []
    ga = rshift0(gm, d)
    for c in range(L.g):
        go = gm[:, rr == c]
        gx[:, cc == c] = gemm(go, blocks[c], 1.0)
        gws.append(sat(gemm(ga[:, rr == c].T, x[:, cc == c], 1.0), abits))
    gx = sat(rshift0(gx, d), abits)
    gxm, gxe = norm_block(gx, E, gbits)
    return (gxm, gxe), gws


def align(m1, e1, m2, e2, bits):
    E = max(int(e1), int(e2))
    return norm_block(rshift0(m1, E - e1) + rshift0(m2, E - e2), E, bits)


def cat(m1, e1, m2, e2, bits):
    E = max(int(e1), int(e2))
    return norm_block(np.concatenate([rshift0(m1, E - e1), rshift0(m2, E - e2)], 1), E, bits)


class BlockSGD:
    def __init__(self, params, pbits, lr_bits, clip, wbits):
        self.p = params
        self.a = [np.zeros_like(w) for w in params]
        self.sh = (pbits - 1) + lr_bits
        self.clip = clip; self.pbits = pbits; self.wbits = wbits

    def renorm(self, g):
        b = block_bitlen(g)
        if b == 0:
            return g
        d = b - self.pbits
        return g << (-d) if d < 0 else (rshift0(g, d) if d > 0 else g)

    def step_one(self, i, gm):
        w, a = self.p[i], self.a[i]
        a -= self.renorm(gm)
        q = np.clip(np.sign(a) * (np.abs(a) >> self.sh), -self.clip, self.clip)
        w += q
        np.clip(w, -(1 << (self.wbits - 1)), (1 << (self.wbits - 1)) - 1, out=w)
        a -= q << self.sh


def train(K, epochs, seed, wbits, sbits, gbits, abits, lr_bits=2, groups=1, hidden=None):
    global H
    if hidden:
        H = hidden
    rng = np.random.default_rng(seed)
    layers = [Coup(rng, wbits, sbits, abits, groups, phase=(k * PHASE_STEP) % groups)
              for k in range(K)]
    params = [w for L in layers for w in L.params()]
    opt = BlockSGD(params, 12, lr_bits, 1, wbits)
    t0 = time.time()
    for _ in range(epochs):
        idx = rng.permutation(len(XTR))
        for st in range(0, len(XTR), BATCH):
            bi = idx[st:st + BATCH]
            v = fwd(XTR[bi], layers, sbits)
            gm, ge = out_grad(v, YTR[bi], gbits)
            for k in range(K - 1, -1, -1):
                if k < K - 1:
                    iv = invperm_for(k)
                    v = v[:, iv]; gm = gm[:, iv]
                    gm, ge = norm_block(gm, ge, gbits)
                L = layers[k]
                a = v[:, :M]
                f, z, h, s1, s2 = L.parts(a)
                v = np.concatenate([a, wrap(v[:, M:] - f, sbits)], 1)
                gam, gbm = gm[:, :M], gm[:, M:]
                (ghm, ghe), gVs = lin_bwd(gbm, ge, h, L, "V", s2, gbits, abits)
                ghm = ghm * (z > 0)
                (dam, dae), gUs = lin_bwd(ghm, ghe, a, L, "U", s1, gbits, abits)
                nm, ne = align(gam, ge, dam, dae, gbits)
                gm, ge = cat(nm, ne, gbm, ge, gbits)
                base = k * 2 * L.g
                for c in range(L.g):
                    opt.step_one(base + c, gUs[c])
                    opt.step_one(base + L.g + c, gVs[c])
    vt = fwd(XTE, layers, sbits)
    acc = float((vt[:, M:M + NCLASS].argmax(1) == YTE).mean())
    xs = XTE[:32]
    rec = inv_net(fwd(xs, layers, sbits), layers)
    tgt = np.concatenate([wrap(xs, sbits), np.zeros_like(xs)], 1)
    nz = int(sum(np.count_nonzero(w) for w in params))
    return dict(K=K, epochs=epochs, seed=seed, groups=groups, phase_step=PHASE_STEP,
                perm=PERM_MODE, hidden=H, scatter=STATS["scatter"], mac=STATS["mac"], shift=STATS["shift"], cmp=STATS["cmp"],
                add=STATS["add"], nonzero_weights=nz,
                state_bits=sbits, weight_bits=wbits,
                grad_bits=gbits, accum_bits=abits, accuracy=round(acc, 4),
                inverse_exact=bool(np.array_equal(rec, tgt)),
                state_wrap_frac=round(STATS["state_wrap"] / max(1, STATS["state_elems"]), 5),
                accum_saturations=STATS["acc_sat"],
                max_weight=int(max(np.abs(w).max() for w in params)),
                seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--state-bits", type=int, default=8)
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--grad-bits", type=int, default=8)
    ap.add_argument("--accum-bits", type=int, default=32)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--phase-step", type=int, default=0,
                    help="layer k uses residue phase k*step mod g (0 = same groups every layer)")
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--perm", choices=["default", "outward", "mixed", "random", "transpose", "bitshuffle"], default="default")
    a = ap.parse_args()
    STATS.update(state_wrap=0, state_elems=0, acc_sat=0, scatter=0)
    m_perm = a.perm
    globals()["PERM_MODE"] = m_perm
    globals()["PHASE_STEP"] = a.phase_step
    globals()["GROUPS"] = a.groups
    print(json.dumps(train(a.K, a.epochs, a.seed, a.weight_bits, a.state_bits,
                           a.grad_bits, a.accum_bits, groups=a.groups,
                           hidden=(a.hidden or None))))
