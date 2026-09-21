"""reversible_jako_mt.py --mode train with split momentum, without editing it.

Adds one option, --source-momentum-bits K. When given, the optimizer becomes
two canonical BlockSGD instances over the same parameters: the target-side
tensors (emb_tgt, pos_tgt, decoder self-attention, decoder FFN, output head)
step with --momentum-bits, the source pathway (emb_src, pos_src, encoder,
decoder cross-attention) steps with K. Every index is stepped by exactly one
instance, so each update is bit-identical to the canonical optimizer at that
index's momentum. Without the option this is exactly reversible_jako_mt.main().

Everything else -- data, model, backward, validation, checkpoints, probes --
is the unmodified trainer.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import reversible_jako_mt as M
from reversible_integer_final import BlockSGD

SOURCE_GROUPS = {"emb_src", "pos_src", "enc_self", "enc_ffn", "dec_cross"}


def groups_of(model):
    idx = M.param_index(model)
    lab = {idx["emb_src"]: "emb_src", idx["emb_tgt"]: "emb_tgt", idx["wout"]: "wout"}
    for stack, base, pre in ((model.enc, idx["enc"], "enc"), (model.dec, idx["dec"], "dec")):
        for f, off in zip(stack.fs, stack.block_offsets()):
            g = pre + (("_cross" if f.cross else "_self") if f.kind == "attn" else "_ffn")
            for c in range(len(f.blocks())):
                lab[base + off + c] = g
    if model.max_pos:
        lab[idx["pos_src"]], lab[idx["pos_tgt"]] = "pos_src", "pos_tgt"
    return lab


class SplitMomentumSGD:
    def __init__(self, params, lab, tgt_bits, src_bits, **kw):
        self.tgt = BlockSGD(params, momentum_bits=tgt_bits, **kw)
        self.src = BlockSGD(params, momentum_bits=src_bits, **kw)
        self.route = [self.src if lab[i] in SOURCE_GROUPS else self.tgt for i in range(len(params))]
        self.params = params
        self.shift, self.stable = self.tgt.shift, self.tgt.stable
        self.weight_bits, self.momentum_bits = self.tgt.weight_bits, tgt_bits
        self.acc = self.tgt.acc          # target-side accumulators (what save_ckpt stores)
        n_src = sum(1 for r in self.route if r is self.src)
        print("split momentum: %d source-pathway tensors at %d bits, %d target-side tensors at %d bits"
              % (n_src, src_bits, len(params) - n_src, tgt_bits), flush=True)

    def _step_one(self, i, gm):
        return self.route[i]._step_one(i, gm)

    def step(self, grad_pairs):
        return max(self.route[i]._step_one(i, gm) for i, (gm, _ge) in enumerate(grad_pairs))

    def step_layer(self, k, gUs, gVs):
        raise NotImplementedError("per-layer stepping is not routed")


def make_ce2_grad(k, cap, norm="exact"):
    """Integer base-2 cross-entropy head gradient.

    p_i = 2**(rshift0(l_i - rowmax, k) + cap) for classes within cap*2**k of the
    row max (0 beyond), Z = sum p_i. The gradient is p/Z - onehot:

      norm="exact": one integer division per row, r = 2**60 // Z, then
                    q_i = (p_i * r) >> 36 is p_i/Z in 24-bit fixed point
                    (error < 2**-24); mantissa q_i - onehot_i * 2**24, exponent -24.
      norm="pow2":  shift-only, Z replaced by 2**bitlen(Z) >= Z; underestimates
                    every probability by up to 2x, so a confidently correct row
                    keeps a residual push of up to -0.5 on its target and the
                    head never settles (measured: collapse to short repeated
                    outputs on the 67k set).

    Then norm_block to hb bits like hinge_grad. Same return shape and stats.
    """
    import reversible_integer_final as R

    def ce2_grad(logits, tgt_out, hb):
        B, T, V = logits.shape
        flat = logits.reshape(B * T, V)
        tgt = tgt_out.reshape(B * T)
        n = flat.shape[0]
        rows = torch.arange(n)
        notpad = tgt != M.PAD
        rowmax = flat.amax(dim=1, keepdim=True)
        e = R.rshift0(flat - rowmax, k) + cap
        p = torch.where(e >= 0, torch.ones_like(e) << torch.clamp(e, min=0), torch.zeros_like(e))
        z = p.sum(dim=1, keepdim=True)                        # 2**cap <= Z < V * 2**cap
        if norm == "exact":
            r = torch.div(torch.full_like(z, 1 << 60), z, rounding_mode="floor")   # <= 2**(60-cap)
            g = (p * r) >> 36                                 # p/Z * 2**24, product <= 2**(80-cap)
            g[rows, tgt] -= 1 << 24
            g[~notpad] = 0
            gm, ge = R.norm_block(g, -24, hb)
        else:
            s = R.bitlen_arr(z)                               # 2**s >= Z
            g = p.clone()
            g[rows, tgt] -= (1 << s.reshape(-1))
            g[~notpad] = 0
            smax = int(s.max())
            g = g << (smax - s)                               # align every row to exponent -smax
            gm, ge = R.norm_block(g, -smax, hb)
        masked = flat.clone()
        masked[rows, tgt] = torch.iinfo(torch.int64).min // 4
        j = masked.argmax(dim=1)
        true_l = flat[rows, tgt]
        best_w = flat[rows, j]
        active = (best_w + 1 > true_l) & notpad
        stats = {
            "tokens": int(notpad.sum().item()),
            "active": int(active.sum().item()),
            "hinge_sum": int(torch.clamp_min((best_w + 1 - true_l)[notpad], 0).sum().item()),
            "correct": int(((flat.argmax(1) == tgt) & notpad).sum().item()),
            "margin_mean": float((true_l - best_w)[notpad].double().mean().item()) if int(notpad.sum()) else 0.0,
            "logit_std": float(flat[notpad].double().std().item()) if int(notpad.sum()) > 1 else 0.0,
        }
        return M.SharedGrad(gm.reshape(B, T, V), ge), stats

    return ce2_grad


def main():
    argv = sys.argv[1:]
    src_bits = None
    if "--source-momentum-bits" in argv:
        i = argv.index("--source-momentum-bits")
        src_bits = int(argv[i + 1])
        del argv[i:i + 2]
    head, ce2_shift, ce2_cap, ce2_norm = "hinge", 0, 20, "exact"
    for name in ("--head", "--ce2-shift", "--ce2-cap", "--ce2-norm"):
        if name in argv:
            i = argv.index(name)
            val = argv[i + 1]
            del argv[i:i + 2]
            if name == "--head":
                head = val
            elif name == "--ce2-shift":
                ce2_shift = int(val)
            elif name == "--ce2-cap":
                ce2_cap = int(val)
            else:
                ce2_norm = val
    if head == "ce2":
        if "--chunk-head" in argv:
            raise SystemExit("--head ce2 needs the dense output-head backward; drop --chunk-head")
        M.hinge_grad = make_ce2_grad(ce2_shift, ce2_cap, ce2_norm)
        print("output head: integer base-2 cross-entropy (shift %d, cap %d, norm %s)" % (ce2_shift, ce2_cap, ce2_norm), flush=True)
    if "--tb" in argv:
        # activation target bits: every linear output is pow2norm'd to TB+2 bits (historical 5 -> 7);
        # pass --attn-score-bits 7 alongside to keep the exp2 attention scoring unchanged
        i = argv.index("--tb")
        tb = int(argv[i + 1])
        del argv[i:i + 2]
        import reversible_integer_final as R
        import reversible_arch as A
        R.TB = M.TB = A.TB = tb
        print("activation bits: TB %d -> %d-bit linear outputs" % (tb, tb + 2), flush=True)
    opt_kind, lr_shift, frac_bits = "blocksgd", 9, 12
    for name in ("--opt", "--lr-shift", "--frac-bits"):
        if name in argv:
            i = argv.index(name)
            val = argv[i + 1]
            del argv[i:i + 2]
            if name == "--opt":
                opt_kind = val
            elif name == "--lr-shift":
                lr_shift = int(val)
            else:
                frac_bits = int(val)
    if opt_kind == "master":
        # IntegerMasterSGD: int32 sub-grid masters, exponent-aware integer SGD (no sign steps,
        # no accumulator); the trainer only needs .step, .acc (checkpointed), .stable
        from integer_master_sgd import IntegerMasterSGD

        class TrainerMasterSGD(IntegerMasterSGD):
            def __init__(self, params, param_bits=12, lr_bits=2, clip=1, weight_bits=None, momentum_bits=0):
                super().__init__(params, lr_shift, frac_bits, 8 if weight_bits is None else weight_bits)
                self.acc = self.masters
                self.stable, self.shift, self.momentum_bits = True, lr_shift, 0
                print("optimizer: IntegerMasterSGD lr_shift %d (eta 2^-%d), frac_bits %d" % (lr_shift, lr_shift, frac_bits), flush=True)

        M.BlockSGD = TrainerMasterSGD
        src_bits = None                      # split momentum does not apply
    if src_bits is not None:
        last = {}
        real_model = M.JaKoMT

        class CapturingJaKoMT(real_model):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                last["model"] = self

        def factory(params, param_bits=12, lr_bits=2, clip=1, weight_bits=None, momentum_bits=0):
            model = last["model"]
            assert params[0] is model.parameters()[0]
            return SplitMomentumSGD(params, groups_of(model), momentum_bits, src_bits,
                                    param_bits=param_bits, lr_bits=lr_bits, clip=clip,
                                    weight_bits=weight_bits)

        M.JaKoMT = CapturingJaKoMT
        M.BlockSGD = factory
    sys.argv = [sys.argv[0]] + argv
    return M.main()


if __name__ == "__main__":
    sys.exit(main())
