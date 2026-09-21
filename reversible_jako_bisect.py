#!/usr/bin/env python3
"""
Bisect the gap between the float model that works and the integer model that
does not.

Established: a float mirror with softmax attention reproduces 128/128 training
pairs by greedy decoding in 1000 Adam steps. The integer model reproduces 0/64
under every attention variant tried — row-mass normalisation at four targets,
score resolution from 7 to 20 bits, non-negative and leaky score maps. Guessing
which difference matters has failed six times, so this stops guessing and walks
the difference one step at a time.

Each rung takes the float model and puts one integer behaviour back:

  0 float      softmax, float weights, Adam, full precision          (known good)
  1 nosoftmax  the leaky power-of-two score map instead of softmax
  2 pow2       + pow2norm after each projection and after A@V
  3 quantw     + weights rounded to the 8-bit integer grid each step
  4 hinge      + the token hinge loss                     (already known good)
  5 sgd        + a plain +-1 sign update instead of Adam

The first rung that drops greedy reproduction is the mechanism. Everything is
float arithmetic throughout, so this isolates *which behaviour* costs the
model its generation, not whether integers are representable.

  python reversible_jako_bisect.py --rung 1 --pairs 64
"""

import argparse
import json
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_jako_data as D

PAD, BOS, EOS = D.PAD, D.BOS, D.EOS

# Which integer behaviours each rung switches on.
RUNGS = {
    0: dict(softmax=True,  pow2=False, quantw=False, sgd=False, name="float"),
    1: dict(softmax=False, pow2=False, quantw=False, sgd=False, name="nosoftmax"),
    2: dict(softmax=False, pow2=True,  quantw=False, sgd=False, name="pow2"),
    3: dict(softmax=False, pow2=True,  quantw=True,  sgd=False, name="quantw"),
    4: dict(softmax=False, pow2=True,  quantw=True,  sgd=True,  name="sgd"),
    5: dict(softmax=True,  pow2=True,  quantw=False, sgd=False, name="softmax+pow2"),
    # Separate softmax's forward shape from its backward coupling.
    #   6: softmax forward, but the normaliser is detached so the backward has
    #      no cross terms (-w_i w_j). If this fails, the coupling is what matters.
    #   7: the integer base-2 forward (power-of-two weights, row max subtracted),
    #      with softmax's full backward substituted through a straight-through
    #      surrogate. If this passes, the forward shape is fine and only the
    #      integer backward needs the coupling.
    6: dict(softmax=True,  pow2=True,  quantw=False, sgd=False, name="softmax_diagbwd", diag=True),
    7: dict(softmax=True,  pow2=True,  quantw=False, sgd=False, name="exp2fwd_softbwd", exp2=True),
    # Faithful float replica of BlockSGD: per-tensor renormalisation of the
    # gradient to a fixed scale (the integer path's 12-bit block), an
    # error-feedback accumulator, and a +-1 quantum step with clip 1, on float
    # weights with softmax attention. Isolates the optimizer at scale.
    8: dict(softmax=True,  pow2=True,  quantw=False, sgd=False, name="blocksgd_replica", blocksgd=True),
}
TB = 5


def pow2norm_f(x, bits, dim=-1):
    """Straight-through power-of-two rescale, the float image of pow2norm.

    Forward divides by the power of two that brings the row max to `bits`, which
    is what the integer path does; the backward is that same constant, so this
    changes the values the model sees without changing what a gradient means.
    """
    mx = x.abs().amax(dim=dim, keepdim=True)
    e = torch.floor(torch.log2(mx.clamp(min=1e-12))) - bits + 1
    s = torch.pow(2.0, torch.clamp(e, min=0)).detach()
    q = x / s
    return q + (torch.floor(q) - q).detach()


class Attn(nn.Module):
    def __init__(self, d, dkv, cross, cfg):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(dkv, d, bias=False)
        self.Wv = nn.Linear(dkv, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.cross, self.cfg, self.d = cross, cfg, d

    def forward(self, a, ctx):
        kv = ctx["memory"] if self.cross else a
        mask = ctx["cross_mask"] if self.cross else ctx["self_mask"]
        q, k, v = self.Wq(a), self.Wk(kv), self.Wv(kv)
        if self.cfg["pow2"]:
            q, k, v = (pow2norm_f(t, TB + 2) for t in (q, k, v))
        s = q @ k.transpose(1, 2)
        if self.cfg["softmax"]:
            s = s / math.sqrt(self.d)
            sm = s.masked_fill(mask == 0, float("-inf"))
            if self.cfg.get("diag"):
                # exp(s - max) / Z with Z detached: forward is exactly softmax,
                # backward is the diagonal approximation, no -w_i w_j terms.
                e = torch.exp(sm - sm.amax(dim=-1, keepdim=True).detach())
                w = e / e.sum(dim=-1, keepdim=True).detach()
            elif self.cfg.get("exp2"):
                # Integer base-2 forward in float: 2**(floor((s-max)/2**k)+cap),
                # normalised to sum 1 so the scale matches softmax. Backward
                # borrowed from softmax via a straight-through surrogate.
                soft = torch.softmax(sm, dim=-1)
                k, cap = 3.0, 12.0
                e2 = torch.floor((sm - sm.amax(dim=-1, keepdim=True)) / k) + cap
                hard = torch.where(mask == 0, torch.zeros_like(e2),
                                   torch.pow(2.0, torch.clamp(e2, min=0)))
                hard = hard / hard.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                w = hard.detach() + (soft - soft.detach())
            else:
                w = torch.softmax(sm, dim=-1)
            w = torch.nan_to_num(w)
        else:
            if self.cfg["pow2"]:
                s = pow2norm_f(s * mask, TB + 2)
            w = torch.where(s > 0, s, s / 8.0) * mask
        o = w @ v
        if self.cfg["pow2"]:
            o = pow2norm_f(o, TB + 2)
        o = self.Wo(o)
        return pow2norm_f(o, TB + 2) if self.cfg["pow2"] else o


class FFN(nn.Module):
    def __init__(self, d, h, cfg):
        super().__init__()
        self.W1 = nn.Linear(d, h, bias=False)
        self.W2 = nn.Linear(h, d, bias=False)
        self.cfg = cfg

    def forward(self, a, ctx):
        h = F.relu(self.W1(a))
        if self.cfg["pow2"]:
            h = pow2norm_f(h, TB + 2)
        o = self.W2(h)
        return pow2norm_f(o, TB + 2) if self.cfg["pow2"] else o


class RevStack(nn.Module):
    def __init__(self, d, h, pattern, dkv, cfg):
        super().__init__()
        fs = []
        for c in pattern:
            fs.append(Attn(d, d, False, cfg) if c == "a" else
                      Attn(d, dkv, True, cfg) if c == "c" else FFN(d, h, cfg))
        self.fs = nn.ModuleList(fs)
        self.d = d

    def forward(self, v, ctx):
        for f in self.fs:
            a, b = v[:, :, :self.d], v[:, :, self.d:]
            v = torch.cat([b + f(a, ctx), a], dim=2)
        return v


class MT(nn.Module):
    def __init__(self, vocab, d, h, enc_p, dec_p, max_len, cfg):
        super().__init__()
        self.d, self.cfg = d, cfg
        self.emb_src = nn.Embedding(vocab, d)
        self.emb_tgt = nn.Embedding(vocab, d)
        self.pos_src = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.pos_tgt = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.enc = RevStack(d, h, enc_p, d, cfg)
        self.dec = RevStack(d, h, dec_p, 2 * d, cfg)
        self.Wout = nn.Linear(2 * d, vocab, bias=False)

    def masks(self, src, tgt):
        B, Ts = src.shape
        Tt = tgt.shape[1]
        ok = (src != PAD).long()
        return (ok[:, None, :].expand(B, Ts, Ts),
                torch.tril(torch.ones(Tt, Tt, dtype=torch.long))[None].expand(B, Tt, Tt),
                ok[:, None, :].expand(B, Tt, Ts))

    def embed(self, ids, tab, pos):
        e = tab(ids) + pos[:ids.shape[1]][None]
        return torch.cat([e, torch.zeros_like(e)], dim=2)

    def encode(self, src):
        es, _c, _x = self.masks(src, src)
        return self.enc(self.embed(src, self.emb_src, self.pos_src),
                        {"self_mask": es, "cross_mask": None, "memory": None})

    def decode(self, tin, mem, src):
        _e, causal, cross = self.masks(src, tin)
        return self.dec(self.embed(tin, self.emb_tgt, self.pos_tgt),
                        {"self_mask": causal, "cross_mask": cross, "memory": mem})

    def forward(self, src, tin):
        return self.Wout(self.decode(tin, self.encode(src), src))


def hinge(logits, tgt):
    B, T, V = logits.shape
    f = logits.reshape(B * T, V)
    t = tgt.reshape(-1)
    keep = t != PAD
    f, t = f[keep], t[keep]
    true_l = f.gather(1, t[:, None]).squeeze(1)
    best_w = f.scatter(1, t[:, None], float("-inf")).max(dim=1).values
    return torch.clamp(best_w + 1.0 - true_l, min=0).mean()


@torch.no_grad()
def quantise_(model, bits=8):
    """Round every weight onto the signed 8-bit grid, as the integer model holds."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    for p in model.parameters():
        p.copy_(p.mul(16.0).round().clamp(lo, hi).div(16.0))


@torch.no_grad()
def greedy(model, src, max_len):
    B = src.shape[0]
    mem = model.encode(src)
    cur = torch.full((B, max_len), PAD, dtype=torch.long)
    cur[:, 0] = BOS
    out = torch.full((B, max_len), PAD, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    for t in range(max_len):
        nxt = model.Wout(model.decode(cur, mem, src))[:, t].argmax(dim=1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        out[:, t] = nxt
        done = done | (nxt == EOS)
        if t + 1 < max_len:
            cur[:, t + 1] = torch.where(done, torch.full_like(nxt, PAD), nxt)
        if bool(done.all()):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--report-every", type=int, default=250)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ffn-hidden", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    cfg = RUNGS[a.rung]

    ds = D.prepare("both", 0, 5000, a.max_len, seed=a.seed, max_tok=14,
                   tok_mode="char", base_pairs=8000, min_char_count=20)
    tok = ds["tokenizer"]
    pairs = [(p[0], p[1]) for p in ds["train_pairs"][:a.pairs]]
    src, tin, tout = [], [], []
    for ja, ko in pairs:
        s = tok.encode(ja).ids[:a.max_len - 1] + [EOS]
        t = tok.encode(ko).ids[:a.max_len - 1]
        src.append(s + [PAD] * (a.max_len - len(s)))
        tin.append([BOS] + t + [PAD] * (a.max_len - len(t) - 1))
        tout.append(t + [EOS] + [PAD] * (a.max_len - len(t) - 1))
    src, tin, tout = (torch.tensor(x) for x in (src, tin, tout))
    refs = [k for _j, k in pairs]
    n = src.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1234))

    model = MT(ds["info"]["vocab_size"], a.dim, a.ffn_hidden, "af", "ccaf",
               a.max_len, cfg)
    opt = (torch.optim.SGD(model.parameters(), lr=1.0) if cfg["sgd"]
           else torch.optim.Adam(model.parameters(), lr=a.lr))
    if cfg.get("blocksgd"):
        # Integer model: weights on a 1/16 grid in [-8, 8]; one quantum = 1/16.
        # acc -= g_renorm (max |g| -> 2**11); q = clip(acc >> shift, +-1);
        # w += q * quantum; acc -= q << shift. shift = 11 + lr_bits(1) = 12.
        acc = [torch.zeros_like(p_) for p_ in model.parameters()]
        SHIFT, QUANT = 12, 1.0 / 16.0
    print(json.dumps({"rung": a.rung, "name": cfg["name"], "pairs": n, **cfg}),
          flush=True)
    print("\n%7s %10s %9s %9s %10s %6s" %
          ("step", "loss", "tf_real%", "tf_shuf%", "greedy", "min"), flush=True)

    g = torch.Generator().manual_seed(a.seed)
    hist, t0 = [], time.time()
    for step in range(a.steps + 1):
        if step % a.report_every == 0:
            model.eval()
            with torch.no_grad():
                lg = model(src, tin)
                keep = tout != PAD
                tf = 100.0 * float(((lg.argmax(-1) == tout) & keep).sum()) / float(keep.sum())
                lgs = model(src[perm], tin)
                tfs = 100.0 * float(((lgs.argmax(-1) == tout) & keep).sum()) / float(keep.sum())
                loss = float(hinge(lg, tout))
            gen = greedy(model, src, a.max_len)
            hyp = [tok.decode([v for v in gen[i].tolist() if v not in (PAD, BOS, EOS)])
                   for i in range(n)]
            ex = sum(1 for h, r in zip(hyp, refs) if h.strip() == r.strip())
            model.train()
            hist.append({"step": step, "loss": round(loss, 4), "tf": round(tf, 3),
                         "tf_shuffled": round(tfs, 3), "greedy_exact": ex,
                         "minutes": round((time.time() - t0) / 60, 1)})
            print("%7d %10.4f %8.3f%% %8.3f%% %7d/%-4d %6.1f" %
                  (step, loss, tf, tfs, ex, n, hist[-1]["minutes"]), flush=True)
            if ex >= n:
                print("all %d reproduced; stopping early" % n, flush=True)
                break
        if step == a.steps:
            break
        idx = torch.randint(0, n, (min(a.batch, n),), generator=g)
        loss = hinge(model(src[idx], tin[idx]), tout[idx])
        opt.zero_grad()
        loss.backward()
        if cfg.get("blocksgd"):
            with torch.no_grad():
                for p, ac in zip(model.parameters(), acc):
                    if p.grad is None:
                        continue
                    gr = p.grad
                    mx = gr.abs().max()
                    if mx > 0:
                        gr = gr * (2.0 ** 11 / mx)         # renorm_tensor(g, 12)
                    ac.sub_(gr)
                    q = torch.sign(ac) * torch.floor(ac.abs() / 2 ** SHIFT)
                    q = q.clamp(-1, 1)
                    p.add_(q * QUANT).clamp_(-8.0, 8.0 - QUANT)
                    ac.sub_(q * 2 ** SHIFT)
            opt.zero_grad()
        elif cfg["sgd"]:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(-torch.sign(p.grad) / 16.0)
            opt.zero_grad()
        else:
            opt.step()
        if cfg["quantw"]:
            quantise_(model)

    best = max(h["greedy_exact"] for h in hist)
    print("\nrung %d (%s) best greedy %d/%d" % (a.rung, cfg["name"], best, n))
    json.dump({"rung": a.rung, "name": cfg["name"], "config": vars(a),
               "history": hist, "best_greedy": best, "pairs": n},
              open("bisect_r%d_p%d.json" % (a.rung, n), "w"), indent=2)


if __name__ == "__main__":
    main()
