#!/usr/bin/env python3
"""
Diagnostic float mirror. Not a production model — its only job is to say which
part of the system is responsible for the failure at 128 pairs.

State of the evidence it is answering:

  32 pairs   greedy 32/32   (integer, with positional encoding)
  128 pairs  greedy 0/128   real 53.8 vs shuffled 51.0
  512 pairs  greedy 0/512   real 26.4 vs shuffled 25.2

So the integer stack can memorise 32 arbitrary sentence pairs but nothing
larger. That could be the integer arithmetic (gradient precision, shared
exponents, +-1 weight updates), or it could be the formulation the integer
stack happens to use — an unnormalised leaky-power-of-two score map instead of
softmax, and a token hinge instead of cross-entropy — which would fail in float
too.

Three variants, run on the identical pairs, tokenizer, sizes and teacher
forcing:

  mirror    float32 autograd, but every formulation choice copied from the
            integer model: additive reversible coupling, no normalisation
            layers, leaky-power-of-two score map with no softmax, token hinge
            loss with margin 1.
  softmax   mirror + scaled dot-product softmax attention.
  standard  softmax + cross-entropy.

Reading:
  mirror fails, standard passes    -> the formulation is the problem, and the
                                      integer arithmetic is not what is blocking
  mirror passes                    -> the formulation is fine in float, so the
                                      integer path (precision/optimizer) is
  all three fail                   -> the architecture itself (reversible
                                      coupling without normalisation) is

  python reversible_jako_floatdiag.py --variant mirror --pairs 128
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


class Attn(nn.Module):
    def __init__(self, d, dkv, cross, softmax):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(dkv, d, bias=False)
        self.Wv = nn.Linear(dkv, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.cross, self.softmax, self.d = cross, softmax, d

    def forward(self, a, ctx):
        kv = ctx["memory"] if self.cross else a
        mask = ctx["cross_mask"] if self.cross else ctx["self_mask"]
        q, k, v = self.Wq(a), self.Wk(kv), self.Wv(kv)
        s = q @ k.transpose(1, 2)
        if self.softmax:
            s = s / math.sqrt(self.d)
            s = s.masked_fill(mask == 0, float("-inf"))
            w = torch.softmax(s, dim=-1)
            w = torch.nan_to_num(w)
        else:
            # The integer model's score map: leaky power-of-two, no softmax,
            # mask applied multiplicatively afterwards.
            w = torch.where(s > 0, s, s / 8.0) * mask
        return self.Wo(w @ v)


class FFN(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.W1 = nn.Linear(d, h, bias=False)
        self.W2 = nn.Linear(h, d, bias=False)

    def forward(self, a, ctx):
        return self.W2(F.relu(self.W1(a)))


class RevStack(nn.Module):
    """[a,b] -> [a, b + F(a)] then swap halves, exactly as the integer model."""

    def __init__(self, d, h, pattern, softmax, dkv):
        super().__init__()
        fs = []
        for c in pattern:
            if c == "a":
                fs.append(Attn(d, d, False, softmax))
            elif c == "c":
                fs.append(Attn(d, dkv, True, softmax))
            else:
                fs.append(FFN(d, h))
        self.fs = nn.ModuleList(fs)
        self.d = d

    def forward(self, v, ctx):
        for f in self.fs:
            a, b = v[:, :, :self.d], v[:, :, self.d:]
            v = torch.cat([b + f(a, ctx), a], dim=2)
        return v


class FloatMT(nn.Module):
    def __init__(self, vocab, d, h, enc_p, dec_p, max_len, softmax):
        super().__init__()
        self.d, self.vocab = d, vocab
        self.emb_src = nn.Embedding(vocab, d)
        self.emb_tgt = nn.Embedding(vocab, d)
        self.pos_src = nn.Parameter(torch.zeros(max_len, d))
        self.pos_tgt = nn.Parameter(torch.zeros(max_len, d))
        nn.init.normal_(self.pos_src, std=0.02)
        nn.init.normal_(self.pos_tgt, std=0.02)
        self.enc = RevStack(d, h, enc_p, softmax, d)
        self.dec = RevStack(d, h, dec_p, softmax, 2 * d)
        self.Wout = nn.Linear(2 * d, vocab, bias=False)

    def masks(self, src, tgt):
        B, Ts = src.shape
        Tt = tgt.shape[1]
        ok = (src != PAD).long()
        enc_self = ok[:, None, :].expand(B, Ts, Ts)
        cross = ok[:, None, :].expand(B, Tt, Ts)
        causal = torch.tril(torch.ones(Tt, Tt, dtype=torch.long))[None].expand(B, Tt, Tt)
        return enc_self, causal, cross

    def embed(self, ids, table, pos):
        e = table(ids) + pos[:ids.shape[1]][None]
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


def hinge_loss(logits, tgt):
    """Token multiclass margin, the integer model's objective."""
    B, T, V = logits.shape
    flat = logits.reshape(B * T, V)
    t = tgt.reshape(-1)
    keep = t != PAD
    flat, t = flat[keep], t[keep]
    true_l = flat.gather(1, t[:, None]).squeeze(1)
    masked = flat.scatter(1, t[:, None], float("-inf"))
    best_w = masked.max(dim=1).values
    return torch.clamp(best_w + 1.0 - true_l, min=0).mean()


def ce_loss(logits, tgt):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           tgt.reshape(-1), ignore_index=PAD)


@torch.no_grad()
def greedy(model, src, max_len):
    B = src.shape[0]
    mem = model.encode(src)
    cur = torch.full((B, max_len), PAD, dtype=torch.long)
    cur[:, 0] = BOS
    out = torch.full((B, max_len), PAD, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    for t in range(max_len):
        lg = model.Wout(model.decode(cur, mem, src))
        nxt = lg[:, t].argmax(dim=1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        out[:, t] = nxt
        done = done | (nxt == EOS)
        if t + 1 < max_len:
            cur[:, t + 1] = torch.where(done, torch.full_like(nxt, PAD), nxt)
        if bool(done.all()):
            break
    return out


@torch.no_grad()
def evaluate(model, src, tin, tout, tok, refs, max_len, perm):
    lg = model(src, tin)
    pred = lg.argmax(-1)
    keep = tout != PAD
    tf = 100.0 * float(((pred == tout) & keep).sum()) / float(keep.sum())
    pos0 = 100.0 * float((pred[:, 0] == tout[:, 0]).sum()) / src.shape[0]
    lg_s = model(src[perm], tin)
    tf_s = 100.0 * float(((lg_s.argmax(-1) == tout) & keep).sum()) / float(keep.sum())
    gen = greedy(model, src, max_len)
    hyp = [tok.decode([v for v in gen[i].tolist() if v not in (PAD, BOS, EOS)])
           for i in range(gen.shape[0])]
    exact = sum(1 for h, r in zip(hyp, refs) if h.strip() == r.strip())
    return tf, tf_s, pos0, exact, hyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["mirror", "softmax", "standard"],
                    default="mirror")
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--report-every", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--ffn-hidden", type=int, default=192)
    ap.add_argument("--enc-pattern", default="af")
    ap.add_argument("--dec-pattern", default="ccaf")
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--max-tok", type=int, default=14)
    ap.add_argument("--vocab-size", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valid-frac", type=float, default=0.0,
                    help="hold out a split; >0 makes the report measure "
                         "generalisation instead of memorisation")
    ap.add_argument("--base-pairs", type=int, default=8000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out_path = a.out or "floatdiag_%s_%d.json" % (a.variant, a.pairs)

    ds = D.prepare("both", 0, a.vocab_size, a.max_len, seed=a.seed,
                   max_tok=a.max_tok, tok_mode="char", base_pairs=a.base_pairs,
                   min_char_count=20, valid_frac=(a.valid_frac or 0.05))
    tok = ds["tokenizer"]
    allp = [(x[0], x[1]) for x in ds["train_pairs"]]
    pairs = allp if (a.valid_frac > 0 and not a.pairs) else allp[:a.pairs]
    vpairs = [(x[0], x[1]) for x in ds["valid_pairs"]] if a.valid_frac > 0 else []
    src, tin, tout = [], [], []
    for ja, ko in pairs:
        s = tok.encode(ja).ids[:a.max_len - 1] + [EOS]
        t = tok.encode(ko).ids[:a.max_len - 1]
        src.append(s + [PAD] * (a.max_len - len(s)))
        tin.append([BOS] + t + [PAD] * (a.max_len - len(t) - 1))
        tout.append(t + [EOS] + [PAD] * (a.max_len - len(t) - 1))
    src = torch.tensor(src); tin = torch.tensor(tin); tout = torch.tensor(tout)
    refs = [ko for _j, ko in pairs]
    n = src.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1234))

    def encode_set(ps):
        S, I, O = [], [], []
        for ja, ko in ps:
            s_ = tok.encode(ja).ids[:a.max_len - 1] + [EOS]
            t_ = tok.encode(ko).ids[:a.max_len - 1]
            S.append(s_ + [PAD] * (a.max_len - len(s_)))
            I.append([BOS] + t_ + [PAD] * (a.max_len - len(t_) - 1))
            O.append(t_ + [EOS] + [PAD] * (a.max_len - len(t_) - 1))
        return torch.tensor(S), torch.tensor(I), torch.tensor(O)

    vsrc = vtin = vtout = vrefs = vperm = None
    if vpairs:
        vpairs = vpairs[:400]
        vsrc, vtin, vtout = encode_set(vpairs)
        vrefs = [ko for _j, ko in vpairs]
        vperm = torch.randperm(vsrc.shape[0],
                               generator=torch.Generator().manual_seed(99))

    use_softmax = a.variant in ("softmax", "standard")
    loss_fn = ce_loss if a.variant == "standard" else hinge_loss
    model = FloatMT(ds["info"]["vocab_size"], a.dim, a.ffn_hidden, a.enc_pattern,
                    a.dec_pattern, a.max_len, use_softmax)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(json.dumps({"variant": a.variant, "pairs": n, "softmax": use_softmax,
                      "loss": "cross_entropy" if a.variant == "standard" else "hinge",
                      "params": sum(p.numel() for p in model.parameters()),
                      "dim": a.dim, "lr": a.lr, "positional": True}), flush=True)
    print("\n%8s %10s %10s %10s %12s %8s" % (
        "step", "loss", "tf_real%", "tf_shuf%", "greedy", "min"), flush=True)

    g = torch.Generator().manual_seed(a.seed)
    hist = []
    t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.report_every == 0:
            model.eval()
            tf, tf_s, pos0, exact, hyp = evaluate(model, src, tin, tout, tok,
                                                  refs, a.max_len, perm)
            model.train()
            with torch.no_grad():
                l = float(loss_fn(model(src, tin), tout))
            row = {"step": step, "loss": round(l, 4), "tf": round(tf, 3),
                   "tf_shuffled": round(tf_s, 3), "pos0": round(pos0, 3),
                   "greedy_exact": exact,
                   "minutes": round((time.time() - t0) / 60, 1)}
            if vsrc is not None:
                # Held-out. Memorisation is not the question any more.
                model.eval()
                with torch.no_grad():
                    vlg = model(vsrc, vtin)
                    vk = vtout != PAD
                    vtf = 100.0 * float(((vlg.argmax(-1) == vtout) & vk).sum()) / float(vk.sum())
                    vtfs = 100.0 * float(((model(vsrc[vperm], vtin).argmax(-1) == vtout) & vk).sum()) / float(vk.sum())
                    vp0 = 100.0 * float((vlg.argmax(-1)[:, 0] == vtout[:, 0]).sum()) / vsrc.shape[0]
                vgen = greedy(model, vsrc, a.max_len)
                vhyp = [tok.decode([v for v in vgen[i].tolist() if v not in (PAD, BOS, EOS)])
                        for i in range(vsrc.shape[0])]
                import reversible_jako_probe as _P, reversible_jako_metrics as _MET
                pid = []
                for ja, _r in _P.PROBE:
                    e = tok.encode(ja).ids[:a.max_len - 1] + [EOS]
                    pid.append(e + [PAD] * (a.max_len - len(e)))
                pgen = greedy(model, torch.tensor(pid), a.max_len)
                phyp = [tok.decode([v for v in pgen[i].tolist() if v not in (PAD, BOS, EOS)])
                        for i in range(len(_P.PROBE))]
                anch = sum(1 for i, (ja, _r) in enumerate(_P.PROBE)
                           for trig, kos in _MET.LEXICAL_ANCHORS
                           if trig in ja and any(k in phyp[i] for k in kos))
                model.train()
                row.update({"valid_tf": round(vtf, 3), "valid_tf_shuffled": round(vtfs, 3),
                            "valid_gap": round(vtf - vtfs, 3), "valid_pos0": round(vp0, 3),
                            "valid_chrf": round(chrf_simple(vhyp, vrefs), 3),
                            "anchors": anch, "probe": phyp})
                print("%8d loss %8.4f | valid tf %6.3f%% shuf %6.3f%% gap %+6.3f "
                      "pos0 %5.2f%% chrF %5.2f anchors %d/8 | %5.1fm"
                      % (step, l, row["valid_tf"], row["valid_tf_shuffled"],
                         row["valid_gap"], row["valid_pos0"], row["valid_chrf"],
                         row["anchors"], row["minutes"]), flush=True)
            else:
                print("%8d %10.4f %9.3f%% %9.3f%% %8d/%-4d %8.1f" % (
                    step, l, tf, tf_s, exact, n, row["minutes"]), flush=True)
            hist.append(row)
            if exact >= n:
                print("all %d reproduced; stopping early" % n, flush=True)
                break
        if step == a.steps:
            break
        idx = torch.randint(0, n, (min(a.batch, n),), generator=g)
        loss = loss_fn(model(src[idx], tin[idx]), tout[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    best = max(h["greedy_exact"] for h in hist)
    print("\nbest greedy exact %d/%d" % (best, n))
    json.dump({"config": vars(a), "history": hist, "best_greedy": best,
               "pairs": n}, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
