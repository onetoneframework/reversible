"""Float ablation ladder on the trainer's split: which integer-path constraint
costs the generalisation gap?

Uses reversible_jako_bisect.MT, the float mirror with the integer path's
constraints as flags: pow2 (7-bit power-of-two row rescale of every linear
output), attention softmax / exp2 (the integer base-2 13-level forward with
softmax's backward) / relu (no softmax at all), the token hinge instead of
cross-entropy, and the BlockSGD replica (per-tensor renorm, error-feedback
accumulator, +-1/16 quantum on a [-8, 8) grid = the int8 grid) instead of
Adam. Same data and validation report as float_ceiling.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import reversible_jako_data as D
import reversible_jako_mt as M
from reversible_jako_bisect import MT, hinge

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--dim", type=int, default=192)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--loss", choices=("ce", "hinge", "ww"), default="hinge",
                help="ww = Weston-Watkins hinge: every class violating the margin gets +1, the true class -1 per violator")
ap.add_argument("--margin", type=float, default=1.0)
ap.add_argument("--ce2", action="store_true",
                help="base-2 cross-entropy: p_i = 2**floor((l_i - max) / 2**k); gradient p/Z - onehot (straight-through)")
ap.add_argument("--ce2-shift", type=int, default=0, help="k: temperature shift of the base-2 softmax")
ap.add_argument("--ce2-pow2z", action="store_true",
                help="normalise by the power of two >= Z instead of Z (shift-only; measured: collapses)")
ap.add_argument("--ce2-cap", type=int, default=0,
                help="classes more than cap*2**k below the row max get p = 0 (the integer head's cap); 0 = no cap")
ap.add_argument("--attn", choices=("softmax", "exp2", "relu"), default="softmax")
ap.add_argument("--pow2", type=int, default=1)
ap.add_argument("--opt", choices=("adam", "blocksgd"), default="adam")
ap.add_argument("--momentum-bits", type=int, default=0, help="blocksgd replica: m -= m/2**k; m += g; vote = renorm(m)")
ap.add_argument("--eval-rows", type=int, default=400)
ap.add_argument("--max-tok", type=int, default=14)
ap.add_argument("--max-len", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.set_flush_denormal(True)      # base-2 probabilities reach 2**-100 and below; denormals stall AVX
torch.manual_seed(a.seed)
dev = torch.device(a.device)
MAX_LEN = a.max_len
ds = D.prepare("both", 0, 5000, MAX_LEN, seed=0, valid_frac=0.05, max_tok=a.max_tok, tok_mode="char",
               base_pairs=0, min_char_count=20)
tok = ds["tokenizer"]
vocab = ds["info"]["vocab_size"]
tr = tuple(torch.tensor(c, dtype=torch.long) for c in ds["train"][:3])
va = tuple(torch.tensor(c, dtype=torch.long) for c in ds["valid"][:3])
vpairs = ds["valid_pairs"]
cfg = {"softmax": a.attn != "relu", "pow2": bool(a.pow2), "exp2": a.attn == "exp2",
       "quantw": False, "sgd": False, "name": "%s_%s_pow2%d_%s" % (a.loss, a.attn, a.pow2, a.opt)}
model = MT(vocab, a.dim, 2 * a.dim, "af", "ccaf", MAX_LEN, cfg).to(dev)
params = list(model.parameters())
opt = torch.optim.Adam(params, lr=a.lr) if a.opt == "adam" else None
SHIFT, QUANT = 12, 1.0 / 16.0
acc = [torch.zeros_like(p) for p in params]
mom = [torch.zeros_like(p) for p in params] if a.momentum_bits else None
n = tr[0].shape[0]
print(json.dumps({"train": n, "valid": va[0].shape[0], "params": sum(p.numel() for p in params), "dim": a.dim,
                  "cfg": cfg, "loss": a.loss, "opt": a.opt, "momentum_bits": a.momentum_bits, "lr": a.lr,
                  "device": a.device}), flush=True)


@torch.no_grad()
def blocksgd_step():
    """reversible_jako_bisect rung 8: renorm_tensor(g, 12) -> acc -> +-1 quantum, clip 1."""
    for i, p in enumerate(params):
        if p.grad is None:
            continue
        g = p.grad
        mx = g.abs().max()
        if mx > 0:
            g = g * (2.0 ** 11 / mx)
        if mom is not None:
            mom[i].sub_(mom[i] / (2.0 ** a.momentum_bits)).add_(g)
            g = mom[i]
            mx = g.abs().max()
            if mx > 0:
                g = g * (2.0 ** 11 / mx)
        acc[i].sub_(g)
        q = torch.sign(acc[i]) * torch.floor(acc[i].abs() / 2 ** SHIFT)
        q = q.clamp(-1, 1)
        p.add_(q * QUANT).clamp_(-8.0, 8.0 - QUANT)
        acc[i].sub_(q * 2 ** SHIFT)
        p.grad = None


@torch.no_grad()
def greedy(src):
    B = src.shape[0]
    mem = model.encode(src)
    cur = torch.full((B, MAX_LEN), D.PAD, dtype=torch.long, device=dev)
    cur[:, 0] = D.BOS
    out = torch.full((B, MAX_LEN), D.PAD, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for t in range(MAX_LEN):
        nxt = model.Wout(model.decode(cur, mem, src))[:, t].argmax(dim=1)
        nxt = torch.where(done, torch.full_like(nxt, D.PAD), nxt)
        out[:, t] = nxt
        done = done | (nxt == D.EOS)
        if t + 1 < MAX_LEN:
            cur[:, t + 1] = torch.where(done, torch.full_like(nxt, D.PAD), nxt)
        if bool(done.all()):
            break
    return out.cpu()


@torch.no_grad()
def validate():
    model.eval()
    keep_tot = corr = 0
    for st in range(0, va[0].shape[0], 128):
        s, ti, to = (x[st:st + 128].to(dev) for x in va)
        pred = model(s, ti).argmax(-1)
        keep = to != D.PAD
        keep_tot += int(keep.sum()); corr += int(((pred == to) & keep).sum())
    m = min(a.eval_rows, va[0].shape[0])
    hyps, refs = [], []
    for st in range(0, m, 64):
        gen = greedy(va[0][st:st + 64].to(dev))
        for i in range(gen.shape[0]):
            if st + i < m:
                hyps.append(M.ids_to_text(tok, gen[i])); refs.append(vpairs[st + i][1])
    exact = sum(1 for h, r in zip(hyps, refs) if h.strip() == r.strip())
    model.train()
    return {"valid_token_acc": 100.0 * corr / max(1, keep_tot), "chrf": M.chrf(hyps, refs), "bleu": M.bleu(hyps, refs),
            "exact": exact, "rows": m, "samples": [(vpairs[i][0], refs[i], hyps[i]) for i in range(3)]}


hist, best, t0 = [], -1.0, time.time()
gen = torch.Generator().manual_seed(a.seed)
for ep in range(a.epochs):
    order = torch.randperm(n, generator=gen)
    tot_loss = tot_tok = 0.0
    for st in range(0, n, a.batch):
        bi = order[st:st + a.batch]
        s, ti, to = (x[bi].to(dev) for x in tr)
        logits = model(s, ti)
        if a.ce2:
            Bq, Tq, Vq = logits.shape
            f = logits.reshape(Bq * Tq, Vq)
            t = to.reshape(-1)
            keep = t != D.PAD
            f, t = f[keep], t[keep]
            with torch.no_grad():
                e2 = torch.floor((f - f.amax(dim=1, keepdim=True)) / (2.0 ** a.ce2_shift))
                if a.ce2_cap:
                    # the integer head: classes at or above max - cap keep 2**e2, the rest are exactly 0
                    e2 = torch.clamp(e2, min=-float(a.ce2_cap) - 1.0)
                    p = torch.where(e2 >= -float(a.ce2_cap), torch.pow(2.0, e2), torch.zeros_like(e2))
                else:
                    p = torch.pow(2.0, e2)
                z = p.sum(dim=1, keepdim=True)
                if a.ce2_pow2z:
                    z = torch.pow(2.0, torch.ceil(torch.log2(z)))
                p = p / z
                y = torch.zeros_like(p).scatter_(1, t[:, None], 1.0)
                g = (p - y) / f.shape[0]
            loss = (f * g).sum()                                   # gradient = p/Z - onehot, straight-through
            with torch.no_grad():
                loss_log = -torch.log(p.gather(1, t[:, None]).clamp(min=1e-12)).mean()
        elif a.loss == "ce":
            loss = F.cross_entropy(logits.transpose(1, 2), to, ignore_index=D.PAD)
        elif a.loss == "ww":
            Bq, Tq, Vq = logits.shape
            f = logits.reshape(Bq * Tq, Vq)
            t = to.reshape(-1)
            keep = t != D.PAD
            f, t = f[keep], t[keep]
            viol = torch.clamp(f + a.margin - f.gather(1, t[:, None]), min=0)
            viol = viol.scatter(1, t[:, None], 0.0)
            loss = viol.sum(1).mean()
        else:
            loss = hinge(logits, to)
        if opt is not None:
            opt.zero_grad()
        loss.backward()
        if opt is not None:
            opt.step()
        else:
            blocksgd_step()
        k = int((to != D.PAD).sum()); tot_loss += float(loss_log if a.ce2 else loss.detach()) * k; tot_tok += k
    v = validate()
    row = {"epoch": ep, "train_loss": tot_loss / max(1, tot_tok), "minutes": round((time.time() - t0) / 60, 1),
           **{k: v[k] for k in ("valid_token_acc", "chrf", "bleu", "exact", "rows")}}
    hist.append(row)
    print("  ep%-3d train_%s %.3f | valid tok %.2f%% | greedy chrF %.2f BLEU %.2f exact %d/%d | %.1fm" % (
        ep, a.loss, row["train_loss"], row["valid_token_acc"], row["chrf"], row["bleu"], row["exact"], row["rows"],
        row["minutes"]), flush=True)
    for ja, ref, hyp in v["samples"]:
        print("      JA %s | REF %s | GEN %s" % (ja, ref, hyp), flush=True)
    if v["chrf"] > best:
        best = v["chrf"]
        torch.save({"state_dict": model.state_dict(), "epoch": ep, "cfg": cfg, "dim": a.dim}, a.out.replace(".json", "_best.pt"))
    json.dump({"args": vars(a), "cfg": cfg, "history": hist, "best_chrf": best}, open(a.out, "w", encoding="utf-8"), indent=1)
