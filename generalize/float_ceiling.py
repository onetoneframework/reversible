"""Float Adam ceiling for the integer model's architecture on the trainer's split.

FloatMT (reversible_jako_floatdiag: the float mirror with softmax attention),
trained with Adam + cross-entropy on exactly the data the integer trainer sees
(same prepare() arguments -> same deterministic train/validation split).
Reports validation token accuracy and greedy chrF / BLEU / exact match every
epoch, so integer runs have a ceiling to be judged against. --device cuda runs
it on a GPU.
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
from reversible_jako_floatdiag import FloatMT

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--dim", type=int, default=96)
ap.add_argument("--k-enc", type=int, default=2, help="encoder pattern = 'af' * k_enc")
ap.add_argument("--k-dec", type=int, default=4, help="decoder pattern = 'ccaf' repeated to k_dec layers")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--eval-rows", type=int, default=400)
ap.add_argument("--max-tok", type=int, default=14)
ap.add_argument("--max-len", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
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
enc_p = "af" * (a.k_enc // 2) if a.k_enc >= 2 else "a"
dec_p = ("ccaf" * ((a.k_dec + 3) // 4))[:a.k_dec] if a.k_dec != 4 else "ccaf"
model = FloatMT(vocab, a.dim, 2 * a.dim, enc_p, dec_p, MAX_LEN, True).float().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
n = tr[0].shape[0]
print(json.dumps({"train": n, "valid": va[0].shape[0], "params": sum(p.numel() for p in model.parameters()),
                  "dim": a.dim, "enc": enc_p, "dec": dec_p, "lr": a.lr, "batch": a.batch, "device": a.device,
                  "max_tok": a.max_tok, "max_len": a.max_len}), flush=True)


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
            "exact": exact, "rows": m, "samples": [(vpairs[i][0], refs[i], hyps[i]) for i in range(6)]}


hist, best, t0 = [], -1.0, time.time()
gen = torch.Generator().manual_seed(a.seed)
for ep in range(a.epochs):
    order = torch.randperm(n, generator=gen)
    tot_loss = tot_tok = 0.0
    for st in range(0, n, a.batch):
        bi = order[st:st + a.batch]
        s, ti, to = (x[bi].to(dev) for x in tr)
        loss = F.cross_entropy(model(s, ti).transpose(1, 2), to, ignore_index=D.PAD)
        opt.zero_grad(); loss.backward(); opt.step()
        k = int((to != D.PAD).sum()); tot_loss += float(loss.detach()) * k; tot_tok += k
    v = validate()
    row = {"epoch": ep, "train_ce": tot_loss / max(1, tot_tok), "minutes": round((time.time() - t0) / 60, 1),
           **{k: v[k] for k in ("valid_token_acc", "chrf", "bleu", "exact", "rows")}}
    hist.append(row)
    print("  ep%-3d train_ce %.3f | valid tok %.2f%% | greedy chrF %.2f BLEU %.2f exact %d/%d | %.1fm" % (
        ep, row["train_ce"], row["valid_token_acc"], row["chrf"], row["bleu"], row["exact"], row["rows"], row["minutes"]), flush=True)
    for ja, ref, hyp in v["samples"][:3]:
        print("      JA %s | REF %s | GEN %s" % (ja, ref, hyp), flush=True)
    if v["chrf"] > best:
        best = v["chrf"]
        torch.save({"state_dict": model.state_dict(), "epoch": ep, "dim": a.dim, "enc": enc_p, "dec": dec_p,
                    "max_len": MAX_LEN}, a.out.replace(".json", "_best.pt"))
    json.dump({"args": vars(a), "history": hist, "best_chrf": best}, open(a.out, "w", encoding="utf-8"), indent=1)
