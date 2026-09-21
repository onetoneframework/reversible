"""Integer weight averaging of trainer checkpoints (Polyak-style, rounded to the grid).

usage: avg_ckpts.py OUT.pt IN1.pt IN2.pt ...
The output copies IN1's metadata and replaces "weights" by the per-element
round-half-away-from-zero mean of all inputs (integer arithmetic only).
"""

import sys

import torch

out, ins = sys.argv[1], sys.argv[2:]
cks = [torch.load(p, weights_only=False) for p in ins]
n = len(cks)
avg = []
for ws in zip(*[c["weights"] for c in cks]):
    s = torch.stack([w.to(torch.int64) for w in ws]).sum(0)
    q = torch.where(s >= 0, (2 * s + n) // (2 * n), -((2 * (-s) + n) // (2 * n)))
    avg.append(q)
base = dict(cks[0])
base["weights"] = avg
base["opt_acc"] = None
base["averaged_from"] = ins
torch.save(base, out)
moved = sum(int((a != w).sum()) for a, w in zip(avg, cks[0]["weights"]))
total = sum(a.numel() for a in avg)
print("averaged %d checkpoints -> %s ; %d/%d weights differ from %s" % (n, out, moved, total, ins[0]))
