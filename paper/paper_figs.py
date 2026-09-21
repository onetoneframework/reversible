"""Figures for the revised manuscript, drawn from the measured values only."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

OUT = Path(__file__).resolve().parent / "paper_fig"
OUT.mkdir(exist_ok=True)
SCALE = Path(r"C:\Project\OnetoneFramework\framework\res\Platform\Python\reversible\blocksgd_scale")

rcParams.update({
    "font.family": "Segoe UI", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#6b7480",
    "axes.labelcolor": "#1b2129", "xtick.color": "#3d4652", "ytick.color": "#3d4652",
    "axes.grid": True, "grid.color": "#e1e5ea", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.2, "svg.fonttype": "none", "figure.dpi": 110,
})
INK, BLUE, GREEN, RED, GREY, ORANGE, PURPLE = "#1b2129", "#1f6f8b", "#2e7d4f", "#b23a3a", "#8a94a0", "#b7791f", "#6a4c93"


def save(fig, name):
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def hist(path):
    return json.load(open(path, encoding="utf-8"))["history"]


# ------------------------------------------------------------- Fig 2 digits
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
K = ["K=8", "K=16", "K=32"]
legacy, legacy_sd = [86.93, 13.60, 9.78], [3.83, 2.57, 0.79]
bfp, bfp_sd = [68.71, 72.27, 84.1], [8.78, 5.78, 2.4]
final, final_sd = [87.3, 86.7, 86.8], [2.2, 0.8, 1.0]
x = range(3); w = 0.26
a.bar([i - w for i in x], legacy, w, yerr=legacy_sd, color=GREY, capsize=3, label="legacy backward (mantissa only)")
a.bar(list(x), bfp, w, yerr=bfp_sd, color=BLUE, capsize=3, label="shared-exponent backward")
a.bar([i + w for i in x], final, w, yerr=final_sd, color=GREEN, capsize=3, label="final 8w/8g/12s configuration")
a.set_xticks(list(x)); a.set_xticklabels(K); a.set_ylim(0, 100); a.set_ylabel("digits test accuracy (%)")
a.set_title("(a) Depth: five-seed mean ± s.d.")
a.legend(loc="lower left", fontsize=7.6)
seeds = range(5)
b.plot([0] * 5, [16.44, 9.56, 14.89, 13.11, 14.00], "o", color=GREY, ms=7)
b.plot([1] * 5, [79.56, 65.78, 76.00, 72.67, 67.33], "o", color=BLUE, ms=7)
for s, (l, r) in enumerate(zip([16.44, 9.56, 14.89, 13.11, 14.00], [79.56, 65.78, 76.00, 72.67, 67.33])):
    b.plot([0, 1], [l, r], "-", color="#c9d0d8", lw=0.8, zorder=0)
b.set_xticks([0, 1]); b.set_xticklabels(["legacy", "shared-exponent"]); b.set_xlim(-0.5, 1.5); b.set_ylim(0, 100)
b.set_title("(b) K=16, the five seeds; only the backward changes")
b.axhline(10, color=RED, lw=0.8, ls="--"); b.text(1.45, 11.5, "chance", color=RED, ha="right", fontsize=7.5)
save(fig, "fig2_digits")

# ------------------------------------------------------ Fig 3 memory + FMNIST
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.0))
Ks = [8, 16, 32]
a.plot(Ks, [6.8, 13.6, 27.3], "s-", color=GREY, label="cached activation tape")
a.plot(Ks, [1.491456] * 3, "o-", color=BLUE, label="reconstructed working set")
for k, v in zip(Ks, [6.8, 13.6, 27.3]):
    a.annotate("%.1f MB" % v, (k, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5, color=GREY)
a.annotate("1.49 MB, constant", (32, 1.49), textcoords="offset points", xytext=(0, 7), ha="right", fontsize=7.5, color=BLUE)
a.set_xticks(Ks); a.set_xlabel("depth K"); a.set_ylabel("activation storage (MB)"); a.set_ylim(0, 30)
a.set_title("(a) Digits MLP activation memory vs depth"); a.legend(loc="upper left")
names = ["float Adam+CE", "integer dense", "integer g=4", "integer g=8", "g=8 roll/xor"]
acc = [80.74, 79.00, 79.34, 78.71, 73.44]; sd = [1.81, 1.29, 0.77, 1.98, 1.04]
cols = [GREY, BLUE, BLUE, BLUE, RED]
b.bar(range(5), acc, color=cols, yerr=sd, capsize=3, width=0.62)
for i, (v, p) in enumerate(zip(acc, ["--", "22.2M", "5.5M", "2.8M", "2.8M"])):
    b.text(i, 60.5, p, ha="center", fontsize=7.4, color="white", fontweight="bold")
b.set_xticks(range(5)); b.set_xticklabels(names, rotation=18, ha="right"); b.set_ylim(60, 84)
b.set_ylabel("Fashion-MNIST test accuracy (%)"); b.set_title("(b) Grouped couplings, 5 seeds, 2 epochs (params in bars)")
save(fig, "fig3_memory_fmnist")

# ------------------------------------------------- Fig 4 deferred debt (8k)
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.9))
lab = ["adaptive-DEFER", "fixed cap256", "TTL16", "debt erased"]
grz = [1.95, 3.38, 13.92, 21.17]
a.bar(range(4), grz, color=[RED, GREY, ORANGE, GREEN], width=0.6)
for i, v in enumerate(grz):
    a.text(i, v + 0.5, "%.2f" % v, ha="center", fontsize=8)
a.set_xticks(range(4)); a.set_xticklabels(lab, rotation=15, ha="right"); a.set_ylabel("G_RZ at step 64 (margin, full 400)")
a.set_title("(a) Retained deferred debt causally lowers source-memory margin")
lab2 = ["ep40 reference", "fixed cap256", "adaptive-DEFER", "TTL16"]
h = [23.4852, 22.4139, 21.0576, 25.3865]
b.bar(range(4), h, color=[GREY, GREY, BLUE, ORANGE], width=0.6)
for i, v in enumerate(h):
    b.text(i, v + 0.15, "%.2f" % v, ha="center", fontsize=8)
b.set_xticks(range(4)); b.set_xticklabels(lab2, rotation=15, ha="right"); b.set_ylim(18, 27)
b.set_ylabel("train hinge after one epoch (ep42)"); b.set_title("(b) One-epoch confirmation: bounded debt ≠ better training")
save(fig, "fig4_debt")

# ------------------------------------------------------- Fig 6 512 trajectories
mom7 = hist(SCALE / "i512_mom7_full.json")
split = hist(SCALE / "i512_split_m7_s0.json")
ckpt = hist(SCALE / "i512_split_ckpt.json")
plain = [(0, 0.0, 0.0, 0), (150, 0.5211, 1.17, 0), (300, 0.3764, 0.39, 0), (450, 0.4922, 0.39, 0)]
fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.9))
series = [("plain BlockSGD (historical, stopped ep450)", plain, RED, "s", "--"),
          ("momentum 7, uniform", [(h["epoch"], h["tf_content_acc"], h["tf_pos0_acc"], h["greedy_exact"]) for h in mom7], GREEN, "o", "-"),
          ("split m7/s0, seed 0", [(h["epoch"], h["tf_content_acc"], h["tf_pos0_acc"], h["greedy_exact"]) for h in split], BLUE, "o", "-"),
          ("split m7/s0, seed 1 (to ep600)", [(h["epoch"], h["tf_content_acc"], h["tf_pos0_acc"], h["greedy_exact"]) for h in ckpt], PURPLE, "^", "-")]
for name, rows, c, m, ls in series:
    e = [r[0] for r in rows]
    ax[0].plot(e, [r[3] for r in rows], marker=m, ls=ls, color=c, ms=3.5, lw=1.4, label=name)
    ax[1].plot(e, [r[1] for r in rows], marker=m, ls=ls, color=c, ms=3.5, lw=1.4)
    ax[2].plot(e, [r[2] for r in rows], marker=m, ls=ls, color=c, ms=3.5, lw=1.4)
ax[0].axhline(256, color=GREY, lw=0.8, ls=":"); ax[0].text(1800, 262, "CASE A: 256", ha="right", fontsize=7.5, color=GREY)
ax[1].axhline(90, color=GREY, lw=0.8, ls=":"); ax[1].text(1800, 92, "CASE A: 90%", ha="right", fontsize=7.5, color=GREY)
ax[0].set_ylabel("greedy exact / 512"); ax[1].set_ylabel("TF content accuracy (%)"); ax[2].set_ylabel("position-0 accuracy (%)")
ax[0].set_title("(a) full-sequence reproduction"); ax[1].set_title("(b) teacher-forced content"); ax[2].set_title("(c) first target token")
for a_ in ax:
    a_.set_xlabel("epoch (16 steps each)"); a_.set_xlim(0, 1850)
ax[0].set_ylim(0, 512); ax[1].set_ylim(0, 100); ax[2].set_ylim(0, 100)
fig.legend(loc="lower center", ncol=4, fontsize=7.6, bbox_to_anchor=(0.5, -0.06))
save(fig, "fig6_512")

# ------------------------------------------------ Fig 7 128 / 256 small sets
cp128 = json.load(open(SCALE.parent / "cp_p128d96.json", encoding="utf-8"))["history"]
s128 = hist(SCALE / "i128_split_m7_s0.json")
p256 = hist(SCALE / "i256_base.json")
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.9))
a.plot([h["epoch"] for h in cp128], [h["greedy_exact"] for h in cp128], "s--", color=GREY, ms=3.5, label="plain BlockSGD, 3000 epochs")
a.plot([h["epoch"] for h in s128], [h["greedy_exact"] for h in s128], "o-", color=BLUE, ms=3.5, label="split m7/s0, 900 epochs")
a.axhline(128, color=GREY, lw=0.6, ls=":")
a.set_xlabel("epoch (4 steps each)"); a.set_ylabel("greedy exact / 128"); a.set_ylim(0, 132); a.set_title("(a) 128 pairs: same model, only the optimizer differs")
a.legend(loc="lower right")
b.plot([h["epoch"] for h in p256], [h["tf_content_acc"] for h in p256], "o-", color=GREY, ms=3.5, label="TF content (%)")
b.plot([h["epoch"] for h in p256], [h["tf_pos0_acc"] for h in p256], "^-", color=ORANGE, ms=3.5, label="position-0 (%)")
b2 = b.twinx(); b2.plot([h["epoch"] for h in p256], [h["greedy_exact"] for h in p256], "s--", color=BLUE, ms=3.5, label="greedy exact")
b2.set_ylabel("greedy exact / 256", color=BLUE); b2.set_ylim(0, 40); b2.grid(False); b2.spines["top"].set_visible(False)
b.set_ylim(0, 100); b.set_xlabel("epoch (8 steps each)"); b.set_ylabel("%"); b.set_title("(b) 256 pairs, plain BlockSGD: slow and oscillating")
h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels(); b.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.4)
save(fig, "fig7_small")

# ---------------------------------------------------- Fig 8 mechanism panels
fire = json.load(open(SCALE / "epoch_firing_diag.json", encoding="utf-8"))
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
metrics = ["TF content", "position-0", "real source", "shuffled", "zeroed"]
plain256 = [8.31, 12.89, 6.94, 0.23, 0.69]; mom256 = [23.04, 1.17, 22.69, 18.52, 31.48]
x = range(5); w = 0.36
a.bar([i - w / 2 for i in x], plain256, w, color=GREY, label="plain BlockSGD")
a.bar([i + w / 2 for i in x], mom256, w, color=GREEN, label="momentum 7 (uniform)")
a.set_xticks(list(x)); a.set_xticklabels(metrics, rotation=15, ha="right"); a.set_ylabel("% at epoch 150, 256 pairs")
a.set_title("(a) Uniform momentum: faster LM, source pathway suppressed"); a.legend(fontsize=7.4)
cfgs = ["512@b32_lr1", "128@b8_lr1", "128@b32_lr1", "512@b128_lr1", "512@b32_lr3"]
labels = ["512 pairs\nbatch 32", "128 pairs\nbatch 8", "128 pairs\nbatch 32", "512 pairs\nbatch 128", "512 pairs\nlr_bits 3"]
groups = [("enc_self", BLUE), ("dec_cross", RED), ("dec_self", ORANGE)]
w = 0.26
for j, (g, c) in enumerate(groups):
    vals = [sum(r["moved_pct"][g] for r in fire[k]["rows"]) / 3 for k in cfgs]
    b.bar([i + (j - 1) * w for i in range(5)], vals, w, color=c, label=g)
b.set_xticks(range(5)); b.set_xticklabels(labels, fontsize=7.6); b.set_ylabel("weights moved per epoch (%)")
b.set_title("(b) Per-epoch displacement from init (3-epoch mean)"); b.legend(fontsize=7.4)
save(fig, "fig8_mechanism")

# ------------------------------------------------- Fig 9 gradient precision
ref = json.load(open(SCALE / "grad_precision_ref.json", encoding="utf-8"))["init"]["rows"]
hbs = [16, 20, 24, 28, 32]
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.9))
for g, c in [("enc_self", BLUE), ("dec_cross", RED), ("dec_self", ORANGE), ("dec_ffn", GREEN)]:
    a.plot(hbs, [ref["hb%d_acc32" % h]["groups"][g]["cosine"] for h in hbs], "o-", color=c, ms=4, label=g)
a.set_xticks(hbs); a.set_xlabel("hidden_bits (adjoint mantissa target), accum_bits = 32"); a.set_ylabel("cosine vs unsaturated reference")
a.set_ylim(-0.05, 1.05); a.set_title("(a) Gradient direction vs mantissa width"); a.legend(fontsize=7.4)
sat = [ref["hb%d_acc32" % h]["saturations"] for h in hbs]
b.bar(range(5), [max(s, 0.5) for s in sat], color=[GREEN if s == 0 else RED for s in sat], width=0.6)
b.set_yscale("log"); b.set_ylim(0.5, 5e6); b.set_xticks(range(5)); b.set_xticklabels(["hb%d" % h for h in hbs])
for i, s in enumerate(sat):
    b.text(i, max(s, 0.5) * 1.6, "{:,}".format(s), ha="center", fontsize=7.6)
b.set_ylabel("saturated backward accumulators / batch"); b.set_title("(b) Why the wide 'reference' was wrong")
save(fig, "fig9_gradient")

# ------------------------------------------------------- Fig 10 67k + averaging
ep = list(range(26))
allt = [12.462, 14.770, 14.878, 15.131, 16.519, 18.810, 19.098, 19.225, 19.116, 19.170, 20.577, 21.010, 21.623, 22.453,
        24.635, 25.392, 23.246, 26.041, 26.078, 22.867, 22.615, 23.120, 23.174, 23.210, 22.146, 22.200]
cont = [3.006, 4.581, 5.239, 5.611, 6.098, 7.157, 7.415, 8.474, 8.846, 9.648, 9.963, 9.677, 10.908, 9.562,
        12.683, 13.227, 10.392, 14.515, 14.171, 9.505, 10.822, 10.135, 9.734, 9.505, 10.707, 8.617]
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
a.plot(ep, allt, "o-", color=GREY, ms=3.2, label="all-token accuracy")
a.plot(ep, cont, "o-", color=BLUE, ms=3.2, label="content accuracy")
a.axvspan(13.5, 25.5, color="#e3eff4", zorder=0); a.text(19.5, 27.6, "resumed from epoch 13", ha="center", fontsize=7.5, color=BLUE)
a.axvspan(15.5, 18.5, color="#cfe3d8", zorder=0); a.text(17, 2.0, "averaged", ha="center", fontsize=7.5, color=GREEN)
a.set_xlabel("epoch (2,089 steps each)"); a.set_ylabel("validation accuracy (%)"); a.set_ylim(0, 30); a.set_title("(a) 66,853 pairs, d192, 10-bit weights, base-2 CE head")
a.legend(loc="center left", fontsize=7.4)
names = ["ep7", "ep12", "ep17", "mean\nep16–18", "mean\nep14–18", "mean\nep14–25", "float,\nsame head", "float CE\nceiling"]
chrf = [5.85, 6.60, 7.08, 7.86, 7.39, 5.10, 9.8, 11.3]
cols = [GREY, GREY, GREY, GREEN, GREEN, RED, PURPLE, PURPLE]
b.bar(range(8), chrf, color=cols, width=0.62)
for i, v in enumerate(chrf):
    b.text(i, v + 0.15, "%.2f" % v if v < 9 else "%.1f" % v, ha="center", fontsize=7.6)
b.set_xticks(range(8)); b.set_xticklabels(names, fontsize=7.2); b.set_ylabel("held-out greedy chrF (400 rows)"); b.set_ylim(0, 13)
b.set_title("(b) Integer checkpoint averaging")
save(fig, "fig10_67k")

# ------------------------------------------------------- Fig 11 memorisation vs unseen
fig, a = plt.subplots(figsize=(8.2, 2.6))
names = ["128 pairs\nplain, ep3000", "512 pairs\nsplit, ep300", "512 pairs\nsplit, ep450", "512 pairs\nmom7, ep1800", "67k pairs\nmean ep16–18"]
train = [89.229, 54.214, 59.607, 95.098, None]; unseen = [2.991, 5.752, 4.906, 5.054, 7.86]
w = 0.36
a.bar([i - w / 2 for i in range(5) if train[i] is not None], [t for t in train if t is not None], w, color=BLUE, label="training pairs")
a.bar([i + w / 2 for i in range(5)], unseen, w, color=ORANGE, label="never-trained pairs (same corpus)")
for i, u in enumerate(unseen):
    a.text(i + w / 2, u + 1.5, "%.1f" % u, ha="center", fontsize=7.6)
for i, t in enumerate(train):
    if t is not None:
        a.text(i - w / 2, t + 1.5, "%.1f" % t, ha="center", fontsize=7.6)
a.set_xticks(range(5)); a.set_xticklabels(names, fontsize=7.8); a.set_ylabel("greedy chrF"); a.set_ylim(0, 105)
a.set_title("Memorisation is not translation: chrF on trained vs never-trained sentences"); a.legend(loc="upper left", fontsize=7.6)
save(fig, "fig11_unseen")
print("figures written to", OUT)
