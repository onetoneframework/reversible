"""Emit every result table of the project as HTML fragments, read from the run logs / JSON files.
Writes compendium_tables.json: {name: html}. No number is typed by hand here."""

import json
import re
from pathlib import Path

ROOT = Path(r"C:\Project\OnetoneFramework\framework\res\Platform\Python\reversible")
SCALE = ROOT / "blocksgd_scale"
LAP = ROOT / "generalize" / "runs_laptop"
SP = Path(__file__).resolve().parent
T = {}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;")


def table(cols, rows, num_from=1, caption=None):
    h = ["<table><tr>" + "".join('<th class="%s">%s</th>' % ("n" if i >= num_from else "", esc(c)) for i, c in enumerate(cols)) + "</tr>"]
    for r in rows:
        h.append("<tr>" + "".join('<td class="%s">%s</td>' % ("n" if i >= num_from else "", esc(v)) for i, v in enumerate(r)) + "</tr>")
    h.append("</table>")
    if caption:
        h.append('<p class="cap">%s</p>' % caption)
    return "\n".join(h)


def f(v, d=2):
    if v is None or v == "":
        return "—"
    try:
        return ("%." + str(d) + "f") % float(v)
    except Exception:
        return str(v)


# ---------------------------------------------------------------- A. JSON histories
def hist_table(path, n, extra=()):
    d = json.load(open(path, encoding="utf-8"))
    H = d["history"]
    cols = ["epoch", "steps", "TF all %", "TF content %", "pos0 %", "greedy /%d" % n, "chrF", "char-BLEU", "real %", "shuffled %", "zeroed %", "real−shuf pp", "real−zero pp"] + list(extra)
    rows = []
    for h in H:
        ep = h["epoch"]
        base = [ep, ep * ((n + 31) // 32), f(h["tf_all_acc"], 3), f(h["tf_content_acc"], 4), f(h["tf_pos0_acc"]), h["greedy_exact"],
                f(h.get("greedy_chrf")), f(h.get("greedy_bleu")), f(h["real_content"], 4), f(h["shuffled_content"], 4), f(h["zeroed_content"], 4),
                f(h["real_content"] - h["shuffled_content"]), f(h["real_content"] - h["zeroed_content"])]
        for e in extra:
            if e == "saturations":
                base.append(h.get("accum_saturations", "—"))
            elif e == "|W| max":
                base.append(h.get("w_max_abs", "—"))
            elif e == "minutes":
                base.append(h.get("minutes", "—"))
        rows.append(base)
    return table(cols, rows), d


for name, path, n, extra in [
    ("i512_mom7_full", SCALE / "i512_mom7_full.json", 512, ()),
    ("i512_split_m7_s0", SCALE / "i512_split_m7_s0.json", 512, ("saturations", "|W| max", "minutes")),
    ("i512_split_ckpt", SCALE / "i512_split_ckpt.json", 512, ("saturations", "|W| max", "minutes")),
    ("i512_split_m7_s0_seed1", SCALE / "i512_split_m7_s0_seed1.json", 512, ("saturations", "|W| max", "minutes")),
    ("i512_split_m5_s0", SCALE / "i512_split_m5_s0.json", 512, ("saturations", "|W| max")),
    ("i512_split_m7_s3", SCALE / "i512_split_m7_s3.json", 512, ("saturations", "|W| max")),
    ("i512_w10", SCALE / "i512_w10.json", 512, ("saturations", "|W| max")),
    ("i512_tb7", SCALE / "i512_tb7.json", 512, ("saturations", "|W| max")),
    ("i128_split_m7_s0", SCALE / "i128_split_m7_s0.json", 128, ("saturations", "|W| max")),
    ("i256_base", SCALE / "i256_base.json", 256, ()),
    ("cp_p128d96", ROOT / "cp_p128d96.json", 128, ()),
]:
    if path.exists():
        T[name], _ = hist_table(path, n, extra)

# ---------------------------------------------------------------- B. harness logs (rows only)
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)/(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+(\d+)\s+(\d+))?\s+([\d.]+)m")


def log_table(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = ROW.match(line)
        if m:
            g = m.groups()
            rows.append([int(g[0]), g[1], g[2], g[3], "%s/%s" % (g[4], g[5]), g[8], g[9], g[10],
                         f(float(g[8]) - float(g[9])), f(float(g[8]) - float(g[10])), g[11] or "—", g[12] or "—", g[13]])
    return table(["epoch", "TF all %", "TF content %", "pos0 %", "greedy", "real %", "shuffled %", "zeroed %", "real−shuf", "real−zero", "saturations", "|W| max", "minutes"], rows)


for name in ["i512_b128", "i512_lr3", "i512_mass10", "i512_mom7_mass10", "n128_b8", "i256_mom7", "i512_hb32_SATURATED_killed"]:
    p = SCALE / (name + ".log")
    if p.exists():
        T[name] = log_table(p)

# ---------------------------------------------------------------- C. historical logs
HROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)/(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)m")
for name in ["_cp_p512d96", "_i512_lr2", "_i512_mom3", "_i512_mom5", "_i512_mom7", "_i512_hb24"]:
    p = ROOT / (name + ".log")
    rows = []
    for line in open(p, encoding="utf-8", errors="replace"):
        m = HROW.match(line)
        if m:
            g = m.groups()
            rows.append([int(g[0]), g[1], g[2], g[3], "%s/%s" % (g[4], g[5]), g[6], g[7], g[8], g[9], g[10], g[11]])
    T[name] = table(["epoch", "TF all %", "TF content %", "pos0 %", "greedy", "give1", "give3", "real %", "shuffled %", "zeroed %", "minutes"], rows)

BROW = re.compile(r"^\s*(\d+)\s+([\d.na]+)\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)/(\d+)\s+([\d.]+)")
for name in ["_bis512_adam", "_bis512_blocksgd"]:
    rows = []
    for line in open(ROOT / (name + ".log"), encoding="utf-8", errors="replace"):
        m = BROW.match(line)
        if m:
            g = m.groups()
            rows.append([int(g[0]), g[1], g[2], g[3], "%s/%s" % (g[4], g[5]), g[6]])
    T[name] = table(["step", "loss", "TF real %", "TF shuffled %", "greedy", "minutes"], rows)

# ---------------------------------------------------------------- D. master-SGD and centered probes
ms = json.load(open(SP / "master_sgd_memorization_results" / "audit.json", encoding="utf-8"))
rows = []
for h in ms["runs"]["128"]["history"]:
    o = h["optimizer"]
    rows.append([h["epoch"], h["steps"], f(h["tf_all_acc"], 3), f(h["tf_content_acc"], 4), f(h["tf_pos0_acc"]), h["greedy_exact"], f(h.get("greedy_chrf")), f(h.get("greedy_bleu")),
                 f(h["real_content"], 4), f(h["shuffled_content"], 4), f(h["zeroed_content"], 4),
                 f(100 * o["mean_master_changed_fraction_per_step"], 3), f(100 * o["mean_quantized_changed_fraction_per_step"], 4), o["master_int32_max_abs"], o["clip_events_total"], h["minutes"]])
T["master_sgd_128"] = table(["epoch", "steps", "TF all %", "TF content %", "pos0 %", "greedy /128", "chrF", "char-BLEU", "real %", "shuffled %", "zeroed %", "master changed %/step", "W changed %/step", "max |M|", "clips", "minutes"], rows)
c = ms["calibration"]; s = ms["calibration_sanity"]
T["master_sgd_calibration"] = table(["quantity", "value"], [
    ["nonzero gradient elements (first batch)", c["nonzero_elements"]], ["b = floor(log2|m|) + e + 12: min / median / p95", "%d / %d / %d" % (c["b_min"], c["b_median"], c["b95"])],
    ["LR_SHIFT = b95 − 6", c["resolved_LR_SHIFT"]], ["|ΔM| median / p90 / p95 / p99 / max", "%d / %d / %d / %d / %d" % (s["median_abs_delta_M"], s["p90"], s["p95"], s["p99"], s["max"])],
    ["nonzero ΔM fraction", f(100 * s["nonzero_delta_fraction"], 3) + " %"], ["first-step master changed", f(s["first_step_master_changed_pct"], 4) + " %"],
    ["first-step quantized W changed", f(s["first_step_quantized_changed_pct"], 4) + " %"], ["int32 overflow", s["int32_overflow"]],
    ["per-tensor gradient exponents", ", ".join(str(x) for x in c["per_tensor_exponents"])]], num_from=99)

cq = json.load(open(SP / "centered_quantizer_results" / "audit.json", encoding="utf-8"))
rows = []
for h in cq["runs"]["128"]["history"]:
    o, r = h["optimizer"], h["residual"]
    rows.append([h["epoch"], f(h["tf_all_acc"], 3), f(h["tf_content_acc"], 4), f(h["tf_pos0_acc"]), h["greedy_exact"], f(h["real_content"], 4), f(h["shuffled_content"], 4), f(h["zeroed_content"], 4),
                 f(100 * o["mean_quantized_changed_fraction_per_step"], 5), o["master_int32_max_abs"], f(r["mean"], 1), f(r["mean_abs"], 1), r["p05"], r["median"], r["p95"], f(100 * r["frac_abs_le_2048"], 2), h["minutes"]])
T["centered_128"] = table(["epoch", "TF all %", "TF content %", "pos0 %", "greedy /128", "real %", "shuffled %", "zeroed %", "W changed %/step", "max |M|", "R mean", "|R| mean", "R p05", "R median", "R p95", "|R|≤2048 %", "minutes"], rows)
pj = cq["first_step_projection"]; iso = cq["first_update_isolation"]
T["centered_projection"] = table(["quantity", "value"], [
    ["ΔM bit-identical to previous probe", iso["delta_M_bit_identical"]], ["M_new bit-identical", iso["M_new_bit_identical"]],
    ["ΔM distribution (median/p90/p95/p99/max)", "%s vs previous %s" % (iso["delta_distribution"], iso["previous_probe_distribution"])],
    ["master changed % (this / previous)", "%.10f / %.10f" % (iso["master_changed_pct"], iso["previous_master_changed_pct"])],
    ["OLD truncation projected W changed", f(pj["old_truncation_quantized_changed_pct"], 4) + " %"], ["NEW centered projected W changed", f(pj["new_centered_quantized_changed_pct"], 5) + " %"],
    ["OLD vs NEW differing coordinates", f(pj["old_vs_new_differing_coordinate_pct"], 4) + " %"], ["both changed / same direction / opposite", "%d / %s / %d" % (pj["both_changed_count"], pj["same_direction_fraction_of_both_changed"], pj["opposite_result_count"])]], num_from=99)

# ---------------------------------------------------------------- E. gradient diagnostics
GROUPS = ["emb_src", "emb_tgt", "pos", "enc_self", "enc_ffn", "dec_self", "dec_cross", "dec_ffn", "wout"]
ref = json.load(open(SCALE / "grad_precision_ref.json", encoding="utf-8"))
for st in ("init", "cp_p128d96"):
    rows = []
    for k, v in ref[st]["rows"].items():
        rows.append([k.replace("_", " / ").replace("hb", "hb ").replace("acc", "accum ")] + [v["saturations"]] + [
            ("%.2f / %s" % (v["groups"][g]["cosine"], f(v["groups"][g]["vote_mass_agree"])) if g in v["groups"] and v["groups"][g]["vote_mass_agree"] is not None else ("%.2f / —" % v["groups"][g]["cosine"] if g in v["groups"] else "—")) for g in GROUPS])
    T["grad_ref_" + st] = table(["hb / accum", "saturations"] + GROUPS, rows, caption="cell = cosine vs hb32/accum63 reference / BlockSGD 12-bit vote-mass sign agreement; state %s, 512 batch[0:32]; forward identical: %s" % (st, ref[st]["forward_identical"]))

dg = json.load(open(SCALE / "grad_precision_diag.json", encoding="utf-8"))
for st in ("init", "cp_p128d96"):
    for tag in ("512@b32", "512@b128", "128@b32"):
        d = dg["consensus"][st][tag]
        rows = [[g, f(d[g]["consensus_ratio"], 4), f(100 * d[g]["epoch_fire_frac"], 4), f(d[g]["top_vote_agree_with_sum"], 4) if d[g]["top_vote_agree_with_sum"] is not None else "—", d[g]["batches"]] for g in GROUPS if g in d]
        T["consensus_%s_%s" % (st, tag.replace("@", "_"))] = table(["group", "consensus ratio ‖Σv‖₁/Σ‖v‖₁", "epoch-sum ≥ 2¹² (%)", "top-vote agreement with sum", "batches"], rows)
# the invalid first diagnostic (hb40/acc32 as reference) is kept for the record
rows = []
for st in ("init", "cp_p128d96"):
    for hb in ("16", "24", "32"):
        r = dg["precision"][st][hb]
        rows.append([st, hb] + [f(r[g]["cosine"], 3) if g in r else "—" for g in GROUPS])
T["grad_diag_invalid"] = table(["state", "hb (accum 32)"] + GROUPS, rows, caption="Cosine against the INVALID hb40/accum32 'reference' (itself saturated). Retained only as the record of the retracted diagnostic.")

fire = json.load(open(SCALE / "epoch_firing_diag.json", encoding="utf-8"))
rows = []
for k, v in fire.items():
    for r in v["rows"]:
        rows.append([k, r["epoch"], r["hinge"], r["correct"]] + [f(r["moved_pct"][g], 3) for g in GROUPS])
T["epoch_firing"] = table(["config", "epoch", "hinge (first 128)", "correct", ] + ["%s moved %%" % g for g in GROUPS], rows, caption="Canonical BlockSGD from a common init; hinge0 = %d, correct0 = %d for every config." % (fire["512@b32_lr1"]["hinge0"], fire["512@b32_lr1"]["correct0"]))

ge = json.load(open(SCALE / "grad_exactness.json", encoding="utf-8"))
rows = []
for k, v in ge.items():
    rows.append([k, v["saturations_hb16_acc32"], v["hb16_acc32_equals_hb16_acc63_elementwise"], "%.7f" % v["cosine_vs_hb32_acc63"], f(v["max_relative_deviation"], 1), f(v["mean_relative_deviation"], 4),
                 v["nonzero_reference_elements"], f(100 * v["fraction_16bit_mantissa_is_truncated_image_of_32bit"], 2), f(100 * v["vote_sign_agreement"], 4), f(100 * v["vote_value_equal_fraction"], 3)])
T["grad_exactness"] = table(["state / batch", "saturations (hb16/acc32)", "hb16/acc32 ≡ hb16/acc63", "cosine vs hb32/acc63", "max rel. dev.", "mean rel. dev.", "nonzero ref elements", "16-bit = truncated 32-bit (%)", "vote sign agree (%)", "vote value equal (%)"], rows)

# ---------------------------------------------------------------- F. 67k integer logs
EPL = re.compile(r"^\s*ep(\d+)\s+all\s+([\d.]+)% content\s+([\d.]+)% bal\s+([\d.]+)% \| cont r/s/z\s+([\d.]+)/\s*([\d.]+)/\s*([\d.]+) gap ([+\-\d.]+) \| pos0 r/s/z\s+([\d.]+)/\s*([\d.]+)/\s*([\d.]+) \| eosFP\s+([\d.]+)% eos_sh\s+([\d.]+)% \| anchors (\d+)/8 blank (\d+) len ([\d.]+) kor (\d+)% rep ([\d.]+)%")


def int67k(path):
    rows = []; jl = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith('{"epoch"'):
            try:
                j = json.loads(line); jl[j["epoch"]] = j
            except Exception:
                pass
        m = EPL.match(line)
        if m:
            g = m.groups()
            rows.append([int(g[0])] + list(g[1:4]) + list(g[4:7]) + [g[7]] + list(g[8:11]) + [g[11], g[12], "%s/8" % g[13], g[14], g[15], g[16]])
    for r in rows:
        j = jl.get(r[0], {})
        r += [f(j.get("train_hinge_per_token"), 3), f(j.get("train_token_acc"), 2), f(j.get("param_grad_nonzero_pct"), 1), f(j.get("weights_moved_since_init_pct"), 1), j.get("accum_saturations", "—"), j.get("max_weight", "—"), j.get("state_wrap_frac", "—")]
    return table(["epoch", "valid all %", "content %", "balanced %", "real %", "shuffled %", "zeroed %", "gap pp", "pos0 real", "pos0 shuf", "pos0 zero", "eos FP %", "eos share %", "anchors", "len", "kor %", "rep %",
                  "train loss/token", "train tok %", "grad nonzero %", "weights moved %", "saturations", "max |W|", "state wrap"], rows)


for name in ["g67k_mom7", "g67k_split", "g67k_split_lr3", "g67k_d192_split", "g67k_d192_ce2", "g67k_d192_ce2x", "g67k_d192_master_ce2", "g67k_d192_master5_ce2", "g67k_d192_tb7_ce2",
             "g67k_d192_w10_ce2", "g67k_d192_w12_ce2", "g67k_d192_w10_uni_ce2", "g67k_d192_w10_ce2_r2", "g67k_d192_w10_ce2_r3lr2", "g67k_d192_w10_ce2_r4swa", "g67k_d192_w10_ce2_r5swa"]:
    T[name] = int67k(LAP / (name + ".log"))

# ---------------------------------------------------------------- G. 67k float logs
FL = re.compile(r"^\s*ep(\d+)\s+(train_\w+) ([\d.na]+) \| valid tok ([\d.]+)% \| greedy chrF ([\d.]+) BLEU ([\d.]+) exact (\d+)/(\d+) \| ([\d.]+)m")
for name in ["float_ceiling_d96", "float_ceiling_d192", "float_ceiling_d192_long", "F1_hinge_softmax_adam", "F2_hinge_exp2_adam", "F3_hinge_softmax_blocksgd", "F4_ww_softmax_adam", "F5_ce2_k0_exactZ", "F6_ce2_k3_exactZ", "F7_ce2_k0_pow2Z", "F8_ce2_exp2_blocksgd"]:
    p = LAP / (name + ".log")
    if not p.exists():
        continue
    hdr = open(p, encoding="utf-8", errors="replace").readline().strip()
    seen = set(); rows = []
    for line in open(p, encoding="utf-8", errors="replace"):
        m = FL.match(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); g = m.groups()
            rows.append([int(g[0]), g[1].replace("train_", ""), g[2], g[3], g[4], g[5], "%s/%s" % (g[6], g[7]), g[8]])
    T[name] = table(["epoch", "train loss", "value", "valid tok %", "greedy chrF", "char-BLEU", "exact", "minutes"], rows, caption=esc(hdr[:300]))

# ---------------------------------------------------------------- H. evaluation files
HO = re.compile(r"HELD-OUT \((\d+) rows\): chrF ([\d.]+)\s+BLEU ([\d.]+)\s+exact ([\d.]+)%\s+token_acc ([\d.]+)%\s+hinge/token ([\d.]+)")
rows = []
for p in sorted(LAP.glob("*.txt")):
    if "probe" in p.name:
        continue
    txt = open(p, encoding="utf-8", errors="replace").read()
    m = HO.search(txt); h = re.search(r"checkpoint (\S+)\s+epoch (\S+)", txt)
    if m:
        rows.append([p.name, h.group(1) if h else "—", h.group(2) if h else "—", ("401–%d" % (400 + int(m.group(1)))) if p.name.startswith("indep") else "1–%s" % m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)])
T["heldout_evals"] = table(["file", "checkpoint", "epoch field", "validation rows", "chrF", "char-BLEU", "exact %", "token acc %", "loss/token"], rows)

TR = re.compile(r"checkpoint (\S+)\s+epoch (\d+).*?TRAINING PAIRS: greedy exact (\d+)/(\d+)\s+chrF ([\d.]+)\s+BLEU ([\d.]+).*?HELD-OUT \(never trained, same corpus\): greedy exact (\d+)/(\d+)\s+chrF ([\d.]+)\s+BLEU ([\d.]+)", re.S)
rows = []
for p in sorted(SCALE.glob("translate_*.txt")):
    m = TR.search(open(p, encoding="utf-8", errors="replace").read())
    if m:
        g = m.groups(); rows.append([p.name, g[0], g[1], "%s/%s" % (g[2], g[3]), g[4], g[5], "%s/%s" % (g[6], g[7]), g[8], g[9]])
T["translate_tests"] = table(["file", "checkpoint", "epoch", "trained: exact", "trained chrF", "trained char-BLEU", "unseen: exact", "unseen chrF", "unseen char-BLEU"], rows)

(SP / "compendium_tables.json").write_text(json.dumps(T, ensure_ascii=False, indent=0), encoding="utf-8")
print(len(T), "tables:", ", ".join(sorted(T)))
