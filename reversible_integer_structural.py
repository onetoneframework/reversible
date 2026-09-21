"""Properties the three equivalence gates cannot see, because they compare two
paths of the same implementation against each other."""
import json
import torch
import reversible_integer_final as R

torch.set_num_threads(1)
out = {}

# 1. Grouped GEMM equals an explicitly block-structured dense layer.
def dense_from_blocks(layer, rows, cols):
    """Rebuild the dense (rows x cols) matrix a grouped layer is equivalent to."""
    g = layer.g
    W = torch.zeros((rows, cols), dtype=torch.int64)
    for c in range(g):
        rs = R.group_start(c, g, layer.phase)
        cs = R.group_start(c, g, layer.phase)
        W[rs::g, cs::g] = layer.Ug[c] if rows == layer.cfg.hidden else layer.Vg[c]
    return W

ok = True
detail = []
for g in (2, 4, 8):
    for hidden in (64, 256):
        for phase in range(g):
            R.set_config(groups=g, hidden=hidden, perm_mode="default", phase_step=0)
            gen = torch.Generator().manual_seed(7)
            L = R.Coupling(gen, R.CFG, phase=phase)
            a = torch.randint(-50, 51, (16, R.M), dtype=torch.int64)
            z_grouped = L._gmm(a, L.Ug, hidden)
            Ud = dense_from_blocks(L, hidden, R.M)
            z_dense = a @ Ud.T
            same_u = bool(torch.equal(z_grouped, z_dense))
            h = torch.randint(0, 40, (16, hidden), dtype=torch.int64)
            f_grouped = L._gmm(h, L.Vg, R.M)
            Vd = dense_from_blocks(L, R.M, hidden)
            f_dense = h @ Vd.T
            same_v = bool(torch.equal(f_grouped, f_dense))
            ok = ok and same_u and same_v
            if not (same_u and same_v):
                detail.append({"g": g, "hidden": hidden, "phase": phase,
                               "U_ok": same_u, "V_ok": same_v})
out["grouped_gemm_equals_block_dense"] = {"passed": ok, "mismatches": detail}

# 2. Transpose permutation disperses every residue group across all g groups.
disp = {}
for g in (2, 4, 8):
    R.set_config(groups=g, perm_mode="transpose")
    p = R._transpose_perm(g)
    # column j of the next layer reads column p[j]; group of an index is i % g
    reach = {c: set() for c in range(g)}
    for j in range(R.M):
        reach[int(p[j]) % g].add(j % g)
    full = all(len(reach[c]) == g for c in range(g))
    disp[f"transpose_g{g}"] = {"every_group_reaches_all": full,
                               "reach_sizes": {c: len(v) for c, v in reach.items()},
                               "is_bijection": sorted(p[:R.M].tolist()) == list(range(R.M))}
    # the default permutation, for contrast
    q = R.ODDEVEN[:R.M]
    reach_d = {c: set() for c in range(g)}
    for j in range(R.M):
        reach_d[int(q[j]) % g].add(j % g)
    disp[f"default_g{g}"] = {"every_group_reaches_all": all(len(reach_d[c]) == g for c in range(g)),
                             "reach_sizes": {c: len(v) for c, v in reach_d.items()}}
out["permutation_dispersion"] = disp

# 3. Parameter and MAC counts are exactly 1/g. MAC per sample == parameter count.
counts = {}
for g, hidden in ((1, 64), (4, 64), (8, 256), (8, 64)):
    R.set_config(groups=g, hidden=hidden, perm_mode="default")
    layers = R.make_layers(16, 0)
    n = sum(w.numel() for w in R.parameters(layers))
    counts[f"g{g}_H{hidden}"] = n
base = counts["g1_H64"]
out["parameter_counts"] = counts
out["parameter_ratio_vs_dense"] = {k: base / v for k, v in counts.items()}

# 4. phase_step actually changes the wiring (not silently ignored).
R.set_config(groups=4, phase_step=1, perm_mode="transpose")
ls = R.make_layers(4, 0)
out["phase_step_applied"] = [L.phase for L in ls]
R.set_config(groups=4, phase_step=0, perm_mode="transpose")
out["phase_step_zero"] = [L.phase for L in R.make_layers(4, 0)]

R.set_config()
print(json.dumps(out, indent=2, default=str))
json.dump(out, open("reversible_integer_structural_checks.json", "w"), indent=2, default=str)
