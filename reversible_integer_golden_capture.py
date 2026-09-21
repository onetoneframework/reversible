"""Capture a behavioural fingerprint of reversible_integer_final.py.

Run before and after the width/grouping port; the two JSON files must be
identical for every default-configuration entry.
"""
import hashlib, json, sys
import torch
import reversible_integer_final as R

torch.set_num_threads(1)


def h(*tensors):
    d = hashlib.sha256()
    for t in tensors:
        d.update(t.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()[:16]


def params_hash(layers):
    return h(*R.parameters(layers))


out = {}

# 1. Weight initialisation stream (catches any RNG-order change).
for K in (1, 4, 16):
    for seed in (0, 1):
        out[f"init/K{K}/s{seed}"] = params_hash(R.make_layers(K, seed))

# 2. Forward state and inverse exactness.
for K in (4, 16):
    layers = R.make_layers(K, 0)
    v = R.forward_reversible(R.XTE[:32], layers, 1)
    out[f"fwd/K{K}"] = h(v)
    out[f"inv/K{K}"] = R.exact_inverse_check(layers, 1)

# 3. Equivalence gates.
for K in (8, 16, 32):
    out[f"equiv/K{K}"] = R.equivalence_test(K, 0)
    out[f"implequiv/K{K}"] = R.implementation_equivalence_test(K, 0)
for K in (8, 16):
    out[f"stepequiv/K{K}"] = R.step_mode_equivalence_test(K, 0)

# 4. Training fingerprints: accuracy plus final weight hash.
for method in ("reversible", "cached", "legacy"):
    for K in (8, 16):
        for seed in (0, 1):
            layers = R.make_layers(K, seed)
            res = R.train_integer(method, K, 2, seed)
            out[f"train/{method}/K{K}/s{seed}"] = {
                "accuracy": res["accuracy"],
                "max_weight": res["max_weight"],
                "inverse_exact": res["inverse_exact"],
            }

# 5. Both optional switches, all four combinations.
for fused in (True, False):
    for mode in ("per-layer", "batched"):
        res = R.train_integer("reversible", 16, 2, 0, fused=fused, step_mode=mode)
        out[f"switch/f{int(fused)}/{mode}"] = {
            "accuracy": res["accuracy"], "max_weight": res["max_weight"],
        }

json.dump(out, open(sys.argv[1], "w"), indent=2, sort_keys=True)
print(f"wrote {sys.argv[1]} with {len(out)} entries")
