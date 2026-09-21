"""Run equiv / implequiv / stepequiv across width and grouping configurations."""
import json, sys, itertools
import torch
import reversible_integer_final as R

torch.set_num_threads(1)

INT8 = dict(state_bits=12, weight_bits=8, accum_bits=32)
CONFIGS = [
    ("default (all options off)", {}),
    ("widths only g=1", INT8),
    ("widths g=1 state8", dict(INT8, state_bits=8)),
    ("g=4 transpose", dict(INT8, groups=4, perm_mode="transpose")),
    ("g=8 transpose H256", dict(INT8, groups=8, perm_mode="transpose", hidden=256)),
    ("g=8 default perm", dict(INT8, groups=8)),
    ("g=4 transpose phase_step=1", dict(INT8, groups=4, perm_mode="transpose", phase_step=1)),
    ("g=2 no widths", dict(groups=2, perm_mode="transpose")),
]

rows = []
ok = True
for name, cfg in CONFIGS:
    R.set_config(**cfg)
    hb = 8 if cfg.get("state_bits") else 9
    e = R.equivalence_test(8, 0, hb)
    i = R.implementation_equivalence_test(8, 0, hb)
    s = R.step_mode_equivalence_test(8, 0, hb, batches=10)
    passed = (
        e["network_inverse_exact"] and e["all_layer_inputs_equal"]
        and e["all_parameter_gradient_mantissas_equal"]
        and e["all_parameter_gradient_exponents_equal"]
        and e["final_input_gradient_mantissa_equal"]
        and e["final_input_gradient_exponent_equal"]
        and i["reconstructed_input_equal"] and i["all_parameter_gradients_equal"]
        and i["final_input_gradient_equal"]
        and s["weights_and_accumulators_identical_every_step"]
    )
    ok = ok and passed
    rows.append(dict(config=name, passed=passed, equiv=e, implequiv=i, stepequiv=s))
    print(f"[{'PASS' if passed else 'FAIL'}] {name:<32} "
          f"inverse={e['network_inverse_exact']} cached==recon={e['all_layer_inputs_equal']} "
          f"fused==unfused={i['all_parameter_gradients_equal']} "
          f"perlayer==batched={s['weights_and_accumulators_identical_every_step']}")
    if not passed:
        print("   ", json.dumps({"equiv": e, "implequiv": i, "stepequiv": s}))

json.dump(rows, open("reversible_integer_gate_matrix.json", "w"), indent=2)
print("ALL GATES PASS WITH OPTIONS ON:", ok)
sys.exit(0 if ok else 1)
