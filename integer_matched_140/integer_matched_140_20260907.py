"""Matched integer control: the final integer stack on the float control's own task.

One cold-start trajectory, seed 0, 140 epochs x 475 updates = 66,500 updates, on the
saved 7,600/400 split the float+Adam train-fit control used. Nothing here implements
model, attention or optimizer arithmetic: those are imported from the canonical
modules and configured from the logged `jako_stage4_exp2` run, which is the latest
successful integer sequence configuration carrying exp2 attention with the
row-coupled adjoint.

Text and generation metrics are the control's own ControlMetrics, so the two runs are
scored by identical code. The teacher-forced objective is the integer hinge, not
cross entropy: integer logits have no probabilistic scale, and a float softmax has no
place here even in measurement.

Lives in a subdirectory on purpose. TrainFitControl hashes ROOT.glob("*.py") into the
float checkpoint's protected manifest; a sibling file would invalidate it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as numpy
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import reversible_integer_final as R
import reversible_arch as A
import reversible_jako_mt as M
import reversible_jako_data as data
from reversible_integer_final import BlockSGD
from float_adam_8k_control_20260906 import ControlMetrics

try:
	sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
	pass


class IntegerMatchedControl:
	"""One integer trajectory on the float control's data and update budget."""

	ROOT = ROOT
	FLOAT_SPLIT = ROOT / "float_adam_8k_control_20260906_results"
	FLOAT_TRAINFIT = ROOT / "float_adam_8k_trainfit_20260906_results"
	OUTPUT = HERE / "results"

	# --- matched to the float control -------------------------------------
	MAX_EPOCHS = 140
	BATCH = 16
	LENGTH = 16
	SEED = 0
	THREADS = 6
	TRAIN_COUNT = 7600
	VALID_COUNT = 400
	DIAGNOSTIC_COUNT = 400
	GREEDY_COUNT = 256
	DIAGNOSTIC_INTERVAL = 10
	MILESTONE_INTERVAL = 10

	# --- integer configuration, read from jako_stage4_exp2.json -----------
	SOURCE_CONFIG = ROOT / "jako_stage4_exp2.json"
	DIM = 96
	FFN_HIDDEN = 192
	K_ENC = 2
	K_DEC = 4
	ENC_PATTERN = ""          # k_enc=2 with no explicit pattern is exactly "af"
	DEC_PATTERN = "ccaf"
	POSITIONAL = 16
	STATE_BITS = 16
	WEIGHT_BITS = 8
	ACCUM_BITS = 32
	HIDDEN_BITS = 16
	PARAM_BITS = 12
	LR_BITS = 1
	MOMENTUM_BITS = 0
	CLIP = 1
	SCORE_EXP2 = 3
	SCORE_EXP2_CAP = 12
	SCORE_EXP2_COUPLED = True
	SCORE_LEAK = None
	ATTN_SCORE_BITS = 0
	ATTN_MASS_BITS = 0
	QUERY_ANCHOR = False

	STATE_NAME = "integer_last_full_state.pt"

	# ------------------------------------------------------------------ setup
	def configure(self) -> None:
		"""Install the audited integer globals before anything is constructed."""
		logged = json.loads(self.SOURCE_CONFIG.read_text(encoding="utf-8"))["config"]
		expected = {"dim": self.DIM, "ffn_hidden": self.FFN_HIDDEN, "k_enc": self.K_ENC, "k_dec": self.K_DEC,
			"enc_pattern": self.ENC_PATTERN, "dec_pattern": self.DEC_PATTERN, "positional": self.POSITIONAL,
			"state_bits": self.STATE_BITS, "weight_bits": self.WEIGHT_BITS, "accum_bits": self.ACCUM_BITS,
			"hidden_bits": self.HIDDEN_BITS, "lr_bits": self.LR_BITS, "score_exp2": self.SCORE_EXP2,
			"score_exp2_cap": self.SCORE_EXP2_CAP, "exp2_diag": not self.SCORE_EXP2_COUPLED,
			"score_leak": self.SCORE_LEAK, "attn_score_bits": self.ATTN_SCORE_BITS,
			"attn_mass_bits": self.ATTN_MASS_BITS, "query_anchor": self.QUERY_ANCHOR,
			"batch": self.BATCH, "max_len": self.LENGTH, "seed": self.SEED, "vocab_size": 5000}
		differing = {key: (logged.get(key), value) for key, value in expected.items() if logged.get(key) != value}
		if differing:
			raise ValueError("Integer configuration differs from the logged run: " + json.dumps(differing))
		self.logged_config = logged
		torch.set_num_threads(self.THREADS)
		A.SCORE_LEAK = self.SCORE_LEAK
		A.SCORE_EXP2 = self.SCORE_EXP2
		A.SCORE_EXP2_CAP = self.SCORE_EXP2_CAP
		A.SCORE_EXP2_COUPLED = self.SCORE_EXP2_COUPLED
		M.CROSS_QUERY_ANCHOR = self.QUERY_ANCHOR
		M.ATTN_SCORE_BITS = self.ATTN_SCORE_BITS
		M.ATTN_ROW_MASS_BITS = self.ATTN_MASS_BITS
		self.cfg = R.RunConfig(state_bits=self.STATE_BITS, weight_bits=self.WEIGHT_BITS, accum_bits=self.ACCUM_BITS)
		R.CFG = self.cfg

	def prepare(self) -> None:
		"""Load the float control's saved split and build the integer model."""
		self.configure()
		random.seed(self.SEED)
		numpy.random.seed(self.SEED)
		torch.manual_seed(self.SEED)
		self.generator = torch.Generator().manual_seed(self.SEED)

		self.float_audit = json.loads((self.FLOAT_SPLIT / "audit.json").read_text(encoding="utf-8"))
		self.dataset = json.loads((self.FLOAT_SPLIT / "split.json").read_text(encoding="utf-8"))
		if (len(self.dataset["train_pairs"]), len(self.dataset["valid_pairs"])) != (self.TRAIN_COUNT, self.VALID_COUNT):
			raise ValueError("Saved split counts differ")
		for name in ("train", "valid"):
			digest = hashlib.sha256(json.dumps(self.dataset[name + "_pairs"], ensure_ascii=False).encode("utf-8")).hexdigest()
			if digest != self.float_audit["split_sha256"][name]:
				raise ValueError("Saved split identity differs")
		training_keys = {tuple(row[:2]) for row in self.dataset["train_pairs"]}
		validation_keys = {tuple(row[:2]) for row in self.dataset["valid_pairs"]}
		if training_keys & validation_keys:
			raise ValueError("Exact JA-KO pair overlap")

		self.tokenizer_path = Path(data.CACHE) / self.float_audit["data"]["tokenizer"]
		self.tokenizer_sha256 = hashlib.sha256(self.tokenizer_path.read_bytes()).hexdigest()
		if self.tokenizer_sha256 != self.float_audit["tokenizer_sha256"]:
			raise ValueError("Tokenizer hash differs")
		self.tokenizer = data.load_tokenizer(str(self.tokenizer_path))
		self.vocab = self.tokenizer.get_vocab_size()
		if self.vocab != self.float_audit["data"]["vocab_size"]:
			raise ValueError("Tokenizer vocabulary differs")

		encoded_training = data.encode_pairs(self.tokenizer, self.dataset["train_pairs"], self.LENGTH)
		encoded_validation = data.encode_pairs(self.tokenizer, self.dataset["valid_pairs"], self.LENGTH)
		if (encoded_training[3], encoded_validation[3]) != (self.TRAIN_COUNT, self.VALID_COUNT):
			raise ValueError("Encoding dropped saved rows")
		self.training = tuple(torch.tensor(column) for column in encoded_training[:3])
		self.validation = tuple(torch.tensor(column) for column in encoded_validation[:3])

		self.model = M.JaKoMT(self.vocab, self.DIM, self.FFN_HIDDEN, self.K_ENC, self.K_DEC, self.cfg,
			self.SEED, enc_pattern=self.ENC_PATTERN, dec_pattern=self.DEC_PATTERN, positional=self.POSITIONAL)
		if (self.model.enc_pattern, self.model.dec_pattern) != (["attn", "ffn"], ["cross", "cross", "attn", "ffn"]):
			raise ValueError("Resolved block pattern does not match the float control topology")
		if self.model.pos_src is None or self.model.pos_tgt is None:
			raise ValueError("Positional tables are off")
		self.optimizer = BlockSGD(self.model.parameters(), param_bits=self.PARAM_BITS, lr_bits=self.LR_BITS,
			clip=self.CLIP, momentum_bits=self.MOMENTUM_BITS)
		if not self.optimizer.stable:
			raise ValueError("BlockSGD is not in the stable regime")
		if self.optimizer.mom is not None:
			raise ValueError("Momentum state exists but momentum must be off")

		# The float control's own fixed permutations, reused verbatim.
		trainfit_audit = json.loads((self.FLOAT_TRAINFIT / "audit.json").read_text(encoding="utf-8"))
		self.permutations = {name: torch.tensor(value, dtype=torch.int64)
			for name, value in trainfit_audit["permutations"].items()}
		if sorted(self.permutations["train"].tolist()) != list(range(self.DIAGNOSTIC_COUNT)):
			raise ValueError("Train diagnostic permutation is not a permutation of 0..399")
		if sorted(self.permutations["valid"].tolist()) != list(range(self.VALID_COUNT)):
			raise ValueError("Valid diagnostic permutation is not a permutation of 0..399")

		self.completed_epoch = 0
		self.global_updates = 0
		self.history = []
		self.OUTPUT.mkdir(parents=True, exist_ok=True)

		paths = sorted(self.ROOT.glob("*.py")) + [self.tokenizer_path, self.SOURCE_CONFIG,
			self.FLOAT_SPLIT / "split.json", self.FLOAT_SPLIT / "audit.json"]
		self.protected = {str(path.relative_to(self.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
			for path in paths if path.is_file()}
		self.audit = {
			"split_source": str(self.FLOAT_SPLIT / "split.json"), "train_count": self.TRAIN_COUNT,
			"valid_count": self.VALID_COUNT, "exact_pair_overlap": 0,
			"shared_japanese_sources": len({row[0] for row in self.dataset["train_pairs"]}
				& {row[0] for row in self.dataset["valid_pairs"]}),
			"tokenizer_path": str(self.tokenizer_path), "tokenizer_sha256": self.tokenizer_sha256,
			"vocab_size": self.vocab, "parameters": self.model.n_params(),
			"integer_configuration": {
				"source": str(self.SOURCE_CONFIG), "dimension": self.DIM, "ffn_hidden": self.FFN_HIDDEN,
				"encoder_pattern": self.model.enc_pattern, "decoder_pattern": self.model.dec_pattern,
				"positional": self.POSITIONAL, "state_bits": self.STATE_BITS, "weight_bits": self.WEIGHT_BITS,
				"accum_bits": self.ACCUM_BITS, "hidden_bits": self.HIDDEN_BITS, "param_bits": self.PARAM_BITS,
				"lr_bits": self.LR_BITS, "momentum_bits": self.MOMENTUM_BITS, "clip": self.CLIP,
				"score_exp2": self.SCORE_EXP2, "score_exp2_cap": self.SCORE_EXP2_CAP,
				"exp2_coupled_adjoint": self.SCORE_EXP2_COUPLED, "score_leak": self.SCORE_LEAK,
				"attn_score_bits": self.ATTN_SCORE_BITS, "attn_mass_bits": self.ATTN_MASS_BITS,
				"query_anchor": self.QUERY_ANCHOR, "optimizer": "BlockSGD", "optimizer_stable": self.optimizer.stable,
				"optimizer_shift": self.optimizer.shift, "batch": self.BATCH, "seed": self.SEED,
				"max_epochs": self.MAX_EPOCHS, "threads": self.THREADS, "torch": torch.__version__},
			"permutations": {name: value.tolist() for name, value in self.permutations.items()},
			"permutation_source": str(self.FLOAT_TRAINFIT / "audit.json"),
			"protected_sha256": self.protected,
			"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
		}

	# -------------------------------------------------------------- measurement
	def teacher_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
		"""ControlMetrics.teacher's counts, with the integer hinge as the objective."""
		predictions = logits.argmax(-1)
		valid = targets != data.PAD
		content = valid & (targets != data.EOS)
		eos = targets == data.EOS
		correct = predictions == targets
		_grad, stats = M.hinge_grad(logits, targets, self.HIDDEN_BITS)
		counts = {"tokens": int(valid.sum()), "content_tokens": int(content.sum()), "eos_tokens": int(eos.sum())}
		result = {
			**counts,
			"all_accuracy": 100.0 * float((correct & valid).sum()) / max(1, counts["tokens"]),
			"content_accuracy": 100.0 * float((correct & content).sum()) / max(1, counts["content_tokens"]),
			"hinge_per_token": stats["hinge_sum"] / max(1, stats["tokens"]),
			"hinge_sum": stats["hinge_sum"], "margin_active": stats["active"],
			"eos_recall": 100.0 * float((correct & eos).sum()) / max(1, counts["eos_tokens"]),
			"eos_false_positive_rate": 100.0 * float(((predictions == data.EOS) & content).sum()) / max(1, counts["content_tokens"]),
			"pos0_accuracy": 100.0 * float(correct[:, 0].double().mean()),
			"reference_mean_tokens": float(content.sum(1).double().mean()),
			"position_buckets": {},
		}
		for label, start, end in ControlMetrics.BUCKETS:
			mask = content[:, start:end]
			result["position_buckets"][label] = {"content_tokens": int(mask.sum()),
				"content_accuracy": 100.0 * float((correct[:, start:end] & mask).sum()) / max(1, int(mask.sum()))}
		return result

	def evaluate(self, columns, mode: str = "real", permutation=None, batch: int = 32) -> dict[str, Any]:
		"""Evaluate all rows with real, permuted, or zero encoder memory."""
		sources, inputs, targets = columns
		if mode == "shuffled":
			sources = sources[permutation]
		elif mode not in ("real", "zeroed"):
			raise ValueError("Unknown source diagnostic")
		parts = []
		for start in range(0, len(sources), batch):
			source = sources[start:start + batch]
			memory, _c, _i, _t = self.model.encode(source)
			if mode == "zeroed":
				memory = torch.zeros_like(memory)
			decoded, _c2, _i2, _t2 = self.model.decode(inputs[start:start + batch], memory, source)
			logits, _shift = self.model.logits(decoded)
			parts.append((len(source), self.teacher_metrics(logits, targets[start:start + batch])))
		result: dict[str, Any] = {"examples": len(sources)}
		for key in ("tokens", "content_tokens", "eos_tokens", "hinge_sum", "margin_active"):
			result[key] = sum(part[key] for _, part in parts)
		weights = {"all_accuracy": "tokens", "content_accuracy": "content_tokens", "hinge_per_token": "tokens",
			"eos_recall": "eos_tokens", "eos_false_positive_rate": "content_tokens",
			"pos0_accuracy": "examples", "reference_mean_tokens": "examples"}
		for key, weight in weights.items():
			result[key] = sum(part[key] * (count if weight == "examples" else part[weight])
				for count, part in parts) / max(1, result[weight])
		result["position_buckets"] = {}
		for label, _s, _e in ControlMetrics.BUCKETS:
			count = sum(part["position_buckets"][label]["content_tokens"] for _, part in parts)
			result["position_buckets"][label] = {"content_tokens": count,
				"content_accuracy": sum(part["position_buckets"][label]["content_accuracy"]
					* part["position_buckets"][label]["content_tokens"] for _, part in parts) / max(1, count)}
		return result

	def source_diagnostics(self, name: str) -> dict[str, Any]:
		"""The float control's fixed ablation, on the same fixed subsets."""
		columns = (tuple(column[:self.DIAGNOSTIC_COUNT] for column in self.training)
			if name == "train" else self.validation)
		permutation = self.permutations[name]
		real, shuffled, zeroed = (self.evaluate(columns, mode, permutation)
			for mode in ("real", "shuffled", "zeroed"))
		return {"real": real, "shuffled": shuffled, "zeroed": zeroed,
			"real_minus_shuffled": real["content_accuracy"] - shuffled["content_accuracy"],
			"real_minus_zeroed": real["content_accuracy"] - zeroed["content_accuracy"]}

	def greedy_evaluation(self, name: str, count: int, batch: int = 32) -> dict[str, Any]:
		"""Exact reproduction under the integer greedy decoder, scored as the control scores it."""
		columns = self.training if name == "train" else self.validation
		pairs = self.dataset[name + "_pairs"][:count]
		if len(pairs) != count:
			raise ValueError("Generation count exceeds saved split")
		generated_content, references, hypotheses = [], [], []
		exact_tokens = exact_strings = 0
		for start in range(0, count, batch):
			stop = min(start + batch, count)
			generated = M.greedy(self.model, columns[0][start:stop], self.LENGTH)
			for offset, row in enumerate(generated.tolist()):
				index = start + offset
				sequence = row[:row.index(data.EOS) + 1] if data.EOS in row else row
				target = columns[2][index].tolist()
				target = target[:target.index(data.EOS) + 1]
				exact_tokens += sequence == target
				content = [token for token in sequence if token not in (data.PAD, data.BOS, data.EOS)]
				hypothesis = self.tokenizer.decode(content)
				reference = self.tokenizer.decode(target[:-1])
				exact_strings += hypothesis == reference
				generated_content.append(content)
				hypotheses.append(hypothesis)
				references.append(pairs[index][1])
		metrics = ControlMetrics.generated(hypotheses, references, generated_content)
		metrics.update({"examples": count, "exact_token_sequences": exact_tokens,
			"exact_token_percent": 100.0 * exact_tokens / count, "exact_decoded_strings": exact_strings,
			"decoded_string_reference": "Tokenizer-decoded target tokens, excluding EOS",
			"exact_original_strings": sum(h == r for h, r in zip(hypotheses, references)),
			"generated_reference_character_ratio": metrics["mean_generated_characters"]
				/ max(1e-12, metrics["mean_reference_characters"])})
		return {"metrics": metrics, "outputs": hypotheses[:64]}

	# -------------------------------------------------------------- checkpoints
	def checkpoint(self) -> dict[str, Any]:
		return {
			"model": [w.clone() for w in self.model.parameters()],
			"optimizer_accumulators": [a.clone() for a in self.optimizer.acc],
			"optimizer_momentum": None,
			"completed_epoch": self.completed_epoch, "global_updates": self.global_updates,
			"python_rng": random.getstate(), "numpy_rng": numpy.random.get_state(),
			"torch_rng": torch.get_rng_state(), "data_order_rng": self.generator.get_state(),
			"diagnostic_permutations": self.permutations, "history": self.history,
			"metrics": self.history[-1] if self.history else {}, "audit": self.audit,
		}

	def save_checkpoint(self) -> None:
		pending = self.OUTPUT / "integer_pending_full_state.pt"
		torch.save(self.checkpoint(), pending)
		loaded = torch.load(pending, weights_only=False)
		if loaded["completed_epoch"] != self.completed_epoch or loaded["global_updates"] != self.global_updates:
			raise RuntimeError("Checkpoint epoch/update round-trip failed")
		if any(not torch.equal(a, b) for a, b in zip(loaded["model"], self.model.parameters())):
			raise RuntimeError("Checkpoint model round-trip failed")
		if any(not torch.equal(a, b) for a, b in zip(loaded["optimizer_accumulators"], self.optimizer.acc)):
			raise RuntimeError("Checkpoint accumulator round-trip failed")
		if loaded["optimizer_momentum"] is not None:
			raise RuntimeError("Momentum state present in checkpoint")
		if not all(name in loaded for name in ("python_rng", "numpy_rng", "torch_rng", "data_order_rng")):
			raise RuntimeError("Checkpoint RNG state incomplete")
		pending.replace(self.OUTPUT / self.STATE_NAME)
		if self.completed_epoch % self.MILESTONE_INTERVAL == 0:
			with (self.OUTPUT / ("integer_epoch_%03d.pt" % self.completed_epoch)).open("xb") as milestone:
				milestone.write((self.OUTPUT / self.STATE_NAME).read_bytes())

	# -------------------------------------------------------------------- gates
	def gates(self) -> dict[str, Any]:
		"""Run the existing gate suites under this run's exact integer configuration."""
		import reversible_jako_verify as V
		import reversible_jako_causalgate as C
		checks: dict[str, Any] = {}

		def passed(row):
			for key in ("passed", "ok", "exact"):
				if key in row:
					return bool(row[key])
			raise KeyError("gate result has no pass field: " + json.dumps(sorted(row)))

		g2 = V.gate2_inverse()
		checks["A_exact_inverse"] = {"passed": passed(g2), "detail": g2}
		g3 = V.gate3_cachefree()
		checks["B_cachefree_equals_cached"] = {"passed": passed(g3), "detail": g3}
		g4 = V.gate4_step_modes()
		checks["C_step_equivalence"] = {"passed": passed(g4), "detail": g4}
		# Gate 5's float mirror builds its oracle as Sf * (S > 0) * mask -- the
		# hard-ReLU score map, SCORE_EXP2=None and SCORE_LEAK=None. It has no
		# exp2 branch, so under SCORE_EXP2 it aborts while constructing the
		# oracle rather than while comparing a gradient. Run it in the
		# configuration it was written for, which still validates the shared
		# Wq/Wk/Wv/Wo adjoint, and check the exp2 coupling separately below.
		exp2, leak = A.SCORE_EXP2, A.SCORE_LEAK
		try:
			A.SCORE_EXP2, A.SCORE_LEAK = None, None
			g5 = V.gate5_attention_cosine()
		finally:
			A.SCORE_EXP2, A.SCORE_LEAK = exp2, leak
		checks["H1_attention_adjoint_cosine_relu_mirror"] = {"passed": passed(g5),
			"note": "gate 5 as written; its float mirror has no exp2 branch, so it runs "
				"under SCORE_EXP2=None/SCORE_LEAK=None and validates the shared "
				"Wq/Wk/Wv/Wo adjoint, not the exp2 score-weight adjoint",
			"detail": g5}

		# The coupled adjoint must actually be the live branch under exp2.
		generator = torch.Generator().manual_seed(11)
		scores = torch.randint(-4096, 4096, (3, 7, 7), dtype=torch.int64, generator=generator)
		mask = (torch.randint(0, 2, (3, 7, 7), dtype=torch.int64, generator=generator) | torch.eye(7, dtype=torch.int64))
		upstream = torch.randint(-(1 << 16), 1 << 16, (3, 7, 7), dtype=torch.int64, generator=generator)
		coupled_flag = A.SCORE_EXP2_COUPLED
		try:
			A.SCORE_EXP2_COUPLED = True
			coupled = A.exp2_attend_backward(upstream, scores, mask)
			A.SCORE_EXP2_COUPLED = False
			diagonal = A.exp2_attend_backward(upstream, scores, mask)
		finally:
			A.SCORE_EXP2_COUPLED = coupled_flag
		checks["H2_coupled_adjoint_live"] = {
			"passed": bool(A.SCORE_EXP2_COUPLED) and not torch.equal(coupled, diagonal),
			"configured_coupled": bool(A.SCORE_EXP2_COUPLED),
			"coupled_differs_from_diagonal": not torch.equal(coupled, diagonal),
			"differing_entries": int((coupled != diagonal).sum()), "entries": int(coupled.numel()),
			"note": "coupled minus diagonal must be nonzero or the row-coupling term is dead"}

		gate_model = C.build(600, self.DIM, self.FFN_HIDDEN, self.K_ENC, self.K_DEC, self.SEED,
			self.ENC_PATTERN or "af", self.DEC_PATTERN, self.POSITIONAL)
		cuts = {}
		for cut in (1, 3, 6):
			cuts[str(cut)] = C.gate_a(gate_model, 600, 12, cut)
		checks["D_causality"] = {"passed": all(
			not row.get("leaked") and row.get("decoder_state_before_cut_identical")
			and row.get("logits_before_cut_identical") and row.get("logits_after_cut_changed")
			for row in cuts.values()), "cuts": cuts}
		# gate_b probes padded lengths 10/16/24, so its model needs a positional
		# table that reaches 24; the trained run never exceeds 16.
		gb = C.gate_b(C.build(600, self.DIM, self.FFN_HIDDEN, self.K_ENC, self.K_DEC, self.SEED,
			self.ENC_PATTERN or "af", self.DEC_PATTERN, 24), 600)
		checks["E_pad_invariance"] = {"passed": passed(gb),
			"note": "gate model built with a 24-row positional table because gate_b pads to 24", "detail": gb}
		gc = C.gate_c(gate_model, 600, 12)
		checks["F_masked_memory_invariance"] = {"passed": passed(gc), "detail": gc}

		checks["G_positional_on"] = {"passed": self.model.pos_src is not None and self.model.pos_tgt is not None,
			"rows": int(self.model.max_pos)}
		checks["I_blocksgd_stable"] = {"passed": bool(self.optimizer.stable), "shift": self.optimizer.shift,
			"param_bits": self.optimizer.param_bits, "lr_bits": self.LR_BITS,
			"momentum_bits": self.optimizer.momentum_bits}

		# J. one batch through the real training path, with range telemetry.
		before = dict(R.GEMM_STATS)
		columns = tuple(column[:self.BATCH] for column in self.training)
		grads, stats = M.forward_backward(self.model, columns[0], columns[1], columns[2], self.HIDDEN_BITS, cached=False)
		limit = 1 << 62
		widest = max(int(pair[0].abs().max()) for pair in grads)
		checks["J_one_batch_range"] = {"passed": widest < limit and stats["tokens"] > 0,
			"tokens": stats["tokens"], "hinge_sum": stats["hinge_sum"], "margin_active": stats["active"],
			"max_gradient_mantissa": widest, "int64_headroom_bits": 62,
			"gemm_float64_calls": R.GEMM_STATS["float64"] - before["float64"],
			"gemm_int64_calls": R.GEMM_STATS["int64"] - before["int64"]}

		checks["all_passed"] = all(value["passed"] for key, value in checks.items() if key != "all_passed")
		return checks

	# --------------------------------------------------------------------- run
	def run(self) -> None:
		self.prepare()
		started_gates = time.monotonic()
		checks = self.gates()
		(self.OUTPUT / "gates.json").write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
		(self.OUTPUT / "audit.json").write_text(json.dumps(self.audit, indent=2), encoding="utf-8")
		print(json.dumps({"gates": {k: v["passed"] for k, v in checks.items() if k != "all_passed"},
			"all_passed": checks["all_passed"], "seconds": time.monotonic() - started_gates}, indent=2), flush=True)
		if not checks["all_passed"]:
			raise SystemExit("GATE FAILURE: training not started")

		# The gate batch above ran a real forward/backward; rebuild so the
		# trajectory starts from the untouched cold-start weights.
		self.model = M.JaKoMT(self.vocab, self.DIM, self.FFN_HIDDEN, self.K_ENC, self.K_DEC, self.cfg,
			self.SEED, enc_pattern=self.ENC_PATTERN, dec_pattern=self.DEC_PATTERN, positional=self.POSITIONAL)
		self.optimizer = BlockSGD(self.model.parameters(), param_bits=self.PARAM_BITS, lr_bits=self.LR_BITS,
			clip=self.CLIP, momentum_bits=self.MOMENTUM_BITS)
		random.seed(self.SEED)
		numpy.random.seed(self.SEED)
		torch.manual_seed(self.SEED)
		self.generator = torch.Generator().manual_seed(self.SEED)
		R.GEMM_STATS["float64"] = 0
		R.GEMM_STATS["int64"] = 0

		for epoch in range(1, self.MAX_EPOCHS + 1):
			started = time.monotonic()
			order = torch.randperm(self.TRAIN_COUNT, generator=self.generator)
			for start in range(0, len(order), self.BATCH):
				indices = order[start:start + self.BATCH]
				sources, inputs, targets = (column[indices] for column in self.training)
				grads, _stats = M.forward_backward(self.model, sources, inputs, targets, self.HIDDEN_BITS, cached=False)
				self.optimizer.step(grads)
				self.global_updates += 1
			self.completed_epoch = epoch
			row = {"epoch": epoch, "global_updates": self.global_updates,
				"train": self.evaluate(self.training), "valid": self.evaluate(self.validation)}
			if epoch % self.DIAGNOSTIC_INTERVAL == 0 or epoch == self.MAX_EPOCHS:
				row["train_source"] = self.source_diagnostics("train")
				row["valid_source"] = self.source_diagnostics("valid")
				row["train_greedy_256"] = self.greedy_evaluation("train", self.GREEDY_COUNT)
			row["seconds"] = time.monotonic() - started
			row["gemm"] = dict(R.GEMM_STATS)
			row["max_weight"] = max(int(w.abs().max()) for w in self.model.parameters())
			row["max_accumulator"] = max(int(a.abs().max()) for a in self.optimizer.acc)
			self.history.append(row)
			self.save_checkpoint()
			(self.OUTPUT / "history.json").write_text(json.dumps(self.history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
			source = row.get("train_source", {})
			print(json.dumps({"epoch": epoch, "updates": self.global_updates, "seconds": row["seconds"],
				"train_hinge": row["train"]["hinge_per_token"], "train_content": row["train"]["content_accuracy"],
				"train_pos0": row["train"]["pos0_accuracy"], "valid_content": row["valid"]["content_accuracy"],
				"real": source.get("real", {}).get("content_accuracy"),
				"shuffled": source.get("shuffled", {}).get("content_accuracy"),
				"zeroed": source.get("zeroed", {}).get("content_accuracy"),
				"greedy_256": row.get("train_greedy_256", {}).get("metrics", {}).get("exact_token_sequences")}), flush=True)

		train_greedy = self.greedy_evaluation("train", self.TRAIN_COUNT)
		valid_greedy = self.greedy_evaluation("valid", self.VALID_COUNT)
		final_train_source = self.source_diagnostics("train")
		final_valid_source = self.source_diagnostics("valid")
		final = self.history[-1]
		result = {"final_epoch": self.completed_epoch, "global_updates": self.global_updates,
			"final_train": final["train"], "final_valid": final["valid"],
			"train_greedy": train_greedy, "valid_greedy": valid_greedy,
			"final_train_source": final_train_source, "final_valid_source": final_valid_source,
			"gemm_calls": dict(R.GEMM_STATS), "gates": {k: v["passed"] for k, v in checks.items()},
			"full_checkpoint": str(self.OUTPUT / self.STATE_NAME), "audit": self.audit}
		with (self.OUTPUT / "result.json").open("x", encoding="utf-8") as output:
			json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
		print(json.dumps({"final_epoch": self.completed_epoch,
			"full_train_greedy_exact": train_greedy["metrics"]["exact_token_sequences"],
			"train_content": final["train"]["content_accuracy"]}), flush=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--mode", choices=("audit", "gates", "train"), required=True)
	arguments = parser.parse_args()
	control = IntegerMatchedControl()
	if arguments.mode == "audit":
		control.prepare()
		control.OUTPUT.mkdir(parents=True, exist_ok=True)
		(control.OUTPUT / "audit.json").write_text(json.dumps(control.audit, indent=2), encoding="utf-8")
		print(json.dumps(control.audit["integer_configuration"], indent=2))
	elif arguments.mode == "gates":
		control.prepare()
		control.OUTPUT.mkdir(parents=True, exist_ok=True)
		checks = control.gates()
		(control.OUTPUT / "gates.json").write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
		print(json.dumps({"gates": {k: v["passed"] for k, v in checks.items() if k != "all_passed"},
			"all_passed": checks["all_passed"]}, indent=2))
	else:
		control.run()
