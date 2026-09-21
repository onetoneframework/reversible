"""Matched held-out float control using the existing diagnostic model."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as functional

import reversible_jako_data as data
from reversible_jako_floatdiag import FloatMT, greedy
from reversible_jako_mt import bleu, chrf


class ControlMetrics:
	"""Aggregate token and generated-text measurements without selecting examples."""

	BUCKETS = (("0", 0, 1), ("1", 1, 2), ("2", 2, 3), ("3", 3, 4), ("4-7", 4, 8), ("8+", 8, 16))
	PERCENT = 100.0

	@staticmethod
	def teacher(logits: Tensor, targets: Tensor) -> dict[str, Any]:
		"""Measure non-padding tokens, with EOS excluded from content."""
		if not bool(torch.isfinite(logits).all()):
			raise FloatingPointError("Nonfinite validation logits")
		predictions = logits.argmax(-1)
		valid = targets != data.PAD
		content = valid & (targets != data.EOS)
		eos = targets == data.EOS
		correct = predictions == targets
		losses = functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
		counts = {"tokens": int(valid.sum()), "content_tokens": int(content.sum()), "eos_tokens": int(eos.sum())}
		result = {
			**counts,
			"all_accuracy": ControlMetrics.PERCENT * float((correct & valid).sum()) / max(1, counts["tokens"]),
			"content_accuracy": ControlMetrics.PERCENT * float((correct & content).sum()) / max(1, counts["content_tokens"]),
			"loss": float(losses[valid].sum()) / max(1, counts["tokens"]),
			"content_loss": float(losses[content].sum()) / max(1, counts["content_tokens"]),
			"eos_recall": ControlMetrics.PERCENT * float((correct & eos).sum()) / max(1, counts["eos_tokens"]),
			"eos_false_positive_rate": ControlMetrics.PERCENT * float(((predictions == data.EOS) & content).sum()) / max(1, counts["content_tokens"]),
			"pos0_accuracy": ControlMetrics.PERCENT * float(correct[:, 0].float().mean()),
			"reference_mean_tokens": float(content.sum(1).float().mean()),
			"position_buckets": {},
		}
		for label, start, end in ControlMetrics.BUCKETS:
			mask = content[:, start:end]
			result["position_buckets"][label] = {
				"content_tokens": int(mask.sum()),
				"content_accuracy": ControlMetrics.PERCENT * float((correct[:, start:end] & mask).sum()) / max(1, int(mask.sum())),
			}
		return result

	@staticmethod
	def generated(hypotheses: list[str], references: list[str], tokens: list[list[int]]) -> dict[str, Any]:
		"""Measure text quality and degeneracy using existing character metrics."""
		if not hypotheses or len(hypotheses) != len(references) or len(tokens) != len(hypotheses):
			raise ValueError("Generation and reference rows must be nonempty and aligned")
		characters = "".join(hypotheses)
		generated_length = sum(len(text) for text in hypotheses) / len(hypotheses)
		reference_length = sum(len(text) for text in references) / len(references)
		result = {
			"chrf": chrf(hypotheses, references),
			"character_bleu_add_one": bleu(hypotheses, references),
			"mean_generated_characters": generated_length,
			"mean_reference_characters": reference_length,
			"reference_generated_character_ratio": reference_length / generated_length if generated_length else None,
			"mean_generated_content_tokens": sum(len(row) for row in tokens) / len(tokens),
			"korean_character_percent": ControlMetrics.PERCENT * sum("\uac00" <= character <= "\ud7a3" for character in characters) / max(1, len(characters)),
			"replacement_character_percent": ControlMetrics.PERCENT * characters.count("\ufffd") / max(1, len(characters)),
			"distinct_outputs": len(set(hypotheses)),
			"most_common_output_fraction": Counter(hypotheses).most_common(1)[0][1] / len(hypotheses),
		}
		for order in (1, 2):
			total = repeated = 0
			for row in tokens:
				grams = [tuple(row[index:index + order]) for index in range(len(row) - order + 1)]
				total += len(grams)
				repeated += len(grams) - len(set(grams))
			result[f"repetition_token_{order}gram"] = repeated / max(1, total)
		return result


class FloatControl:
	"""Run one matched training trajectory with held-out checkpoint selection."""

	ROOT = Path(__file__).resolve().parent
	OUTPUT = ROOT / "float_adam_8k_control_20260906_results"
	MAX_EPOCHS = 60
	PATIENCE = 10
	LEARNING_RATE = 0.0003
	BATCH = 16
	LENGTH = 16
	SEED = 0
	THREADS = 6
	CORE_DIGEST = "ee22aee1e7486d560d03535bdba05e54146cf2bb262c14529a5c99453b6076a3"

	def __init__(self) -> None:
		self.history: list[dict[str, Any]] = []
		self.best_loss = float("inf")
		self.best_epoch = 0
		self.stale_epochs = 0
		self.elapsed_seconds = 0.0

	def prepare(self) -> None:
		"""Reproduce logged data and verify protected input identity."""
		torch.set_num_threads(self.THREADS)
		torch.manual_seed(self.SEED)
		torch.use_deterministic_algorithms(True)
		self.generator = torch.Generator().manual_seed(self.SEED)
		self.dataset = data.prepare("both", 0, 5000, self.LENGTH, seed=self.SEED,
			valid_frac=0.05, max_tok=14, tok_mode="char", base_pairs=8000, min_char_count=20)
		logged = json.loads((self.ROOT / "jako_stage4_pos.json").read_text(encoding="utf-8"))
		causal = json.loads((self.ROOT / "jako_small_causalfix.json").read_text(encoding="utf-8"))
		if self.dataset["info"] != logged["data"] or self.dataset["info"] != causal["data"]:
			raise ValueError("Matched data metadata differs from logged integer runs")
		training_keys = {(data.norm_key(row[0]), data.norm_key(row[1])) for row in self.dataset["train_pairs"]}
		validation_keys = {(data.norm_key(row[0]), data.norm_key(row[1])) for row in self.dataset["valid_pairs"]}
		if training_keys & validation_keys:
			raise ValueError("Validation pair leakage")
		acceptance = data.acceptance_sentences()
		all_keys = {data.norm_key(text) for row in self.dataset["train_pairs"] + self.dataset["valid_pairs"] for text in row[:2]}
		leakage = sum(data.norm_key(row[0]) in all_keys or data.norm_key(row[1]) in all_keys for row in acceptance)
		if len(acceptance) != 30 or leakage:
			raise ValueError("Acceptance exclusion failed")
		self.protected = {}
		for path in self.ROOT.glob("*.py"):
			if path != Path(__file__).resolve() and not path.name.startswith("test_float_adam_8k_control"):
				self.protected[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
		if self.protected["reversible_integer_final.py"] != self.CORE_DIGEST:
			raise ValueError("Canonical core changed")
		self.tokenizer = self.dataset["tokenizer"]
		self.training = tuple(torch.tensor(column) for column in self.dataset["train"])
		self.validation = tuple(torch.tensor(column) for column in self.dataset["valid"])
		# A nonzero cyclic shift is a fixed source permutation with no fixed points.
		self.permutation = torch.arange(len(self.validation[0])).roll(1)
		self.model = FloatMT(5000, 96, 192, "af", "ccaf", self.LENGTH, True).float()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
		self.audit = {
			"data": self.dataset["info"], "acceptance_leakage": leakage, "acceptance_count": len(acceptance),
			"normalized_pair_overlap": 0, "protected_sha256": self.protected,
			"tokenizer_sha256": hashlib.sha256((Path(data.CACHE) / self.dataset["info"]["tokenizer"]).read_bytes()).hexdigest(),
			"split_sha256": {name: hashlib.sha256(json.dumps(self.dataset[name + "_pairs"], ensure_ascii=False).encode("utf-8")).hexdigest() for name in ("train", "valid")},
			"parameters": sum(parameter.numel() for parameter in self.model.parameters()),
			"float_configuration": {"dtype": "float32", "optimizer": "Adam", "learning_rate": self.LEARNING_RATE,
				"max_epochs": self.MAX_EPOCHS, "patience": self.PATIENCE, "batch": self.BATCH, "seed": self.SEED,
				"dimension": 96, "ffn_hidden": 192, "encoder": "af", "decoder": "ccaf", "positional": "learned tables, length 16",
				"selection": "minimum held-out non-EOS content cross entropy", "source_diagnostic_examples": 400,
				"torch": torch.__version__, "threads": self.THREADS, "device": "cpu"},
		}

	@torch.no_grad()
	def evaluate(self, columns: tuple[Tensor, ...], mode: str = "real") -> dict[str, Any]:
		"""Evaluate all rows with real, permuted, or zero encoder memory."""
		self.model.eval()
		sources, inputs, targets = columns
		if mode == "shuffled":
			sources = sources[self.permutation]
		elif mode not in ("real", "zeroed"):
			raise ValueError("Unknown source diagnostic")
		# Retain logits only per evaluation batch, then aggregate weighted metrics.
		parts = []
		for start in range(0, len(sources), self.BATCH):
			source = sources[start:start + self.BATCH]
			memory = self.model.encode(source)
			if mode == "zeroed":
				memory = torch.zeros_like(memory)
			logits = self.model.Wout(self.model.decode(inputs[start:start + self.BATCH], memory, source))
			parts.append((len(source), ControlMetrics.teacher(logits, targets[start:start + self.BATCH])))
		result: dict[str, Any] = {"examples": len(sources)}
		for key in ("tokens", "content_tokens", "eos_tokens"):
			result[key] = sum(part[key] for _, part in parts)
		weights = {"all_accuracy": "tokens", "loss": "tokens", "content_accuracy": "content_tokens", "content_loss": "content_tokens",
			"eos_recall": "eos_tokens", "eos_false_positive_rate": "content_tokens", "pos0_accuracy": "examples", "reference_mean_tokens": "examples"}
		for key, weight in weights.items():
			result[key] = sum(part[key] * (count if weight == "examples" else part[weight]) for count, part in parts) / max(1, result[weight])
		result["position_buckets"] = {}
		for label, _, _ in ControlMetrics.BUCKETS:
			count = sum(part["position_buckets"][label]["content_tokens"] for _, part in parts)
			result["position_buckets"][label] = {"content_tokens": count,
				"content_accuracy": sum(part["position_buckets"][label]["content_accuracy"] * part["position_buckets"][label]["content_tokens"] for _, part in parts) / max(1, count)}
		return result

	@torch.no_grad()
	def generate(self, mode: str = "real") -> dict[str, Any]:
		"""Greedily decode the complete held-out split, stopping at EOS."""
		self.model.eval()
		sources = self.validation[0]
		if mode == "shuffled":
			sources = sources[self.permutation]
		elif mode != "real":
			raise ValueError("Unknown greedy source diagnostic")
		tokens = []
		for start in range(0, len(sources), self.BATCH):
			generated = greedy(self.model, sources[start:start + self.BATCH], self.LENGTH)
			for row in generated.tolist():
				if data.EOS in row:
					row = row[:row.index(data.EOS)]
				tokens.append([token for token in row if token not in (data.PAD, data.BOS, data.EOS)])
		hypotheses = [self.tokenizer.decode(row) for row in tokens]
		references = [row[1] for row in self.dataset["valid_pairs"]]
		return {"metrics": ControlMetrics.generated(hypotheses, references, tokens), "outputs": hypotheses}

	def validate(self, epoch: int) -> dict[str, Any]:
		"""Record held-out quality and source dependence independently of selection."""
		real = self.evaluate(self.validation)
		shuffled = self.evaluate(self.validation, "shuffled")
		zeroed = self.evaluate(self.validation, "zeroed")
		generation = self.generate()
		warnings = []
		if real["content_accuracy"] <= shuffled["content_accuracy"]:
			warnings.append("real content accuracy <= shuffled")
		if real["content_accuracy"] <= zeroed["content_accuracy"]:
			warnings.append("real content accuracy <= zeroed")
		if generation["metrics"]["mean_generated_content_tokens"] < real["reference_mean_tokens"] / 2:
			warnings.append("generated token length below half reference")
		if generation["metrics"]["most_common_output_fraction"] >= 0.5:
			warnings.append("at least half the outputs are identical")
		if self.history:
			previous = self.history[-1]["held_out"]
			if real["eos_recall"] > previous["eos_recall"] and real["content_accuracy"] <= previous["content_accuracy"]:
				warnings.append("EOS recall increased without content accuracy improvement")
		return {"epoch": epoch, "train_monitor_first_400": self.evaluate(tuple(column[:400] for column in self.training)),
			"held_out": real, "shuffled": shuffled, "zeroed": zeroed,
			"real_minus_shuffled": {key: real[key] - shuffled[key] for key in ("content_accuracy", "pos0_accuracy", "loss")},
			"real_minus_zeroed": {key: real[key] - zeroed[key] for key in ("content_accuracy", "pos0_accuracy", "loss")},
			"greedy": generation, "warnings": warnings}

	def run(self, phase: str) -> None:
		"""Execute smoke or continue its single trajectory with early stopping."""
		self.prepare()
		if phase == "smoke":
			self.OUTPUT.mkdir(exist_ok=False)
			(self.OUTPUT / "audit.json").write_text(json.dumps(self.audit, indent=2), encoding="utf-8")
			(self.OUTPUT / "split.json").write_text(json.dumps({key: self.dataset[key] for key in ("train_pairs", "valid_pairs")}, ensure_ascii=False), encoding="utf-8")
			initial = self.validate(0)
			self.history.append(initial)
			self.best_loss = initial["held_out"]["content_loss"]
			torch.save(self.model.state_dict(), self.OUTPUT / "best.pt")
			start_epoch, stop_epoch = 1, 1
		else:
			if (self.OUTPUT / "full_started.json").exists():
				raise FileExistsError("The single full run has already been started")
			state = torch.load(self.OUTPUT / "smoke_state.pt", weights_only=False)
			self.model.load_state_dict(state["model"])
			self.optimizer.load_state_dict(state["optimizer"])
			self.generator.set_state(state["generator"])
			torch.set_rng_state(state["torch_rng"])
			self.history = state["history"]
			self.best_loss, self.best_epoch = state["best_loss"], state["best_epoch"]
			self.stale_epochs = state["stale_epochs"]
			self.elapsed_seconds = state["elapsed_seconds"]
			(self.OUTPUT / "full_started.json").write_text(json.dumps({"resume_epoch": 1, "maximum_epochs": self.MAX_EPOCHS, "patience": self.PATIENCE}), encoding="utf-8")
			start_epoch, stop_epoch = 2, self.MAX_EPOCHS
		for epoch in range(start_epoch, stop_epoch + 1):
			started = time.monotonic()
			self.model.train()
			order = torch.randperm(len(self.training[0]), generator=self.generator)
			for start in range(0, len(order), self.BATCH):
				indices = order[start:start + self.BATCH]
				sources, inputs, targets = (column[indices] for column in self.training)
				self.optimizer.zero_grad(set_to_none=True)
				logits = self.model(sources, inputs)
				loss = functional.cross_entropy(logits.transpose(1, 2), targets, ignore_index=data.PAD)
				if not bool(torch.isfinite(loss)):
					raise FloatingPointError("Nonfinite training loss")
				loss.backward()
				if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in self.model.parameters()):
					raise FloatingPointError("Nonfinite gradient")
				self.optimizer.step()
			row = self.validate(epoch)
			row["epoch_seconds"] = time.monotonic() - started
			self.elapsed_seconds += row["epoch_seconds"]
			self.history.append(row)
			if row["held_out"]["content_loss"] < self.best_loss:
				self.best_loss, self.best_epoch, self.stale_epochs = row["held_out"]["content_loss"], epoch, 0
				torch.save(self.model.state_dict(), self.OUTPUT / "best.pt")
			else:
				self.stale_epochs += 1
			(self.OUTPUT / "history.json").write_text(json.dumps(self.history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
			print(json.dumps({"epoch": epoch, "seconds": row["epoch_seconds"], "train": row["train_monitor_first_400"]["content_accuracy"],
				"held_out": row["held_out"]["content_accuracy"], "content_loss": row["held_out"]["content_loss"],
				"shuffled": row["shuffled"]["content_accuracy"], "zeroed": row["zeroed"]["content_accuracy"],
				"chrf": row["greedy"]["metrics"]["chrf"], "best_epoch": self.best_epoch, "stale_epochs": self.stale_epochs}), flush=True)
			if self.stale_epochs >= self.PATIENCE:
				break
		if phase == "smoke":
			torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "generator": self.generator.get_state(),
				"torch_rng": torch.get_rng_state(), "history": self.history, "best_loss": self.best_loss, "best_epoch": self.best_epoch,
				"stale_epochs": self.stale_epochs, "elapsed_seconds": self.elapsed_seconds}, self.OUTPUT / "smoke_state.pt")
			print("SMOKE PASS: finite forward/backward, validation, greedy, shuffled and zeroed memory", flush=True)
		else:
			self.model.load_state_dict(torch.load(self.OUTPUT / "best.pt", weights_only=True))
			selected = self.history[self.best_epoch]
			result = {"selected_epoch": self.best_epoch, "epochs_completed": self.history[-1]["epoch"],
				"training_validation_seconds": self.elapsed_seconds, "train_all": self.evaluate(self.training), "selected": selected,
				"greedy_shuffled": self.generate("shuffled"), "initial": self.history[0],
				"held_out_pairs": self.dataset["valid_pairs"], "audit": self.audit}
			(self.OUTPUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
			print("FULL RUN COMPLETE", flush=True)
		for name, digest in self.protected.items():
			if hashlib.sha256((self.ROOT / name).read_bytes()).hexdigest() != digest:
				raise RuntimeError("Protected input changed: " + name)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--phase", choices=("smoke", "full"), required=True)
	FloatControl().run(parser.parse_args().phase)
