"""Fixed-configuration train-fit control with complete epoch checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as numpy
import torch
from torch import Tensor
import torch.nn.functional as functional

from float_adam_8k_control_20260906 import ControlMetrics, FloatControl
from reversible_jako_floatdiag import FloatMT, greedy
import reversible_jako_data as data


class TrainFitControl(FloatControl):
	"""Measure training-set fit without validation-driven stopping."""

	PREVIOUS = FloatControl.OUTPUT
	OUTPUT = FloatControl.ROOT / "float_adam_8k_trainfit_20260906_results"
	MAX_EPOCHS = 120
	TRAIN_THRESHOLD = 95.0
	GREEDY_THRESHOLD = 90.0
	TRAIN_COUNT = 7600
	VALID_COUNT = 400
	DIAGNOSTIC_COUNT = 400
	GREEDY_COUNT = 256
	DIAGNOSTIC_INTERVAL = 5
	MILESTONE_INTERVAL = 10
	COMPARISON_EPOCH = 23
	STATE_NAME = "trainfit_last_full_state.pt"

	def prepare(self) -> None:
		"""Load and validate the saved split and unchanged model configuration."""
		self.previous_audit = json.loads((self.PREVIOUS / "audit.json").read_text(encoding="utf-8"))
		self.configuration = self.previous_audit["float_configuration"]
		if (self.configuration["batch"], self.configuration["seed"], self.configuration["learning_rate"], self.configuration["threads"]) != (self.BATCH, self.SEED, self.LEARNING_RATE, self.THREADS):
			raise ValueError("Inherited training constants differ from previous audit")
		torch.set_num_threads(self.THREADS)
		torch.use_deterministic_algorithms(True)
		random.seed(self.SEED)
		numpy.random.seed(self.SEED)
		torch.manual_seed(self.SEED)
		self.generator = torch.Generator().manual_seed(self.SEED)
		self.dataset = json.loads((self.PREVIOUS / "split.json").read_text(encoding="utf-8"))
		if (len(self.dataset["train_pairs"]), len(self.dataset["valid_pairs"])) != (self.TRAIN_COUNT, self.VALID_COUNT):
			raise ValueError("Saved split counts differ")
		for name in ("train", "valid"):
			digest = hashlib.sha256(json.dumps(self.dataset[name + "_pairs"], ensure_ascii=False).encode("utf-8")).hexdigest()
			if digest != self.previous_audit["split_sha256"][name]:
				raise ValueError("Saved split identity differs")
		training_keys = {tuple(row[:2]) for row in self.dataset["train_pairs"]}
		validation_keys = {tuple(row[:2]) for row in self.dataset["valid_pairs"]}
		if training_keys & validation_keys:
			raise ValueError("Exact JA-KO pair overlap")
		self.tokenizer_path = Path(data.CACHE) / self.previous_audit["data"]["tokenizer"]
		if hashlib.sha256(self.tokenizer_path.read_bytes()).hexdigest() != self.previous_audit["tokenizer_sha256"]:
			raise ValueError("Tokenizer hash differs")
		self.tokenizer = data.load_tokenizer(str(self.tokenizer_path))
		if self.tokenizer.get_vocab_size() != self.previous_audit["data"]["vocab_size"]:
			raise ValueError("Tokenizer vocabulary differs")
		encoded_training = data.encode_pairs(self.tokenizer, self.dataset["train_pairs"], self.LENGTH)
		encoded_validation = data.encode_pairs(self.tokenizer, self.dataset["valid_pairs"], self.LENGTH)
		if (encoded_training[3], encoded_validation[3]) != (self.TRAIN_COUNT, self.VALID_COUNT):
			raise ValueError("Encoding dropped saved rows")
		self.training = tuple(torch.tensor(column) for column in encoded_training[:3])
		self.validation = tuple(torch.tensor(column) for column in encoded_validation[:3])
		self.model = FloatMT(self.tokenizer.get_vocab_size(), self.configuration["dimension"], self.configuration["ffn_hidden"],
			self.configuration["encoder"], self.configuration["decoder"], self.LENGTH, True).float()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
		# Diagnostics have their own generator and cannot consume training-order RNG.
		diagnostic_generator = torch.Generator().manual_seed(self.SEED)
		self.permutations = {
			"train": torch.randperm(self.DIAGNOSTIC_COUNT, generator=diagnostic_generator),
			"valid": torch.randperm(self.VALID_COUNT, generator=diagnostic_generator),
		}
		self.permutation = self.permutations["valid"]
		self.completed_epoch = 0
		self.global_updates = 0
		self.history = []
		self.best_train_epoch = 0
		self.best_train_accuracy = -1.0
		self.best_held_out_epoch = 0
		self.best_held_out_loss = float("inf")
		paths = list(self.PREVIOUS.rglob("*")) + list(self.ROOT.glob("*.py")) + [self.tokenizer_path]
		self.protected = {str(path.relative_to(self.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
			for path in paths if path.is_file() and path.resolve() != Path(__file__).resolve()}
		self.audit = {
			"split_source": str(self.PREVIOUS / "split.json"), "train_count": self.TRAIN_COUNT, "valid_count": self.VALID_COUNT,
			"exact_pair_overlap": 0, "shared_japanese_sources": len({row[0] for row in self.dataset["train_pairs"]} & {row[0] for row in self.dataset["valid_pairs"]}),
			"tokenizer_path": str(self.tokenizer_path), "tokenizer_sha256": self.previous_audit["tokenizer_sha256"],
			"previous_configuration": self.configuration, "optimizer_defaults": self.optimizer.defaults,
			"maximum_epochs": self.MAX_EPOCHS, "training_content_stop_threshold": self.TRAIN_THRESHOLD,
			"validation_stopping": False, "subset_selection": "First 400 train rows for source diagnostics; first 256 for periodic greedy",
			"permutations": {name: value.tolist() for name, value in self.permutations.items()},
			"protected_sha256": self.protected, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
		}

	def verify_protected(self) -> None:
		"""Verify the original files against the pre-test manifest."""
		manifest_path = self.OUTPUT / "audit.json"
		manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["protected_sha256"] if manifest_path.exists() else self.protected
		for name, digest in manifest.items():
			if hashlib.sha256((self.ROOT / name).read_bytes()).hexdigest() != digest:
				raise RuntimeError("Protected file changed: " + name)

	def checkpoint(self) -> dict[str, Any]:
		"""Capture all state needed to continue after a completed epoch."""
		return {
			"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
			"completed_epoch": self.completed_epoch, "global_updates": self.global_updates,
			"python_rng": random.getstate(), "numpy_rng": numpy.random.get_state(), "torch_rng": torch.get_rng_state(),
			"data_order_rng": self.generator.get_state(), "diagnostic_permutations": self.permutations,
			"train_metrics": self.history[-1]["train"] if self.history else {},
			"validation_metrics": self.history[-1]["valid"] if self.history else {}, "history": self.history,
			"best_train_epoch": self.best_train_epoch, "best_train_accuracy": self.best_train_accuracy,
			"best_held_out_epoch": self.best_held_out_epoch, "best_held_out_loss": self.best_held_out_loss,
			"audit": self.audit,
		}

	def restore(self, state: dict[str, Any]) -> None:
		"""Restore a complete checkpoint, rejecting model-only files."""
		required = {"model", "optimizer", "completed_epoch", "global_updates", "python_rng", "numpy_rng", "torch_rng",
			"data_order_rng", "diagnostic_permutations", "history", "train_metrics", "validation_metrics", "audit"}
		if not required.issubset(state) or not state["optimizer"]["state"]:
			raise ValueError("Incomplete training checkpoint")
		if state["audit"]["runner_sha256"] != self.audit["runner_sha256"] or state["audit"]["protected_sha256"] != self.audit["protected_sha256"]:
			raise ValueError("Checkpoint inputs or runner differ")
		self.model.load_state_dict(state["model"])
		self.optimizer.load_state_dict(state["optimizer"])
		self.completed_epoch, self.global_updates = state["completed_epoch"], state["global_updates"]
		self.history = state["history"]
		self.best_train_epoch, self.best_train_accuracy = state["best_train_epoch"], state["best_train_accuracy"]
		self.best_held_out_epoch, self.best_held_out_loss = state["best_held_out_epoch"], state["best_held_out_loss"]
		self.permutations = state["diagnostic_permutations"]
		random.setstate(state["python_rng"])
		numpy.random.set_state(state["numpy_rng"])
		torch.set_rng_state(state["torch_rng"])
		self.generator.set_state(state["data_order_rng"])

	def save_checkpoint(self) -> None:
		"""Atomically publish and read back full epoch state before proceeding."""
		pending = self.OUTPUT / "trainfit_pending_full_state.pt"
		torch.save(self.checkpoint(), pending)
		loaded = torch.load(pending, weights_only=False)
		if loaded["completed_epoch"] != self.completed_epoch or loaded["global_updates"] != self.global_updates:
			raise RuntimeError("Checkpoint epoch/update round-trip failed")
		if not loaded["optimizer"]["state"] or any(not torch.equal(value, loaded["model"][name]) for name, value in self.model.state_dict().items()):
			raise RuntimeError("Checkpoint model/optimizer round-trip failed")
		for parameter_state in loaded["optimizer"]["state"].values():
			if not {"step", "exp_avg", "exp_avg_sq"}.issubset(parameter_state) or int(parameter_state["step"]) != self.global_updates:
				raise RuntimeError("Adam state/update count incomplete")
		if not all(name in loaded for name in ("python_rng", "numpy_rng", "torch_rng", "data_order_rng")):
			raise RuntimeError("Checkpoint RNG state incomplete")
		pending.replace(self.OUTPUT / self.STATE_NAME)
		if self.completed_epoch % self.MILESTONE_INTERVAL == 0:
			with (self.OUTPUT / f"trainfit_epoch_{self.completed_epoch:03d}.pt").open("xb") as milestone:
				milestone.write((self.OUTPUT / self.STATE_NAME).read_bytes())

	def source_diagnostics(self, name: str) -> dict[str, Any]:
		"""Measure fixed source ablations without consuming the training RNG."""
		if name == "train":
			columns = tuple(column[:self.DIAGNOSTIC_COUNT] for column in self.training)
		elif name == "valid":
			columns = self.validation
		else:
			raise ValueError("Unknown diagnostic split")
		self.permutation = self.permutations[name]
		real, shuffled, zeroed = (self.evaluate(columns, mode) for mode in ("real", "shuffled", "zeroed"))
		return {"real": real, "shuffled": shuffled, "zeroed": zeroed,
			"real_minus_shuffled": real["content_accuracy"] - shuffled["content_accuracy"],
			"real_minus_zeroed": real["content_accuracy"] - zeroed["content_accuracy"]}

	@torch.no_grad()
	def greedy_evaluation(self, name: str, count: int) -> dict[str, Any]:
		"""Measure exact reproduction with the previous EOS-stopping decoder."""
		if name not in ("train", "valid"):
			raise ValueError("Unknown generation split")
		columns = self.training if name == "train" else self.validation
		pairs = self.dataset[name + "_pairs"][:count]
		if len(pairs) != count:
			raise ValueError("Generation count exceeds saved split")
		self.model.eval()
		generated_content, references, hypotheses = [], [], []
		exact_tokens = exact_strings = 0
		for start in range(0, count, self.BATCH):
			stop = min(start + self.BATCH, count)
			generated = greedy(self.model, columns[0][start:stop], self.LENGTH)
			for offset, row in enumerate(generated.tolist()):
				index = start + offset
				# Exact sequence equality includes EOS and preserves erroneous PAD/BOS.
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
		metrics.update({"examples": count, "exact_token_sequences": exact_tokens, "exact_token_percent": ControlMetrics.PERCENT * exact_tokens / count,
			"exact_decoded_strings": exact_strings, "decoded_string_reference": "Tokenizer-decoded target tokens, excluding EOS",
			"exact_original_strings": sum(hypothesis == reference for hypothesis, reference in zip(hypotheses, references)),
			"generated_reference_character_ratio": metrics["mean_generated_characters"] / max(1e-12, metrics["mean_reference_characters"])})
		return {"metrics": metrics, "outputs": hypotheses}

	def focused_tests(self) -> None:
		"""Verify the required contracts before any full training trajectory."""
		self.prepare()
		self.OUTPUT.mkdir(exist_ok=False)
		(self.OUTPUT / "audit.json").write_text(json.dumps(self.audit, indent=2), encoding="utf-8")
		checks = {"exact_previous_split_reuse": True, "tokenizer_hash_identical": True}
		initial = {name: value.clone() for name, value in self.model.state_dict().items()}
		with torch.random.fork_rng():
			torch.manual_seed(self.SEED)
			duplicate = FloatMT(self.tokenizer.get_vocab_size(), self.configuration["dimension"], self.configuration["ffn_hidden"],
				self.configuration["encoder"], self.configuration["decoder"], self.LENGTH, True).float()
			assert all(torch.equal(value, duplicate.state_dict()[name]) for name, value in initial.items())
		checks["deterministic_initialization_seed_zero"] = True
		order = torch.randperm(self.TRAIN_COUNT, generator=self.generator)
		sources, inputs, targets = (column[order[:self.BATCH]] for column in self.training)
		logits = self.model(sources, inputs)
		loss = functional.cross_entropy(logits.transpose(1, 2), targets, ignore_index=data.PAD)
		assert bool(torch.isfinite(loss))
		checks["first_batch_loss_finite"] = True
		loss.backward()
		assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in self.model.parameters())
		checks["gradients_finite"] = True
		self.optimizer.step()
		self.global_updates = 1
		assert greedy(self.model, sources, self.LENGTH).shape == targets.shape
		checks["greedy_decoder_works"] = True
		buffer = io.BytesIO()
		torch.save(self.checkpoint(), buffer)
		buffer.seek(0)
		loaded = torch.load(buffer, weights_only=False)
		assert len(loaded["optimizer"]["state"]) == len(list(self.model.parameters()))
		for parameter_state in loaded["optimizer"]["state"].values():
			assert {"step", "exp_avg", "exp_avg_sq"}.issubset(parameter_state)
		checks["full_state_checkpoint_contains_adam"] = True
		# The next update must be bit-identical after a complete round trip.
		next_indices = order[self.BATCH:self.BATCH + self.BATCH]
		self.optimizer.zero_grad(set_to_none=True)
		next_sources, next_inputs, next_targets = (column[next_indices] for column in self.training)
		next_loss = functional.cross_entropy(self.model(next_sources, next_inputs).transpose(1, 2), next_targets, ignore_index=data.PAD)
		next_loss.backward()
		self.optimizer.step()
		expected = {name: value.clone() for name, value in self.model.state_dict().items()}
		self.restore(loaded)
		self.optimizer.zero_grad(set_to_none=True)
		restored_loss = functional.cross_entropy(self.model(next_sources, next_inputs).transpose(1, 2), next_targets, ignore_index=data.PAD)
		restored_loss.backward()
		self.optimizer.step()
		assert torch.equal(next_loss, restored_loss)
		assert all(torch.equal(value, self.model.state_dict()[name]) for name, value in expected.items())
		checks["next_update_bit_identical_after_restore"] = True
		try:
			self.restore({"model": initial})
		except ValueError:
			checks["model_only_restore_rejected"] = True
		else:
			raise AssertionError("Model-only restore accepted")
		self.verify_protected()
		checks["protected_hashes_unchanged"] = True
		(self.OUTPUT / "focused_tests.json").write_text(json.dumps({"checks": checks, "first_batch_loss": float(loss.detach()),
			"runner_sha256": self.audit["runner_sha256"], "passed": True}, indent=2), encoding="utf-8")
		print(json.dumps({"focused_tests": checks, "first_batch_loss": float(loss.detach())}), flush=True)

	def run(self, phase: str) -> None:
		"""Train to the fixed training criterion and publish the final measurements."""
		self.prepare()
		self.verify_protected()
		tests = json.loads((self.OUTPUT / "focused_tests.json").read_text(encoding="utf-8"))
		if not tests["passed"] or tests["runner_sha256"] != self.audit["runner_sha256"]:
			raise RuntimeError("Focused tests did not pass for this runner")
		if phase == "resume":
			if (self.OUTPUT / "result.json").exists():
				raise RuntimeError("The experiment is already complete")
			self.restore(torch.load(self.OUTPUT / self.STATE_NAME, weights_only=False))
		else:
			with (self.OUTPUT / "trajectory_started.json").open("x", encoding="utf-8") as marker:
				json.dump({"seed": self.SEED, "from_epoch": 0}, marker)
		for epoch in range(self.completed_epoch + 1, self.MAX_EPOCHS + 1):
			if self.history and self.history[-1]["train"]["content_accuracy"] >= self.TRAIN_THRESHOLD:
				break
			started = time.monotonic()
			self.model.train()
			order = torch.randperm(self.TRAIN_COUNT, generator=self.generator)
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
					raise FloatingPointError("Nonfinite gradients")
				self.optimizer.step()
				self.global_updates += 1
			self.completed_epoch = epoch
			row = {"epoch": epoch, "global_updates": self.global_updates, "train": self.evaluate(self.training), "valid": self.evaluate(self.validation)}
			best_train = row["train"]["content_accuracy"] > self.best_train_accuracy
			if best_train:
				self.best_train_accuracy, self.best_train_epoch = row["train"]["content_accuracy"], epoch
			if row["valid"]["content_loss"] < self.best_held_out_loss:
				self.best_held_out_loss, self.best_held_out_epoch = row["valid"]["content_loss"], epoch
			# Extra table epochs are observational; only train content accuracy can stop.
			if epoch % self.DIAGNOSTIC_INTERVAL == 0 or epoch in (1, self.COMPARISON_EPOCH) or best_train or epoch == self.MAX_EPOCHS:
				row["train_source"] = self.source_diagnostics("train")
				row["valid_source"] = self.source_diagnostics("valid")
			if epoch % self.MILESTONE_INTERVAL == 0:
				row["train_greedy_256"] = self.greedy_evaluation("train", self.GREEDY_COUNT)
			row["seconds"] = time.monotonic() - started
			self.history.append(row)
			self.save_checkpoint()
			(self.OUTPUT / "history.json").write_text(json.dumps(self.history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
			print(json.dumps({"epoch": epoch, "updates": self.global_updates, "seconds": row["seconds"],
				"train_loss": row["train"]["loss"], "train_content": row["train"]["content_accuracy"],
				"train_pos0": row["train"]["pos0_accuracy"], "valid_content": row["valid"]["content_accuracy"],
				"train_greedy_256_exact": row.get("train_greedy_256", {}).get("metrics", {}).get("exact_token_sequences")}), flush=True)
			if epoch == 1:
				previous = torch.load(self.PREVIOUS / "smoke_state.pt", weights_only=False)
				if not all(torch.equal(value, previous["model"][name]) for name, value in self.model.state_dict().items()):
					raise RuntimeError("Epoch-one model differs from previous seed-zero trajectory")
				if not torch.equal(self.generator.get_state(), previous["generator"]):
					raise RuntimeError("Epoch-one data-order RNG differs")
				print("Epoch-one model and data-order state exactly match the previous trajectory", flush=True)
		train_greedy = self.greedy_evaluation("train", self.TRAIN_COUNT)
		final_train_source = self.source_diagnostics("train")
		final_valid_source = self.source_diagnostics("valid")
		valid_greedy = self.greedy_evaluation("valid", self.VALID_COUNT)
		self.verify_protected()
		final = self.history[-1]
		if final["train"]["content_accuracy"] < self.TRAIN_THRESHOLD:
			verdict = "C"
			conclusion = "The matched float+Adam model did not fit the 7,600-pair training set under this fixed configuration within 120 epochs."
		elif train_greedy["metrics"]["exact_token_percent"] < self.GREEDY_THRESHOLD:
			verdict = "B"
			conclusion = "The model fits the teacher-forced objective but does not reliably reproduce the training targets autoregressively."
		else:
			if min(final_train_source["real_minus_shuffled"], final_train_source["real_minus_zeroed"]) <= 0:
				raise RuntimeError("Train-fit thresholds met without positive source-dependence gaps; verdict requires review")
			verdict = "A"
			conclusion = "The matched float+Adam model can fit the 7,600-pair training set. The previous epoch-23 stop did not establish a float optimization failure."
		result = {"verdict": verdict, "conclusion": conclusion, "final_epoch": self.completed_epoch, "best_train_epoch": self.best_train_epoch,
			"best_held_out_epoch": self.best_held_out_epoch, "final_train": final["train"], "final_valid": final["valid"],
			"train_greedy": train_greedy, "valid_greedy": valid_greedy, "final_train_source": final_train_source, "final_valid_source": final_valid_source,
			"protected_hashes_unchanged": True, "full_checkpoint": str(self.OUTPUT / self.STATE_NAME)}
		with (self.OUTPUT / "result.json").open("x", encoding="utf-8") as output:
			json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
		print(json.dumps({"verdict": verdict, "final_epoch": self.completed_epoch, "full_train_greedy_exact": train_greedy["metrics"]["exact_token_sequences"]}), flush=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--mode", choices=("test", "train", "resume"), required=True)
	arguments = parser.parse_args()
	control = TrainFitControl()
	if arguments.mode == "test":
		control.focused_tests()
	else:
		control.run(arguments.mode)
