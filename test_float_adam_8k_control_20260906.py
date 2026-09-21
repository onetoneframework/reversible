"""Focused metric, causality, and runner checks for the float control."""

import unittest
from unittest.mock import patch

import torch

import float_adam_8k_control_20260906 as control


class FloatControlTests(unittest.TestCase):
	"""Verify observable diagnostics on deterministic synthetic inputs."""

	def setUp(self) -> None:
		torch.set_num_threads(1)
		torch.manual_seed(0)
		self.runner = control.FloatControl()
		self.runner.model = control.FloatMT(8, 4, 8, "af", "ccaf", 4, True)
		self.sources = torch.tensor([[4, 2, 0, 0], [5, 2, 0, 0]])
		self.inputs = torch.tensor([[1, 4, 0, 0], [1, 5, 0, 0]])
		self.targets = torch.tensor([[4, 2, 0, 0], [5, 2, 0, 0]])
		self.runner.validation = (self.sources, self.inputs, self.targets)
		self.runner.training = self.runner.validation
		self.runner.permutation = torch.tensor([1, 0])

	def test_teacher_excludes_padding_and_eos_from_content(self) -> None:
		predictions = torch.tensor([[2, 2, 7, 7], [5, 2, 7, 7]])
		logits = torch.nn.functional.one_hot(predictions, 8).float()
		measured = control.ControlMetrics.teacher(logits, self.targets)
		self.assertEqual(measured["all_accuracy"], 75.0)
		self.assertEqual(measured["content_accuracy"], 50.0)
		self.assertEqual(measured["eos_recall"], 100.0)
		self.assertEqual(measured["eos_false_positive_rate"], 50.0)
		self.assertEqual(measured["position_buckets"]["1"]["content_tokens"], 0)

	def test_teacher_rejects_nonfinite_and_handles_no_content(self) -> None:
		logits = torch.zeros(1, 2, 8)
		measured = control.ControlMetrics.teacher(logits, torch.tensor([[2, 0]]))
		self.assertEqual(measured["content_loss"], 0.0)
		logits[0, 0, 0] = float("nan")
		with self.assertRaises(FloatingPointError):
			control.ControlMetrics.teacher(logits, torch.tensor([[2, 0]]))

	def test_generation_reports_collapse_repetition_and_invalid_rows(self) -> None:
		measured = control.ControlMetrics.generated(["abcabc", "abcabc"], ["abcabc", "abcabc"], [[4, 4], [4, 4]])
		self.assertEqual(measured["distinct_outputs"], 1)
		self.assertEqual(measured["chrf"], 100.0)
		self.assertEqual(measured["repetition_token_1gram"], 0.5)
		self.assertIsNone(control.ControlMetrics.generated([""], ["abc"], [[]])["reference_generated_character_ratio"])
		with self.assertRaises(ValueError):
			control.ControlMetrics.generated([], [], [])

	def test_evaluation_aggregates_and_zeroes_encoder_memory(self) -> None:
		self.runner.BATCH = 1
		measured = self.runner.evaluate(self.runner.validation)
		with torch.no_grad():
			direct = control.ControlMetrics.teacher(self.runner.model(self.sources, self.inputs), self.targets)
		self.assertAlmostEqual(measured["loss"], direct["loss"], places=6)
		with patch.object(self.runner.model, "decode", wraps=self.runner.model.decode) as decode:
			self.runner.evaluate(self.runner.validation, "zeroed")
			self.assertTrue(all(bool((call.args[1] == 0).all()) for call in decode.call_args_list))
		self.assertEqual(self.runner.evaluate(self.runner.validation, "shuffled")["examples"], 2)
		with self.assertRaises(ValueError):
			self.runner.evaluate(self.runner.validation, "invalid")

	def test_existing_model_is_causal_and_masks_padding_memory(self) -> None:
		with torch.no_grad():
			original = self.runner.model(self.sources, self.inputs)
			mutated = self.inputs.clone()
			mutated[:, 1:] = 7
			changed = self.runner.model(self.sources, mutated)
			self.assertTrue(torch.equal(original[:, 0], changed[:, 0]))
			memory = self.runner.model.encode(self.sources)
			decoded = self.runner.model.decode(self.inputs, memory, self.sources)
			memory[:, 2:] = 10000
			self.assertTrue(torch.equal(decoded, self.runner.model.decode(self.inputs, memory, self.sources)))

	def test_greedy_and_validation_use_all_held_out_rows(self) -> None:
		self.runner.LENGTH = 4
		self.runner.dataset = {"valid_pairs": [("source one", "target one"), ("source two", "target two")]}
		with patch.object(self.runner, "tokenizer", create=True) as tokenizer:
			tokenizer.decode.return_value = "output"
			self.assertEqual(len(self.runner.generate()["outputs"]), 2)
			self.assertEqual(len(self.runner.generate("shuffled")["outputs"]), 2)
			row = self.runner.validate(0)
			self.assertEqual(row["zeroed"]["examples"], 2)
			self.assertIn("real_minus_shuffled", row)
		with self.assertRaises(ValueError):
			self.runner.generate("invalid")

	def test_prepare_rejects_mismatched_data_before_training(self) -> None:
		with patch.object(control.data, "prepare", return_value={"info": {"pairs": 1}}):
			with patch.object(control.Path, "read_text", return_value='{"data": {"pairs": 2}}'):
				with self.assertRaisesRegex(ValueError, "metadata differs"):
					self.runner.prepare()

	def test_run_rejects_a_second_full_run(self) -> None:
		with patch.object(self.runner, "prepare"):
			with patch.object(control.Path, "exists", return_value=True):
				with self.assertRaisesRegex(FileExistsError, "already been started"):
					self.runner.run("full")


if __name__ == "__main__":
	unittest.main()
