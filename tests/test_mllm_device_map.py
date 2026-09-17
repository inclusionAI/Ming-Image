import json
import math
from pathlib import Path
import tempfile
import unittest

from mllm_device_map import (
    MLLMDeviceMapError,
    allocate_mllm_layer_counts,
    build_mllm_device_plan,
    load_mllm_num_hidden_layers,
    validate_loaded_layer_devices,
)


class MLLMDeviceMapTest(unittest.TestCase):
    def test_eight_gpu_plan_reserves_gpu_zero(self):
        plan = build_mllm_device_plan(32, 8)
        self.assertEqual(plan.layer_counts, (1, 4, 4, 4, 4, 5, 5, 5))
        self.assertEqual(len(plan.layer_devices), 32)
        self.assertEqual(plan.device_map["vision"], 0)
        self.assertEqual(plan.device_map["model.model.layers.31"], 7)

    def _expected_layer_devices(self, num_layers, plan):
        if plan.n_gpu == 1:
            return (0,) * num_layers
        return tuple(
            device for device, count in enumerate(plan.layer_counts) for _ in range(count)
        )

    def _assert_plan_valid(self, plan, num_layers, n_gpu):
        self.assertEqual(sum(plan.layer_counts), num_layers)
        self.assertEqual(len(plan.layer_devices), num_layers)
        self.assertEqual(len(plan.layer_counts), n_gpu)
        self.assertEqual(plan.layer_devices, self._expected_layer_devices(num_layers, plan))
        self.assertTrue(all(0 <= d < n_gpu for d in plan.layer_devices))
        for key in ("vision", "linear_proj", "model.model.word_embeddings.weight",
                    "model.model.norm.weight", "model.lm_head.weight", "model.model.norm"):
            self.assertEqual(plan.device_map[key], 0, key)
        self.assertEqual(plan.device_map, build_mllm_device_plan(num_layers, n_gpu).device_map)

    def test_one_gpu_plan_is_all_on_device_zero(self):
        plan = build_mllm_device_plan(30, 1)
        self.assertEqual(plan.layer_counts, (30,))
        self.assertEqual(plan.layer_devices, (0,) * 30)
        for key, value in plan.device_map.items():
            self.assertEqual(value, 0)
        self._assert_plan_valid(plan, 30, 1)

    def test_intermediate_counts_cover_all_layers_once(self):
        for n_gpu in (3, 5, 6, 7):
            with self.subTest(n_gpu=n_gpu):
                num_hidden_layers = 30
                plan = build_mllm_device_plan(num_hidden_layers, n_gpu)
                self._assert_plan_valid(plan, num_hidden_layers, n_gpu)
                # GPU 0 must carry fewer layers than the equal share for n_gpu > 1.
                self.assertLess(plan.layer_counts[0], math.ceil(num_hidden_layers / n_gpu) + 1)

    def test_reads_nested_checkpoint_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"llm_config": {"num_hidden_layers": 32}}),
                encoding="utf-8",
            )
            self.assertEqual(load_mllm_num_hidden_layers(directory), 32)

    def test_rejects_non_positive_gpu_count(self):
        for bad in (0, -1, -8):
            with self.subTest(n_gpu=bad):
                with self.assertRaisesRegex(MLLMDeviceMapError, "positive integer"):
                    allocate_mllm_layer_counts(30, bad)

    def test_loaded_devices_must_match_plan(self):
        plan = build_mllm_device_plan(32, 8)
        validate_loaded_layer_devices(list(plan.layer_devices), plan)
        with self.assertRaisesRegex(MLLMDeviceMapError, "disagrees"):
            validate_loaded_layer_devices([0] * 32, plan)


if __name__ == "__main__":
    unittest.main()