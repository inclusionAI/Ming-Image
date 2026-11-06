"""CPU tests for the alignment bundle comparison tool."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
COMPARE = REPOSITORY / "tools" / "compare_case.py"


def _bundle(root: Path, *, family="tinyv3", seed=42, noise_scale=1.0, pixel_shift=0):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    meta = {
        "case": {
            "task": "text-to-image",
            "prompt_sha256": "ab" * 32,
            "input_image": None,
            "width": 64,
            "height": 64,
            "steps": 2,
            "cfg": 1.0,
            "seed": seed,
            "dtype": "bfloat16",
            "attn_implementation": None,
        },
        "mllm_family": family,
        "profile": {
            "inference_profile": "generation_edit",
            "alignment_padding_mode": "zero_masked",
            "multi_frame_output": False,
            "vae_input_channels": 4,
            "vae_sample_mode": "argmax",
        },
        "model": {"config_hashes": {}},
        "environment": {},
        "notes": {},
    }
    (root / "case.json").write_text(json.dumps(meta), encoding="utf-8")
    arrays = {
        "input_ids": rng.integers(0, 1000, size=(1, 12), dtype=np.int64),
        "attention_mask": np.ones((1, 12), dtype=np.int64),
        "condition_embeds": rng.normal(size=(1, 8, 16)).astype(np.float32),
        "directvlm_hidden_states": rng.normal(size=(1, 6, 16)).astype(np.float32),
        "initial_noise": (rng.normal(size=(1, 4, 8, 8)) * noise_scale).astype(np.float32),
        "step_latents": rng.normal(size=(2, 1, 4, 8, 8)).astype(np.float32),
        "final_pixels": rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8),
    }
    if pixel_shift:
        arrays["final_pixels"] = np.clip(
            arrays["final_pixels"].astype(np.int32) + pixel_shift, 0, 255
        ).astype(np.uint8)
    np.savez(root / "arrays.npz", **arrays)
    return root, arrays


class CompareCaseTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="compare-case-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _run(self, ref, cand, extra=()):
        return subprocess.run(
            [sys.executable, str(COMPARE), "--reference", str(ref),
             "--candidate", str(cand), *extra],
            capture_output=True, text=True,
        )

    def test_identical_bundles_pass(self):
        ref, _ = _bundle(self.root / "a")
        cand, _ = _bundle(self.root / "b")
        result = self._run(ref, cand)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["verdict"], "pass")
        self.assertTrue(report["segments"]["token_inputs"]["input_ids_match"])

    def test_noise_mismatch_fails(self):
        ref, _ = _bundle(self.root / "a")
        cand, _ = _bundle(self.root / "b", noise_scale=1.01)
        result = self._run(ref, cand)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertFalse(report["segments"]["initial_noise"]["pass"])

    def test_family_mismatch_is_incomparable(self):
        ref, _ = _bundle(self.root / "a", family="tinyv3")
        cand, _ = _bundle(self.root / "b", family="miniv2")
        result = self._run(ref, cand)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["verdict"], "incomparable")

    def test_seed_mismatch_is_incomparable(self):
        ref, _ = _bundle(self.root / "a", seed=42)
        cand, _ = _bundle(self.root / "b", seed=7)
        result = self._run(ref, cand)
        report = json.loads(result.stdout)
        self.assertEqual(report["verdict"], "incomparable")

    def test_token_ids_mismatch_fails(self):
        ref, _ = _bundle(self.root / "a")
        cand, _ = _bundle(self.root / "b")
        arrays = dict(np.load(cand / "arrays.npz"))
        arrays["input_ids"] = arrays["input_ids"] + 1
        np.savez(cand / "arrays.npz", **arrays)
        result = self._run(ref, cand)
        report = json.loads(result.stdout)
        self.assertFalse(report["segments"]["token_inputs"]["pass"])
        self.assertEqual(report["verdict"], "fail")

    def test_step_latents_all_needs_review_not_pass(self):
        ref, _ = _bundle(self.root / "a")
        cand, _ = _bundle(self.root / "b")
        result = self._run(ref, cand)
        report = json.loads(result.stdout)
        self.assertEqual(
            report["segments"]["step_latents_all"]["status"], "needs_review"
        )

    def test_report_file_written(self):
        ref, _ = _bundle(self.root / "a")
        cand, _ = _bundle(self.root / "b")
        report_path = self.root / "report.json"
        result = self._run(ref, cand, ("--report", str(report_path)))
        self.assertEqual(result.returncode, 0)
        self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
