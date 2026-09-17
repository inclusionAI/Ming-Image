import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from infer import parse_args


REPOSITORY = Path(__file__).resolve().parents[1]
INFER = REPOSITORY / "infer.py"


class InferenceCliTest(unittest.TestCase):
    def _model_directory(self, profile):
        temporary = tempfile.TemporaryDirectory()
        model_directory = Path(temporary.name)
        (model_directory / "inference_profile.json").write_text(
            json.dumps(profile), encoding="utf-8"
        )
        return temporary, model_directory

    def test_cli_defaults_to_one_gpu(self):
        with patch.object(
            sys,
            "argv",
            ["infer.py", "--model", "checkpoint", "--task", "text-to-image"],
        ):
            args = parse_args()
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.device_map, "balanced")
        self.assertEqual(args.num_gpus, 1)

    def test_validate_only_accepts_local_generation_checkpoint(self):
        temporary, model_directory = self._model_directory(
            {
                "schema_version": 1,
                "inference_profile": "generation_edit",
                "alignment_padding_mode": "zero_masked",
                "multi_frame_output": False,
                "vae_input_channels": 4,
                "vae_sample_mode": "argmax",
            }
        )
        self.addCleanup(temporary.cleanup)
        result = subprocess.run(
            [
                sys.executable,
                str(INFER),
                "--model",
                str(model_directory),
                "--task",
                "text-to-image",
                "--prompt",
                "test",
                "--validate-only",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["task"], "text-to-image")
        self.assertEqual(payload["sampling"], {"steps": 12, "cfg": 1.0})

    def test_validate_only_accepts_long_literal_prompt(self):
        temporary, model_directory = self._model_directory(
            {
                "schema_version": 1,
                "inference_profile": "generation_edit",
                "alignment_padding_mode": "zero_masked",
                "multi_frame_output": False,
                "vae_input_channels": 4,
                "vae_sample_mode": "argmax",
            }
        )
        self.addCleanup(temporary.cleanup)
        long_prompt = "Create a detailed ocean research poster. " * 40
        result = subprocess.run(
            [
                sys.executable,
                str(INFER),
                "--model",
                str(model_directory),
                "--task",
                "text-to-image",
                "--prompt",
                long_prompt,
                "--validate-only",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["task"], "text-to-image")
        self.assertEqual(payload["sampling"], {"steps": 12, "cfg": 1.0})

    def test_validate_only_uses_layer_defaults_and_accepts_overrides(self):
        temporary, model_directory = self._model_directory(
            {
                "schema_version": 1,
                "inference_profile": "layer_decompose",
                "alignment_padding_mode": "learned",
                "multi_frame_output": True,
                "vae_input_channels": 4,
                "vae_sample_mode": "argmax",
            }
        )
        self.addCleanup(temporary.cleanup)
        input_image = model_directory / "input.png"
        input_image.write_bytes(b"validation-only")

        default_result = subprocess.run(
            [
                sys.executable,
                str(INFER),
                "--model",
                str(model_directory),
                "--task",
                "layer-decompose",
                "--input-image",
                str(input_image),
                "--num-layers",
                "4",
                "--validate-only",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(default_result.stdout)["sampling"],
            {"steps": 12, "cfg": 2.0},
        )

        override_result = subprocess.run(
            [
                sys.executable,
                str(INFER),
                "--model",
                str(model_directory),
                "--task",
                "layer-decompose",
                "--input-image",
                str(input_image),
                "--steps",
                "16",
                "--cfg",
                "1.25",
                "--validate-only",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(override_result.stdout)["sampling"],
            {"steps": 16, "cfg": 1.25},
        )

    def test_validate_only_rejects_wrong_checkpoint_family(self):
        temporary, model_directory = self._model_directory(
            {
                "schema_version": 1,
                "inference_profile": "layer_decompose",
                "alignment_padding_mode": "learned",
                "multi_frame_output": True,
                "vae_input_channels": 4,
                "vae_sample_mode": "argmax",
            }
        )
        self.addCleanup(temporary.cleanup)
        result = subprocess.run(
            [
                sys.executable,
                str(INFER),
                "--model",
                str(model_directory),
                "--task",
                "text-to-image",
                "--prompt",
                "test",
                "--validate-only",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("generation_edit checkpoint", result.stderr)


if __name__ == "__main__":
    unittest.main()
