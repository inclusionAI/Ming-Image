"""Optional end-to-end GPU smoke tests for the maintained demo cases.

Required environment variables:

- MING_GENERATION_MODEL
- MING_LAYER_MODEL

The model variables may be local directories or Hugging Face Hub IDs.

Optional environment variables:

- MING_SMOKE_OUTPUT_DIR: keep outputs under this directory (created when
  missing) instead of a temporary directory.
- MING_SMOKE_MODE: "short" (default) runs the two-step probes only;
  "full" additionally runs the default-parameter acceptance cases.
- MING_SMOKE_DEVICE, MING_SMOKE_DEVICE_MAP, MING_SMOKE_NUM_GPUS,
  MING_SMOKE_DTYPE and MING_SMOKE_ATTN_IMPLEMENTATION are forwarded to
  infer.py. The smoke defaults to the validated single-GPU layout in BF16
  and FlashAttention 2; set MING_SMOKE_NUM_GPUS (e.g. 8) for a multi-GPU
  run.

Text-to-image and layer decomposition use the same assets displayed in the
public README. The image-edit compatibility probe keeps its established input
and prompt.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
INFER = REPOSITORY / "infer.py"
T2I_PROMPT = REPOSITORY / "assets" / "t2i_four_seasons_cabin_prompt.json"
IMAGE_EDIT_INPUT = REPOSITORY / "tests" / "assets" / "smoke_input.png"
IMAGE_EDIT_PROMPT = "Change the background to blue"
LAYER_INPUT = REPOSITORY / "assets" / "layer_samples" / "card_making_input.png"
LAYER_PROMPT = REPOSITORY / "assets" / "layer_samples" / "card_making_prompt.txt"
LAYER_COUNT = 6


class SmokeAssetContractTest(unittest.TestCase):
    def test_smoke_assets_are_present_and_well_formed(self):
        for path in (T2I_PROMPT, IMAGE_EDIT_INPUT, LAYER_INPUT, LAYER_PROMPT):
            self.assertTrue(path.is_file(), path)

        with T2I_PROMPT.open(encoding="utf-8") as handle:
            structured_prompt = json.load(handle)
        self.assertEqual(set(structured_prompt), {"canvas_settings", "layers"})

        layer_prompt = LAYER_PROMPT.read_text(encoding="utf-8")
        self.assertIn(f"Number of layers: {LAYER_COUNT}", layer_prompt)


def _smoke_output_root():
    value = os.environ.get("MING_SMOKE_OUTPUT_DIR")
    return Path(value).expanduser().resolve() if value else None


@unittest.skipUnless(
    os.environ.get("MING_GENERATION_MODEL")
    and os.environ.get("MING_LAYER_MODEL"),
    "set MING_GENERATION_MODEL and MING_LAYER_MODEL",
)
class InferenceSmokeTest(unittest.TestCase):
    def _run(self, *arguments, steps=2, resolution=None):
        command = [
            sys.executable,
            str(INFER),
            *map(str, arguments),
            "--seed",
            "42",
            "--device",
            os.environ.get("MING_SMOKE_DEVICE", "cuda:0"),
            "--device-map",
            os.environ.get("MING_SMOKE_DEVICE_MAP", "balanced"),
            "--num-gpus",
            os.environ.get("MING_SMOKE_NUM_GPUS", "1"),
            "--dtype",
            os.environ.get("MING_SMOKE_DTYPE", "bfloat16"),
            "--attn-implementation",
            os.environ.get("MING_SMOKE_ATTN_IMPLEMENTATION", "flash_attention_2"),
        ]
        if resolution is not None:
            command += ["--resolution", str(resolution)]
        if steps is not None:
            command += ["--steps", str(steps)]
        subprocess.run(command, cwd=REPOSITORY, check=True)

    def _output_root(self, test_case_dir):
        configured = _smoke_output_root()
        if configured is not None:
            return configured / test_case_dir
        return Path(self._temporary_directory.name) / test_case_dir

    def setUp(self):
        self._temporary_directory = (
            tempfile.TemporaryDirectory() if _smoke_output_root() is None else None
        )
        self.generation_model = os.environ["MING_GENERATION_MODEL"]
        self.layer_model = os.environ["MING_LAYER_MODEL"]
        for path in (T2I_PROMPT, IMAGE_EDIT_INPUT, LAYER_INPUT, LAYER_PROMPT):
            self.assertTrue(path.is_file(), path)

    def tearDown(self):
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()

    def _assert_png_set(self, directory, count, mode=None):
        from PIL import Image

        paths = sorted(Path(directory).glob("*.png"))
        self.assertEqual(len(paths), count, f"{directory}: {paths}")
        pixels = []
        for path in paths:
            with Image.open(path) as image:
                if mode is not None:
                    self.assertEqual(image.mode, mode, path)
                sample = image.convert("RGBA")
                pixels.append(sample.tobytes())
                extrema = sample.getextrema()
                for channel, (low, high) in enumerate(extrema):
                    self.assertGreater(
                        high, low, f"{path} channel {channel} is constant"
                    )
        return paths, pixels

    def _assert_showcase_layers(self, directory):
        from PIL import Image

        paths, pixels = self._assert_png_set(directory, LAYER_COUNT, mode="RGBA")
        self.assertEqual(
            len(set(pixels)),
            LAYER_COUNT,
            "showcase layer output must contain six distinct images",
        )
        alpha_ranges = []
        for path in paths:
            with Image.open(path) as image:
                alpha_ranges.append(image.getchannel("A").getextrema())
        self.assertTrue(
            any(low < high for low, high in alpha_ranges),
            f"no alpha variation in showcase layer outputs: {alpha_ranges}",
        )

    def _run_showcase_cases(self, root, *, steps, t2i_resolution, layer_resolution):
        self._run(
            "--model", self.generation_model,
            "--task", "text-to-image",
            "--prompt", T2I_PROMPT,
            "--output-dir", root / "text-to-image",
            steps=steps,
            resolution=t2i_resolution,
        )
        self._run(
            "--model", self.generation_model,
            "--task", "image-edit",
            "--input-image", IMAGE_EDIT_INPUT,
            "--prompt", IMAGE_EDIT_PROMPT,
            "--output-dir", root / "image-edit",
            steps=steps,
            resolution=1024,
        )
        self._run(
            "--model", self.layer_model,
            "--task", "layer-decompose",
            "--input-image", LAYER_INPUT,
            "--prompt", LAYER_PROMPT,
            "--output-dir", root / "layer-decompose",
            steps=steps,
            resolution=layer_resolution,
        )

        self._assert_png_set(root / "text-to-image", 1)
        self._assert_png_set(root / "image-edit", 1)
        self._assert_showcase_layers(root / "layer-decompose")

    def test_two_step_showcase_smoke(self):
        """Two-step probes for both showcase cases and image-edit regression."""
        self._run_showcase_cases(
            self._output_root("short"),
            steps=2,
            t2i_resolution=1024,
            layer_resolution=512,
        )

    @unittest.skipUnless(
        os.environ.get("MING_SMOKE_MODE") == "full",
        "set MING_SMOKE_MODE=full to run default-parameter acceptance",
    )
    def test_default_parameter_acceptance(self):
        """Default steps, CFG and recommended showcase resolutions; seed 42."""
        self._run_showcase_cases(
            self._output_root("default"),
            steps=None,
            t2i_resolution=2048,
            layer_resolution=1024,
        )


if __name__ == "__main__":
    unittest.main()
