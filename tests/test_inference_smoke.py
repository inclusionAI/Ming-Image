"""Optional end-to-end GPU smoke tests.

Required environment variables:

- MING_GENERATION_MODEL
- MING_LAYER_MODEL
- MING_TEST_IMAGE

The model variables may be local directories or Hugging Face Hub IDs.

Optional environment variables:

- MING_SMOKE_OUTPUT_DIR: keep outputs under this directory (created when
  missing) instead of a temporary directory. Use it to keep the fixed
  two-step regression baselines and the default-parameter results.
- MING_SMOKE_MODE: "short" (default) runs the two-step probes only;
  "full" additionally runs the default-parameter acceptance matrix.
- MING_SMOKE_DEVICE, MING_SMOKE_DEVICE_MAP, MING_SMOKE_NUM_GPUS,
  MING_SMOKE_DTYPE and MING_SMOKE_ATTN_IMPLEMENTATION are forwarded to
  infer.py. The smoke test defaults to the validated single-GPU layout,
  BF16 and FlashAttention 2.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
INFER = REPOSITORY / "infer.py"
T2I_POSTER_PROMPT = (
    'Create a square retro-futurist poster for a speculative ocean '
    'research festival. Build the poster as a simple table with five rows and '
    'three columns, using straight thin copper dividers and equal inner padding. '
    'Row 1, columns 1 through 3: center the headline "BEYOND THE BLUE" on a '
    'cream background. Row 2, columns 1 through 3: center the subtitle "OCEAN '
    'FUTURES" on a deep navy background. Row 3, column 1: place the event text '
    '"SEPTEMBER 18 IN SHANGHAI". Row 3, column 2: place one glass orbital '
    'greenhouse above calm midnight-blue water. Row 3, column 3: place the '
    'program text "LIVE LABS". Row 4, column 1: place one red research vessel '
    'in side view. Row 4, column 2: place exactly two translucent jellyfish '
    'against dark water. Row 4, column 3: place one coral specimen in front of '
    'a violet sunrise. Row 5, columns 1 through 3: center the call to action '
    '"RESERVE YOUR PASS". Keep every object fully inside its assigned cell. '
    'Do not overlap cells, repeat objects, add floating panels, or add decorative '
    'text. Render exactly the five quoted text strings once each and fully '
    'legible. The quotation marks only delimit the required copy and must not '
    'appear in the poster. Do not add any other letters, numbers, logos, or '
    'placeholder copy. Use deep navy, cyan, coral, cream, and metallic copper, '
    'with sharp vector edges and controlled halftone texture.'
)


def _smoke_output_root():
    value = os.environ.get("MING_SMOKE_OUTPUT_DIR")
    return Path(value).expanduser().resolve() if value else None


@unittest.skipUnless(
    os.environ.get("MING_GENERATION_MODEL")
    and os.environ.get("MING_LAYER_MODEL")
    and os.environ.get("MING_TEST_IMAGE"),
    "set MING_GENERATION_MODEL, MING_LAYER_MODEL and MING_TEST_IMAGE",
)
class InferenceSmokeTest(unittest.TestCase):
    def _run(self, *arguments, steps=2):
        command = [
            sys.executable,
            str(INFER),
            *map(str, arguments),
            "--resolution",
            "512",
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
        self.test_image = Path(os.environ["MING_TEST_IMAGE"])
        self.assertTrue(self.test_image.is_file(), self.test_image)

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

    def test_two_step_probes(self):
        """Two-step load/shape probes; also the fixed regression baselines."""
        root = self._output_root("short")

        self._run(
            "--model", self.generation_model,
            "--task", "text-to-image",
            "--prompt", T2I_POSTER_PROMPT,
            "--output-dir", root / "text-to-image",
        )
        self._run(
            "--model", self.generation_model,
            "--task", "image-edit",
            "--input-image", self.test_image,
            "--prompt", "Change the background to blue",
            "--output-dir", root / "image-edit",
        )
        self._run(
            "--model", self.layer_model,
            "--task", "layer-decompose",
            "--input-image", self.test_image,
            "--num-layers", "2",
            "--output-dir", root / "layer-decompose-2",
        )
        # Single-layer output exercises the list/count boundary.
        self._run(
            "--model", self.layer_model,
            "--task", "layer-decompose",
            "--input-image", self.test_image,
            "--num-layers", "1",
            "--output-dir", root / "layer-decompose-1",
        )

        self._assert_png_set(root / "text-to-image", 1)
        self._assert_png_set(root / "image-edit", 1)
        _, two_layer_pixels = self._assert_png_set(
            root / "layer-decompose-2", 2, mode="RGBA"
        )
        self.assertNotEqual(
            two_layer_pixels[0], two_layer_pixels[1],
            "two-layer output must not duplicate one image",
        )
        self._assert_png_set(root / "layer-decompose-1", 1, mode="RGBA")

    @unittest.skipUnless(
        os.environ.get("MING_SMOKE_MODE") == "full",
        "set MING_SMOKE_MODE=full to run the default-parameter matrix",
    )
    def test_default_parameter_matrix(self):
        """Checkpoint-family default steps (12/12), CFG defaults, seed 42."""
        root = self._output_root("default")

        self._run(
            "--model", self.generation_model,
            "--task", "text-to-image",
            "--prompt", T2I_POSTER_PROMPT,
            "--output-dir", root / "text-to-image",
            steps=None,
        )
        self._run(
            "--model", self.generation_model,
            "--task", "image-edit",
            "--input-image", self.test_image,
            "--prompt", "Change the background to blue",
            "--output-dir", root / "image-edit",
            steps=None,
        )
        self._run(
            "--model", self.layer_model,
            "--task", "layer-decompose",
            "--input-image", self.test_image,
            "--num-layers", "2",
            "--output-dir", root / "layer-decompose-2",
            steps=None,
        )
        self._run(
            "--model", self.layer_model,
            "--task", "layer-decompose",
            "--input-image", self.test_image,
            "--num-layers", "5",
            "--output-dir", root / "layer-decompose-5",
            steps=None,
        )

        self._assert_png_set(root / "text-to-image", 1)
        self._assert_png_set(root / "image-edit", 1)
        two_paths, two_pixels = self._assert_png_set(
            root / "layer-decompose-2", 2, mode="RGBA"
        )
        self.assertNotEqual(two_pixels[0], two_pixels[1])
        five_paths, five_pixels = self._assert_png_set(
            root / "layer-decompose-5", 5, mode="RGBA"
        )
        self.assertEqual(
            len(set(five_pixels)), 5,
            "five-layer output must contain five distinct images",
        )
        # Transparency check: alpha must vary within at least one layer.
        from PIL import Image

        alpha_values = set()
        for path in two_paths + five_paths:
            with Image.open(path) as image:
                low, high = image.getchannel("A").getextrema()
            alpha_values.add((low, high))
        self.assertTrue(
            any(low < high for low, high in alpha_values),
            f"no alpha variation in layer outputs: {sorted(alpha_values)}",
        )


if __name__ == "__main__":
    unittest.main()
