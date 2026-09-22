import json
import tempfile
from pathlib import Path
import unittest

from inference_profile import (
    InferenceProfile,
    InferenceProfileError,
    load_checkpoint_capabilities,
    load_inference_profile,
    resolve_model_directory,
)


GENERATION = {
    "schema_version": 1,
    "inference_profile": "generation_edit",
    "alignment_padding_mode": "zero_masked",
    "multi_frame_output": False,
    "vae_input_channels": 4,
    "vae_sample_mode": "argmax",
}

LAYER = {
    "schema_version": 1,
    "inference_profile": "layer_decompose",
    "alignment_padding_mode": "learned",
    "multi_frame_output": True,
    "vae_input_channels": 4,
    "vae_sample_mode": "argmax",
}


class InferenceProfileTest(unittest.TestCase):
    def test_all_profile_fields_are_required(self):
        for field in GENERATION:
            raw = dict(GENERATION)
            raw.pop(field)
            with self.assertRaisesRegex(InferenceProfileError, "missing required fields"):
                InferenceProfile.from_dict(raw)

    def test_profile_rejects_unknown_fields(self):
        raw = dict(GENERATION, inferred_from_directory_name=True)
        with self.assertRaisesRegex(InferenceProfileError, "unsupported fields"):
            InferenceProfile.from_dict(raw)

    def test_generation_profile_task_matrix(self):
        profile = InferenceProfile.from_dict(GENERATION)
        self.assertEqual(profile.resolve_sampling_parameters().steps, 12)
        self.assertEqual(profile.resolve_sampling_parameters().cfg, 1.0)
        profile.validate_task("text-to-image", has_reference_image=False)
        profile.validate_task("image-edit", has_reference_image=True)
        with self.assertRaisesRegex(InferenceProfileError, "layer_decompose checkpoint"):
            profile.validate_task("layer-decompose", has_reference_image=True, num_layers=4)

    def test_layer_profile_task_matrix(self):
        profile = InferenceProfile.from_dict(LAYER)
        self.assertEqual(profile.resolve_sampling_parameters().steps, 12)
        self.assertEqual(profile.resolve_sampling_parameters().cfg, 2.0)
        profile.validate_task("layer-decompose", has_reference_image=True, num_layers=4)
        with self.assertRaisesRegex(InferenceProfileError, "generation_edit checkpoint"):
            profile.validate_task("image-edit", has_reference_image=True)

    def test_generation_profile_requires_qwen_vae_contract(self):
        with self.assertRaisesRegex(InferenceProfileError, "vae_input_channels=4"):
            InferenceProfile.from_dict(dict(GENERATION, vae_input_channels=3))
        with self.assertRaisesRegex(InferenceProfileError, "vae_sample_mode='argmax'"):
            InferenceProfile.from_dict(dict(GENERATION, vae_sample_mode="sample"))

    def test_layer_profile_requires_qwen_vae_contract(self):
        with self.assertRaisesRegex(InferenceProfileError, "vae_sample_mode='argmax'"):
            InferenceProfile.from_dict(dict(LAYER, vae_sample_mode="sample"))

    def test_sampling_overrides_are_validated(self):
        profile = InferenceProfile.from_dict(GENERATION)
        resolved = profile.resolve_sampling_parameters(steps=18, cfg=1.5)
        self.assertEqual((resolved.steps, resolved.cfg), (18, 1.5))
        with self.assertRaisesRegex(InferenceProfileError, "steps"):
            profile.resolve_sampling_parameters(steps=0)
        with self.assertRaisesRegex(InferenceProfileError, "CFG"):
            profile.resolve_sampling_parameters(cfg=float("inf"))

    def test_profile_is_loaded_from_local_model_directory(self):
        with self.subTest("local profile"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as directory:
                model_directory = Path(directory)
                (model_directory / "inference_profile.json").write_text(
                    json.dumps(LAYER), encoding="utf-8"
                )
                self.assertEqual(
                    load_inference_profile(model_directory).inference_profile,
                    "layer_decompose",
                )
                self.assertEqual(
                    resolve_model_directory(model_directory), model_directory.resolve()
                )

    def test_missing_profile_is_an_error(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(InferenceProfileError, "must contain"):
                load_inference_profile(directory)

    def test_missing_absolute_local_path_is_not_treated_as_hub_id(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_model_directory(Path(directory) / "missing")


VAE_QWEN_4CH = {"_class_name": "AutoencoderKLQwenImage", "input_channels": 4}


def build_component_package(
    root: Path,
    transformer_extra=None,
    vae=None,
    profile=None,
) -> None:
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    (root / "vae").mkdir(parents=True, exist_ok=True)
    transformer = {"_class_name": "DiffusionTransformer", "dim": 4}
    transformer.update(transformer_extra or {})
    (root / "transformer" / "config.json").write_text(json.dumps(transformer))
    (root / "vae" / "config.json").write_text(json.dumps(vae or VAE_QWEN_4CH))
    if profile is not None:
        (root / "inference_profile.json").write_text(json.dumps(profile))


class ComponentCapabilityTest(unittest.TestCase):
    def capabilities(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_component_package(root, **kwargs)
            return load_checkpoint_capabilities(root)

    def test_generation_pair_without_profile(self):
        profile = self.capabilities(
            transformer_extra={
                "alignment_padding_mode": "zero_masked",
                "multi_frame_output": False,
            }
        )
        self.assertEqual(profile.inference_profile, "generation_edit")
        self.assertEqual(profile.vae_input_channels, 4)
        self.assertEqual(profile.vae_sample_mode, "argmax")
        self.assertEqual(profile.resolve_sampling_parameters().steps, 12)
        self.assertEqual(profile.resolve_sampling_parameters().cfg, 1.0)

    def test_layer_pair_without_profile(self):
        profile = self.capabilities(
            transformer_extra={
                "alignment_padding_mode": "learned",
                "multi_frame_output": True,
            }
        )
        self.assertEqual(profile.inference_profile, "layer_decompose")
        self.assertEqual(profile.resolve_sampling_parameters().cfg, 2.0)

    def test_partial_pair_is_rejected(self):
        for field in ("alignment_padding_mode", "multi_frame_output"):
            with self.subTest(only=field):
                pair = {
                    "alignment_padding_mode": "zero_masked",
                    "multi_frame_output": False,
                }
                with self.assertRaisesRegex(InferenceProfileError, "declared together"):
                    self.capabilities(transformer_extra={field: pair[field]})

    def test_every_invalid_pair_is_rejected(self):
        invalid = [
            {"alignment_padding_mode": "zero_masked", "multi_frame_output": True},
            {"alignment_padding_mode": "learned", "multi_frame_output": False},
            {"alignment_padding_mode": "zero", "multi_frame_output": False},
            {"alignment_padding_mode": "learned", "multi_frame_output": 1},
            {"alignment_padding_mode": None, "multi_frame_output": False},
        ]
        for pair in invalid:
            with self.subTest(pair=pair):
                with self.assertRaises(InferenceProfileError):
                    self.capabilities(transformer_extra=pair)

    def test_vae_class_must_be_qwen(self):
        with self.assertRaisesRegex(InferenceProfileError, "AutoencoderKLQwenImage"):
            self.capabilities(
                transformer_extra={
                    "alignment_padding_mode": "zero_masked",
                    "multi_frame_output": False,
                },
                vae={"_class_name": "AutoencoderKL", "in_channels": 4},
            )

    def test_vae_channel_fields_must_agree_and_be_four(self):
        base = {"alignment_padding_mode": "zero_masked", "multi_frame_output": False}
        with self.assertRaisesRegex(InferenceProfileError, "disagree"):
            self.capabilities(
                transformer_extra=base,
                vae=dict(VAE_QWEN_4CH, in_channels=16),
            )
        with self.assertRaisesRegex(InferenceProfileError, "4-channel"):
            self.capabilities(
                transformer_extra=base,
                vae={"_class_name": "AutoencoderKLQwenImage", "input_channels": 3},
            )
        with self.assertRaisesRegex(InferenceProfileError, "must declare"):
            self.capabilities(
                transformer_extra=base,
                vae={"_class_name": "AutoencoderKLQwenImage"},
            )
        # in_channels alone is accepted when it agrees by itself.
        profile = self.capabilities(
            transformer_extra=base,
            vae={"_class_name": "AutoencoderKLQwenImage", "in_channels": 4},
        )
        self.assertEqual(profile.vae_input_channels, 4)

    def test_legacy_profile_fallback(self):
        profile = self.capabilities(profile=LAYER)
        self.assertEqual(profile.inference_profile, "layer_decompose")
        self.assertEqual(profile.alignment_padding_mode, "learned")

    def test_legacy_fallback_requires_profile_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_component_package(root)
            with self.assertRaisesRegex(InferenceProfileError, "must contain"):
                load_checkpoint_capabilities(root)

    def test_component_fields_and_agreeing_legacy_profile(self):
        profile = self.capabilities(
            transformer_extra={
                "alignment_padding_mode": "zero_masked",
                "multi_frame_output": False,
            },
            profile=GENERATION,
        )
        self.assertEqual(profile.inference_profile, "generation_edit")

    def test_component_fields_disagreeing_with_legacy_profile(self):
        with self.assertRaisesRegex(InferenceProfileError, "disagree"):
            self.capabilities(
                transformer_extra={
                    "alignment_padding_mode": "learned",
                    "multi_frame_output": True,
                },
                profile=GENERATION,
            )


if __name__ == "__main__":
    unittest.main()
