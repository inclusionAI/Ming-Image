import json
import unittest

from inference_profile import (
    InferenceProfile,
    InferenceProfileError,
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


if __name__ == "__main__":
    unittest.main()
