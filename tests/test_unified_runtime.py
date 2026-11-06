"""Contract tests for the unified tinyv3/miniv2 runtime (CPU only)."""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from inference_profile import (  # noqa: E402
    FAMILY_MINIV2,
    FAMILY_TINYV3,
    InferenceProfileError,
    load_checkpoint_capabilities,
    load_mllm_family,
)
from mllm_device_map import (  # noqa: E402
    build_mllm_device_plan,
    load_mllm_num_hidden_layers,
)


def _write_package(root: Path, *, model_type, pair=None, legacy_profile=None):
    root.mkdir(parents=True, exist_ok=True)
    mllm = root / "mllm"
    mllm.mkdir(exist_ok=True)
    (mllm / "config.json").write_text(json.dumps({"model_type": model_type}))
    transformer = root / "transformer"
    transformer.mkdir(exist_ok=True)
    transformer_cfg = {"_class_name": "DiffusionTransformer"}
    if pair is not None:
        transformer_cfg["alignment_padding_mode"] = pair[0]
        transformer_cfg["multi_frame_output"] = pair[1]
    (transformer / "config.json").write_text(json.dumps(transformer_cfg))
    vae = root / "vae"
    vae.mkdir(exist_ok=True)
    (vae / "config.json").write_text(
        json.dumps({"_class_name": "AutoencoderKLQwenImage", "input_channels": 4})
    )
    if legacy_profile is not None:
        (root / "inference_profile.json").write_text(json.dumps(legacy_profile))
    return root


LEGACY_GENERATION_PROFILE = {
    "schema_version": 1,
    "inference_profile": "generation_edit",
    "alignment_padding_mode": "zero_masked",
    "multi_frame_output": False,
    "vae_input_channels": 4,
    "vae_sample_mode": "argmax",
}


class FamilyDispatchTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="unified-family-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_known_model_types(self):
        pkg = _write_package(
            self.root / "v2",
            model_type="bailingmm_moe_v2_lite",
            pair=("zero_masked", False),
        )
        self.assertEqual(load_mllm_family(pkg), FAMILY_MINIV2)
        pkg3 = _write_package(
            self.root / "v3",
            model_type="bailing_moe_v3_vl",
            pair=("zero_masked", False),
        )
        self.assertEqual(load_mllm_family(pkg3), FAMILY_TINYV3)

    def test_component_layout_wins_over_root(self):
        pkg = _write_package(
            self.root / "mix",
            model_type="bailing_moe_v3_vl",
            pair=("zero_masked", False),
        )
        # a conflicting root config.json must not override mllm/config.json
        (pkg / "config.json").write_text(
            json.dumps({"model_type": "bailingmm_moe_v2_lite"})
        )
        self.assertEqual(load_mllm_family(pkg), FAMILY_TINYV3)

    def test_unknown_model_type_is_a_hard_error(self):
        pkg = _write_package(
            self.root / "unknown",
            model_type="something_else",
            pair=("zero_masked", False),
        )
        with self.assertRaises(InferenceProfileError):
            load_mllm_family(pkg)

    def test_configless_legacy_package_defaults_to_miniv2(self):
        pkg = self.root / "configless"
        pkg.mkdir()
        (pkg / "inference_profile.json").write_text(
            json.dumps(LEGACY_GENERATION_PROFILE)
        )
        self.assertEqual(load_mllm_family(pkg), FAMILY_MINIV2)

    def test_capability_still_validated_for_both_families(self):
        pkg = _write_package(
            self.root / "v3ok", model_type="bailing_moe_v3_vl",
            pair=("zero_masked", False),
        )
        profile = load_checkpoint_capabilities(pkg)
        self.assertEqual(profile.inference_profile, "generation_edit")
        bad = _write_package(
            self.root / "v3bad", model_type="bailing_moe_v3_vl",
            pair=("learned", False),
        )
        with self.assertRaises(InferenceProfileError):
            load_checkpoint_capabilities(bad)


class FamilySamplingDefaultsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="unified-sampling-"))
        pkg = _write_package(
            self.root / "v3",
            model_type="bailing_moe_v3_vl",
            pair=("zero_masked", False),
        )
        self.profile = load_checkpoint_capabilities(pkg)
    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_tinyv3_generation_defaults(self):
        sampling = self.profile.resolve_sampling_parameters(family=FAMILY_TINYV3)
        self.assertEqual((sampling.steps, sampling.cfg), (30, 1.0))

    def test_miniv2_defaults_unchanged(self):
        sampling = self.profile.resolve_sampling_parameters(family=FAMILY_MINIV2)
        self.assertEqual((sampling.steps, sampling.cfg), (12, 1.0))
        # family=None keeps the legacy behaviour for existing callers
        legacy = self.profile.resolve_sampling_parameters()
        self.assertEqual((legacy.steps, legacy.cfg), (12, 1.0))

    def test_explicit_overrides_win(self):
        sampling = self.profile.resolve_sampling_parameters(
            steps=7, cfg=2.5, family=FAMILY_TINYV3
        )
        self.assertEqual((sampling.steps, sampling.cfg), (7, 2.5))

    def test_tinyv3_layer_profile_rejected(self):
        from inference_profile import FAMILY_SAMPLING_PARAMETERS

        # tinyv3 does not declare a layer-decompose family; there must be no
        # registered sampling default for that combination.
        self.assertNotIn((FAMILY_TINYV3, "layer_decompose"), FAMILY_SAMPLING_PARAMETERS)


class FamilyDevicePlanTest(unittest.TestCase):
    def test_miniv2_keys_unchanged(self):
        plan = build_mllm_device_plan(32, 8, family=FAMILY_MINIV2)
        self.assertIn("model.model.layers.0", plan.device_map)
        self.assertIn("vision", plan.device_map)
        self.assertNotIn("model.layers.0", plan.device_map)

    def test_tinyv3_keys(self):
        plan = build_mllm_device_plan(24, 1, family=FAMILY_TINYV3)
        self.assertEqual(plan.layer_counts, (24,))
        self.assertIn("model.layers.0", plan.device_map)
        self.assertIn("model.layers.23", plan.device_map)
        self.assertIn("model.visual", plan.device_map)
        self.assertIn("linear_proj", plan.device_map)
        self.assertIn("lm_head", plan.device_map)
        self.assertNotIn("model.model.layers.0", plan.device_map)

    def test_num_hidden_layers_from_text_config(self):
        root = Path(tempfile.mkdtemp(prefix="unified-depth-"))
        try:
            mllm = root / "mllm"
            mllm.mkdir(parents=True)
            (mllm / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "bailing_moe_v3_vl",
                        "text_config": {"num_hidden_layers": 24},
                    }
                )
            )
            self.assertEqual(load_mllm_num_hidden_layers(mllm), 24)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_ambiguous_depth_rejected(self):
        root = Path(tempfile.mkdtemp(prefix="unified-depth-bad-"))
        try:
            mllm = root / "mllm"
            mllm.mkdir(parents=True)
            (mllm / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "bailing_moe_v3_vl",
                        "text_config": {"num_hidden_layers": 24},
                        "llm_config": {"num_hidden_layers": 32},
                    }
                )
            )
            with self.assertRaises(Exception):
                load_mllm_num_hidden_layers(mllm)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class LazyDependencyTest(unittest.TestCase):
    def test_public_import_does_not_require_fla(self):
        """Importing infer + miniv2 backend must not pull fla/triton kernels."""
        saved = {
            name: sys.modules.get(name)
            for name in list(sys.modules)
            if name == "fla" or name.startswith("fla.")
        }
        for name in list(saved):
            sys.modules[name] = None
        try:
            sys.modules.pop("modeling_bailing_moe_v3_vl", None)
            sys.modules.pop("modeling_bailing_moe_v3", None)
            module = importlib.import_module("modeling_bailing_moe_v3_vl")
            self.assertTrue(hasattr(module, "require_fla_kernels"))
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_infer_module_importable(self):
        sys.modules.pop("infer", None)
        module = importlib.import_module("infer")
        self.assertTrue(hasattr(module, "load_model_and_processor"))
        self.assertTrue(hasattr(module, "resolve_family_backend"))


if __name__ == "__main__":
    unittest.main()
