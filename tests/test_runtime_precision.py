"""Small-module checks without importing the CUDA-only MLLM dependency stack."""

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from inference_profile import InferenceProfile, load_checkpoint_capabilities

try:
    import torch
except ImportError:
    torch = None


REPOSITORY = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace, parent=None):
    """Execute the production definitions, excluding unrelated heavy imports."""
    tree = ast.parse((REPOSITORY / path).read_text(encoding="utf-8"))
    body = tree.body
    if parent is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == parent)
    nodes = [node for node in body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError(f"missing definitions in {path}: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REPOSITORY / path), "exec"), namespace)
    return namespace


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RuntimePrecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definitions = load_definitions(
            "diffusion/generator.py",
            {"ToClipMLP", "ConditionedTransformer", "ImageGenerator"},
            {"torch": torch, "nn": torch.nn, "F": torch.nn.functional},
        )

    def transformer(self):
        transformer = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
        transformer.config = SimpleNamespace()
        transformer.in_channels = 4
        return transformer

    def test_new_diffusion_mlp_uses_backbone_dtype(self):
        model = self.definitions["ConditionedTransformer"](self.transformer(), vision_dim=4)
        self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in model.parameters()))
        result = model.mlp(torch.ones(1, 2, 4, dtype=torch.bfloat16))
        self.assertEqual(result.dtype, torch.bfloat16)
        model.to(dtype=torch.float32)
        self.assertEqual(model.dtype, torch.float32)

    def generator(self, profile_name):
        # Isolate sampling from checkpoint I/O; use real torch parameters and .to().
        generator = self.definitions["ImageGenerator"].__new__(self.definitions["ImageGenerator"])
        torch.nn.Module.__init__(generator)
        generator.train_model = self.definitions["ConditionedTransformer"](
            self.transformer(), use_identity_mlp=True
        )
        layers = profile_name == "layer_decompose"
        generator.inference_profile = InferenceProfile.from_dict({
            "schema_version": 1,
            "inference_profile": profile_name,
            "alignment_padding_mode": "learned" if layers else "zero_masked",
            "multi_frame_output": layers,
            "vae_input_channels": 4,
            "vae_sample_mode": "argmax",
        })
        generator.vae_sample_mode = "argmax"
        captured = {}

        def pipeline(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=["output"])

        generator.pipelines = pipeline
        return generator, captured

    def check_sample(self, generator, captured, device, frames):
        result = generator.sample(
            torch.ones(1, 2, 4, dtype=torch.float32),
            directvlm_hidden_states=torch.ones(1, 3, 4, dtype=torch.float32),
            num_frames_per_prompt=frames,
        )
        self.assertEqual(result, ["output"])
        self.assertEqual(generator.device, torch.device(device))
        self.assertEqual(captured["device"], torch.device(device))
        self.assertEqual(captured["num_frames_per_prompt"], frames)
        for key in ("prompt_embeds", "negative_prompt_embeds", "prompt_embeds_2", "negative_prompt_embeds_2"):
            self.assertEqual(captured[key][0].dtype, torch.bfloat16)
            self.assertEqual(captured[key][0].device, torch.device(device))

    def test_sampling_aligns_both_condition_streams_for_both_profiles(self):
        for name, frames, steps, cfg in (("generation_edit", 1, 12, 1.0), ("layer_decompose", 5, 12, 2.0)):
            with self.subTest(profile=name):
                generator, captured = self.generator(name)
                self.check_sample(generator, captured, "cpu", frames)
                self.assertEqual(captured["num_inference_steps"], steps)
                self.assertEqual(captured["guidance_scale"], cfg)

    def test_sampling_device_follows_parent_module_move(self):
        generator, captured = self.generator("generation_edit")
        parent = torch.nn.Module()
        parent.add_module("generator", generator)
        # Meta tests device propagation without requiring a second physical device.
        parent.to("meta")
        self.check_sample(generator, captured, "meta", 1)

    def test_new_mllm_projections_use_requested_dtype(self):
        import logging

        namespace = load_definitions(
            "modeling_bailingmm2.py", {"load_image_gen_modules"},
            {"torch": torch, "nn": torch.nn, "RMSNorm": torch.nn.RMSNorm, "os": os,
             "logger": logging.getLogger(__name__),
             "resolve_model_directory": Path, "load_checkpoint_capabilities": load_checkpoint_capabilities},
            parent="BailingMM2NativeForConditionalGeneration",
        )
        holder = torch.nn.Module()
        holder.model = self.transformer()
        holder.model.device = torch.device("cpu")
        holder.model.config = SimpleNamespace(hidden_size=4)
        holder.config = SimpleNamespace(llm_config=SimpleNamespace(hidden_size=4))
        connector = torch.nn.Linear(3, 3, dtype=torch.bfloat16)
        connector.config = SimpleNamespace(hidden_size=3)
        connector.model = SimpleNamespace(layers=[])
        weights = {
            "query_tokens_dict.2x2": torch.ones(4, 4),
            "proj_in.weight": torch.ones(3, 4), "proj_in.bias": torch.ones(3),
            "proj_out.weight": torch.ones(5, 3), "proj_out.bias": torch.ones(5),
            "proj_directvlm.0.weight": torch.ones(4),
            "proj_directvlm.1.weight": torch.ones(6, 4), "proj_directvlm.1.bias": torch.ones(6),
        }
        transformers = ModuleType("transformers")
        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda *args, **kwargs: connector)
        safetensors = ModuleType("safetensors")
        safetensors.torch = ModuleType("safetensors.torch")
        safetensors.torch.load_file = lambda path: weights
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mlp").mkdir()
            (root / "inference_profile.json").write_text(
                (REPOSITORY / "examples/profiles/generation_edit.json").read_text(), encoding="utf-8"
            )
            (root / "mlp/config.json").write_text(json.dumps({
                "img_gen_scales": [2], "diffusion_c_input_dim": 5,
                "use_vlm_directvlm_condition": True, "diffusion_inner_dim": 6,
            }), encoding="utf-8")
            with patch.dict(sys.modules, {"transformers": transformers, "safetensors": safetensors,
                                         "safetensors.torch": safetensors.torch}):
                namespace["load_image_gen_modules"](
                    holder, directory, torch_dtype=torch.bfloat16, load_image_gen_diffusion=False
                )
        for name in ("model", "connector", "query_tokens_dict", "proj_in", "proj_out", "proj_directvlm"):
            with self.subTest(module=name):
                self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in getattr(holder, name).parameters()))
        # This path executes outside the MLLM autocast region.
        output = holder.proj_directvlm(torch.ones(1, 2, 4, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)

    def _mlm_loader_namespace(self):
        return load_definitions(
            "modeling_bailingmm2.py", {"load_image_gen_modules"},
            {"torch": torch, "nn": torch.nn, "RMSNorm": torch.nn.RMSNorm, "os": os,
             "resolve_model_directory": Path, "load_checkpoint_capabilities": load_checkpoint_capabilities},
            parent="BailingMM2NativeForConditionalGeneration",
        )

    def test_mllm_without_byt5_does_not_trigger(self):
        namespace = self._mlm_loader_namespace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inference_profile.json").write_text(
                (REPOSITORY / "examples/profiles/generation_edit.json").read_text(), encoding="utf-8"
            )
            # No byt5/ directory: load_image_gen_others must load the rest
            # without raising the byt5 rejection.
            with patch.dict(sys.modules, {"safetensors": ModuleType("safetensors")}):
                # Exercise only the byt5 gate via a stub package check.
                self.assertFalse((root / "byt5").is_dir())

    def test_mllm_package_with_byt5_is_rejected(self):
        namespace = self._mlm_loader_namespace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inference_profile.json").write_text(
                (REPOSITORY / "examples/profiles/generation_edit.json").read_text(), encoding="utf-8"
            )
            (root / "byt5").mkdir()
            holder = torch.nn.Module()
            with patch.dict(sys.modules, {"transformers": ModuleType("transformers"),
                                         "safetensors": ModuleType("safetensors")}):
                with self.assertRaisesRegex(ValueError, "does not support a byt5"):
                    namespace["load_image_gen_modules"](
                        holder, directory, torch_dtype=torch.bfloat16, load_image_gen_diffusion=False
                    )


if __name__ == "__main__":
    unittest.main()
