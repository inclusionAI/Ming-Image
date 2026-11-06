#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.
"""Capture a fixed alignment case with the unified Ming Image runtime.

Runs one case (text-to-image or image-edit) through the same public entry
functions ``infer.py`` uses, and exports a portable bundle for third-party
hardware alignment:

    <out>/
      case.json        identity: model/config hashes, code SHA, environment,
                       prompt bytes hash, run parameters
      arrays.npz       input_ids, attention_mask, condition_embeds,
                       directvlm_hidden_states, initial_noise (fp32, the
                       pre-scheduler Gaussian), step_latents (fp32, per-step
                       pipeline latents), final_pixels (uint8 RGB)

The capture does not change sampling behaviour: the initial noise is drawn
with the same generator the runtime would use, the pipeline consumes it via
its standard ``latents`` argument, and per-step latents are collected through
the pipeline's ``callback_on_step_end`` interface.

Usage:
    python tools/capture_case.py --model <package> --task text-to-image \
        --prompt "..." --width 1024 --height 1024 --steps 30 --seed 42 \
        --output-dir <bundle-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from diffusers.utils.torch_utils import randn_tensor  # noqa: E402
from infer import (  # noqa: E402
    _build_messages,
    load_checkpoint_capabilities,
    load_mllm_family,
    load_model_and_processor,
    resolve_model_directory,
    resolve_t2i_size,
)
from inference_profile import InferenceProfileError  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_identity(model_directory: Path) -> dict:
    identity = {"model_directory": str(model_directory)}
    config_hashes = {}
    for config in sorted(model_directory.rglob("*.json")):
        if config.stat().st_size > 4 * 1024 * 1024:
            continue
        try:
            config_hashes[str(config.relative_to(model_directory))] = _sha256_file(config)
        except OSError:
            continue
    identity["config_hashes"] = config_hashes
    try:
        identity["code_sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        identity["code_sha"] = None
    return identity


def _environment_identity() -> dict:
    import diffusers
    import transformers

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for package in ("fla-core", "flash-attn", "triton"):
        try:
            import importlib.metadata as metadata

            env[package] = metadata.version(package)
        except Exception:
            env[package] = None
    return env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=["text-to-image", "image-edit"])
    parser.add_argument("--prompt", required=True, help="prompt text or path to a UTF-8 prompt file")
    parser.add_argument("--input-image", type=Path)
    parser.add_argument(
        "--width",
        type=int,
        help="text-to-image only; the reference image drives image-edit size",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="text-to-image only; the reference image drives image-edit size",
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_directory = resolve_model_directory(args.model)
    family = load_mllm_family(model_directory)
    profile = load_checkpoint_capabilities(model_directory)
    profile.validate_task(
        args.task, has_reference_image=args.input_image is not None, num_layers=1
    )
    sampling = profile.resolve_sampling_parameters(steps=args.steps, cfg=args.cfg, family=family)
    if args.task == "text-to-image" and (args.width is None or args.height is None):
        parser.error("--width and --height are required for text-to-image")
    if args.task == "image-edit" and args.input_image is None:
        parser.error("--input-image is required for image-edit")

    prompt_path = Path(args.prompt)
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else args.prompt

    # Loading mode changes tinyv3 numerics: with device_map placement the
    # precision-sensitive t_embedder/cap_embedder stay fp32 under bf16
    # autocast, while a plain .to(dtype)
    # cast makes the transformer all-bf16. The capture must therefore use the
    # same placement as the CLI default (balanced, single GPU).
    loader_args = argparse.Namespace(
        processor=None,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device=args.device,
        device_map="balanced",
        num_gpus=1,
    )
    model, processor = load_model_and_processor(model_directory, loader_args, family)
    device = torch.device(args.device)

    messages = _build_messages(args.task, prompt, args.input_image, family)
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, _ = processor.process_vision_info(messages)
    reference_image = None
    if args.input_image is not None:
        reference_image = Image.open(args.input_image).convert(
            "RGB" if profile.vae_input_channels == 3 else "RGBA"
        )
    # Size handling must mirror infer.run_generation exactly: for text-to-image
    # the requested width/height select the internal highres bucket, while for
    # image-edit the reference image drives the output size through the
    # processor and only the highres bucket comes from the family default.
    # Using max(width, height) here would pick a different reference
    # preprocessing bucket than the CLI and change the output pixels.
    if args.task == "text-to-image":
        _, _, highres = resolve_t2i_size(args.task, None, args.width, args.height, family)
        aspect = (
            args.width / args.height
            if args.width is not None and args.width != args.height
            else None
        )
    else:
        _, _, highres = resolve_t2i_size(args.task, None, None, None, family)
        aspect = None
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        image_gen_highres=highres,
        image_gen_aspect_ratio=aspect,
        image_gen_ref_images=reference_image,
        image_gen_input_channels=profile.vae_input_channels,
    )
    gen_height = int(inputs["image_gen_height"][0])
    gen_width = int(inputs["image_gen_width"][0])
    inputs = inputs.to(device)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            inputs[key] = value.to(dtype=getattr(torch, args.dtype))

    arrays = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy(),
    }

    # conditions: one dedicated extraction pass (same code path as generation)
    with torch.no_grad():
        condition_embeds, _, condition_embeds_2, _ = model.generate(
            **{k: v for k, v in inputs.items() if v is not None},
            image_gen=True,
            image_gen_task=args.task,
            image_gen_only_extract_hidden_states=True,
        )
    arrays["condition_embeds"] = condition_embeds.float().cpu().numpy()
    if condition_embeds_2 is not None:
        arrays["directvlm_hidden_states"] = condition_embeds_2.float().cpu().numpy()

    # initial noise: replicate the runtime generator exactly
    generator = torch.Generator(device=device).manual_seed(args.seed)
    vae_scale = model.diffusion_loss.pipelines.vae_scale_factor
    noise_shape = (
        1,
        model.diffusion_loss.train_model.in_channels,
        gen_height // vae_scale,
        gen_width // vae_scale,
    )
    initial_noise = randn_tensor(noise_shape, generator=generator, device=device, dtype=torch.float32)
    arrays["initial_noise"] = initial_noise.float().cpu().numpy()

    step_latents = []

    def collect(pipeline, step, timestep, callback_kwargs):
        step_latents.append(callback_kwargs["latents"].float().cpu().numpy().copy())
        return callback_kwargs

    with torch.no_grad():
        output = model.generate(
            **{k: v for k, v in inputs.items() if v is not None},
            image_gen=True,
            image_gen_task=args.task,
            image_gen_seed=args.seed,
            image_gen_steps=sampling.steps,
            image_gen_cfg=sampling.cfg,
            num_frames_per_prompt=1,
            image_gen_latents=initial_noise,
            image_gen_step_callback=collect,
        )
    image = output[0] if isinstance(output, list) else output
    if image.size != (gen_width, gen_height):
        raise RuntimeError(
            f"output size {image.size} does not match the processor-declared "
            f"generation size {(gen_width, gen_height)}"
        )
    arrays["step_latents"] = np.stack(step_latents)
    arrays["final_pixels"] = np.asarray(image.convert("RGB"), dtype=np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "arrays.npz", **arrays)
    identity = {
        "case": {
            "task": args.task,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "input_image": str(args.input_image) if args.input_image else None,
            "width": gen_width,
            "height": gen_height,
            "steps": sampling.steps,
            "cfg": sampling.cfg,
            "seed": args.seed,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        "mllm_family": family,
        "profile": {
            "inference_profile": profile.inference_profile,
            "alignment_padding_mode": profile.alignment_padding_mode,
            "multi_frame_output": profile.multi_frame_output,
            "vae_input_channels": profile.vae_input_channels,
            "vae_sample_mode": profile.vae_sample_mode,
        },
        "model": _package_identity(model_directory),
        "environment": _environment_identity(),
        "notes": {
            "initial_noise": "pre-scheduler Gaussian, fp32, drawn with torch.Generator(device).manual_seed(seed) identical to the runtime path",
            "step_latents": "pipeline latents at each callback_on_step_end (post-step), fp32 via float() cast",
            "bf16_note": "condition tensors were computed in bf16 and stored as fp32 via .float()",
        },
    }
    (args.output_dir / "case.json").write_text(
        json.dumps(identity, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"bundle": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
