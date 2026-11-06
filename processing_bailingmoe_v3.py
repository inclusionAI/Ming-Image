#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.
"""Processor for the tinyv3 (bailing_moe_v3_vl) MLLM family.

Mirrors the call surface ``infer.run_generation`` uses for the miniv2
processor (``apply_chat_template`` / ``process_vision_info`` / ``__call__``)
so the public entry stays family-agnostic. The tinyv3 preprocessing contract is:

* tokenizer/chat template come from the checkpoint's own mllm/ data
  (``PreTrainedTokenizerFast`` + ``chat_template.jinja``);
* vision inputs use the Qwen3-VL patch layout with parameters declared in
  the package's ``preprocessor_config.json`` (no undeclared fallbacks);
* the RGBA VAE reference uses ``(to_tensor - 0.5) * 2`` encoding.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from bailingmm_utils import get_default_image_gen_hw
from processing_bailingmm2 import transform_reference_images

# Legacy vision preprocessing defaults. A package's preprocessor_config.json
# overrides every key; complete packages declare their preprocessing settings
# explicitly.
DEFAULT_VISION_PREPROCESS = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "min_pixels": 4 * 32 * 32,
    "max_pixels": 16384 * 32 * 32,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
}

IMAGE_PLACEHOLDER = "<|image_pad|>"


class BailingMoeV3Processor:
    attributes = ["tokenizer"]

    def __init__(self, tokenizer, vision_preprocess: Optional[Dict[str, Any]] = None):
        self.tokenizer = tokenizer
        cfg = dict(DEFAULT_VISION_PREPROCESS)
        if vision_preprocess:
            for key, value in vision_preprocess.items():
                if value is not None and key in cfg:
                    cfg[key] = value
        self.vision_preprocess = cfg

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, data_directory: Union[str, os.PathLike]):
        from transformers import AutoTokenizer

        data_directory = str(data_directory)
        tokenizer = AutoTokenizer.from_pretrained(data_directory)
        preprocess_config_path = os.path.join(data_directory, "preprocessor_config.json")
        vision_preprocess = None
        if os.path.isfile(preprocess_config_path):
            with open(preprocess_config_path, encoding="utf-8") as handle:
                vision_preprocess = json.load(handle)
        return cls(tokenizer, vision_preprocess)

    # ------------------------------------------------------------------ #
    # chat / vision-info surface (same names as the miniv2 processor)
    # ------------------------------------------------------------------ #
    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **kwargs,
    ):
        return self.tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    def process_vision_info(self, conversations):
        """Extract PIL images from chat-message content (images only)."""
        images = []
        for message in conversations:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image" or "image" in item:
                    value = item.get("image", item.get("image_url"))
                    if isinstance(value, Image.Image):
                        images.append(value)
                    elif isinstance(value, str):
                        images.append(Image.open(value))
        return images, [], []

    # ------------------------------------------------------------------ #
    # Qwen3-VL vision preprocessing (reference _encode_vision_images)
    # ------------------------------------------------------------------ #
    def _smart_resize(self, height: int, width: int):
        cfg = self.vision_preprocess
        factor = cfg["patch_size"] * cfg["merge_size"]
        min_pixels = cfg["min_pixels"]
        max_pixels = cfg["max_pixels"]
        if height < factor or width < factor:
            raise ValueError(
                f"height:{height} or width:{width} must be larger than factor:{factor}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return max(h_bar, factor), max(w_bar, factor)

    def encode_vision_images(self, images: List[Image.Image]):
        cfg = self.vision_preprocess
        patch = int(cfg["patch_size"])
        temporal = int(cfg["temporal_patch_size"])
        merge = int(cfg["merge_size"])
        mean = np.array(cfg["image_mean"], dtype=np.float32)
        std = np.array(cfg["image_std"], dtype=np.float32)
        pixels, grids = [], []
        for image in images:
            image = image.convert("RGB")
            width, height = image.size
            resized_h, resized_w = self._smart_resize(height, width)
            image = image.resize((resized_w, resized_h), Image.BICUBIC)
            arr = np.asarray(image, dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            arr = np.transpose(arr, (2, 0, 1))
            frames = np.tile(arr[None], (temporal, 1, 1, 1))
            channel = frames.shape[1]
            grid_t = frames.shape[0] // temporal
            grid_h, grid_w = resized_h // patch, resized_w // patch
            frames = frames.reshape(
                grid_t, temporal, channel,
                grid_h // merge, merge, patch,
                grid_w // merge, merge, patch,
            )
            frames = frames.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flat = frames.reshape(
                grid_t * grid_h * grid_w, channel * temporal * patch * patch
            )
            pixels.append(flat)
            grids.append([grid_t, grid_h, grid_w])
        return {
            "pixel_values": torch.from_numpy(np.concatenate(pixels, axis=0)),
            "image_grid_thw": torch.tensor(grids, dtype=torch.long),
        }

    # ------------------------------------------------------------------ #
    # main call
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        return_tensors=None,
        image_gen_highres=512,
        image_gen_aspect_ratio=None,
        image_gen_ref_images=None,
        image_gen_input_channels=None,
        **kwargs,
    ) -> BatchFeature:
        if videos:
            raise ValueError("video input is not supported by Ming Image inference")
        if isinstance(text, str):
            text = [text]
        texts = list(text)

        image_inputs: Dict[str, torch.Tensor] = {}
        if images:
            vision = self.encode_vision_images(images)
            grids = vision["image_grid_thw"]
            merge = int(self.vision_preprocess["merge_size"])
            merge_sq = merge * merge
            expanded = []
            index = 0
            for sample in texts:
                while IMAGE_PLACEHOLDER in sample and index < grids.shape[0]:
                    n = max(int(grids[index].prod().item()) // merge_sq, 1)
                    sample = sample.replace(IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER * n, 1)
                    index += 1
                expanded.append(sample)
            if index != grids.shape[0]:
                raise RuntimeError(
                    f"image prompt has {index} {IMAGE_PLACEHOLDER!r} placeholders "
                    f"for {grids.shape[0]} images"
                )
            texts = expanded
            image_inputs = vision

        encoded = self.tokenizer(texts, return_tensors="pt", padding=True)
        # keep the miniv2 processor's output contract: input_ids + mask only
        text_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

        image_gen_inputs: Dict[str, Any] = {}
        if image_gen_ref_images is None:
            # mirror the miniv2 processor contract: generation size travels
            # with the batch feature (aspect-aware) so generate() never falls
            # back to an internal default that could disagree with the CLI.
            gen_h, gen_w = get_default_image_gen_hw(
                image_gen_highres, image_gen_aspect_ratio
            )
            count = len(texts)
            image_gen_inputs["image_gen_height"] = [int(gen_h)] * count
            image_gen_inputs["image_gen_width"] = [int(gen_w)] * count
        else:
            if image_gen_input_channels not in (3, 4):
                raise ValueError(
                    "image_gen_input_channels must be explicitly set to 3 or 4 "
                    "from the checkpoint capability contract"
                )
            if not isinstance(image_gen_ref_images, list):
                image_gen_ref_images = [image_gen_ref_images]
            ref_tensor, ref_h, ref_w = transform_reference_images(
                image_gen_ref_images,
                image_gen_aspect_ratio,
                image_gen_highres,
                image_gen_input_channels,
            )
            image_gen_inputs["image_gen_pixel_values_reference"] = ref_tensor
            image_gen_inputs["image_gen_height"] = [int(ref_h)]
            image_gen_inputs["image_gen_width"] = [int(ref_w)]

        data = {**text_inputs, **image_inputs, **image_gen_inputs}
        return BatchFeature(data=data)
