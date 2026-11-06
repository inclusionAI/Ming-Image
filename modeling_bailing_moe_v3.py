#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.
"""Image-generation wrapper for the tinyv3 (bailing_moe_v3_vl) MLLM family.

Public-API twin of ``BailingMM2NativeForConditionalGeneration``
(``modeling_bailingmm2.py``): same ``from_pretrained``/``generate``/
``load_image_gen_modules``/``get_condition_embeds_for_image_gen`` contract so
``infer.py`` stays family-agnostic. Family-specific conventions:

* generation token IDs come from the config special-token fields
  (``image_patch_token`` / ``image_start_token`` / vision end), not +1/+2
  arithmetic;
* learnable-token grids use ``vision_config.spatial_merge_size``;
* patch embeddings are injected with ``masked_scatter`` and positions come
  from ``BailingMoeV3VLModel.get_rope_index`` (3D M-RoPE);
* the direct-VLM mask uses the reference index convention.

Diffusion components (connector/mlp/transformer/vae/scheduler) are the shared
public ones from ``diffusion.generator``.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from diffusers.models.normalization import RMSNorm
from transformers.utils import logging

from bailingmm_utils import (
    find_first_index_of_consecutive_ones,
    merge_consecutive_ones,
)
from configuration_bailing_moe_v3_vl import BailingMoeV3VLConfig
from inference_profile import load_checkpoint_capabilities, resolve_model_directory
from modeling_bailing_moe_v3_vl import (
    BailingMoeV3VLForConditionalGeneration,
    require_fla_kernels,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BailingMoeV3VLConfig"

MLLM_FAMILY = "tinyv3"


class BailingMoeV3VLNativeForConditionalGeneration(BailingMoeV3VLForConditionalGeneration):
    """tinyv3 VL backbone plus the shared image-generation condition modules."""

    config_class = BailingMoeV3VLConfig
    mllm_family = MLLM_FAMILY

    def __init__(self, config: BailingMoeV3VLConfig, **kwargs):
        require_fla_kernels()
        super().__init__(config, **kwargs)
        self.loaded_image_gen_modules = False
        self.inference_profile = None
        self.byt5_model = None
        self.diffusion_loss = None
        self.connector = None
        self.query_tokens_dict = None
        self.proj_in = None
        self.proj_out = None
        self.proj_directvlm = None
        self.img_gen_scales = [16]
        self.scale_indices = []
        self.use_vlm_directvlm_condition = False
        self.use_learnable_token_condition = True
        self.selected_hidden_states_layers = None
        self.connector_norm = False
        self.diffusion_inner_dim = None

    # ------------------------------------------------------------------ #
    # family metadata used by infer.py / device validation
    # ------------------------------------------------------------------ #
    def iter_mllm_decoder_layers(self):
        """Decoder layer modules (family-aware accessor for device checks)."""
        return self.model.layers

    def fixed_image_modules(self):
        """Image-conditioning modules that must stay on the primary device."""
        return {
            "vision": self.model.visual,
            "linear_proj": self.linear_proj,
            "connector": getattr(self, "connector", None),
            "proj_in": getattr(self, "proj_in", None),
            "proj_out": getattr(self, "proj_out", None),
            "proj_directvlm": getattr(self, "proj_directvlm", None),
            "query_tokens": getattr(self, "query_tokens_dict", None),
            "diffusion": getattr(self, "diffusion_loss", None),
        }

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        load_image_gen = bool(kwargs.pop("load_image_gen", False))
        load_image_gen_diffusion = bool(kwargs.pop("load_image_gen_diffusion", True))
        load_image_gen_others = bool(kwargs.pop("load_image_gen_others", True))
        image_gen_device = kwargs.pop("image_gen_device", None)

        vlm_directory = str(pretrained_model_name_or_path)
        if load_image_gen:
            package_root = str(
                resolve_model_directory(
                    pretrained_model_name_or_path,
                    revision=kwargs.get("revision"),
                    cache_dir=kwargs.get("cache_dir"),
                    local_files_only=kwargs.get("local_files_only", False),
                    token=kwargs.get("token"),
                )
            )
            vlm_directory = os.path.join(package_root, "mllm")
            if not os.path.isdir(vlm_directory):
                raise FileNotFoundError(
                    "checkpoint package is missing the mllm/ component: "
                    f"{vlm_directory}. Migrate the package to the component "
                    "layout before loading."
                )
        else:
            package_root = vlm_directory

        model = super().from_pretrained(vlm_directory, *model_args, **kwargs)
        if load_image_gen:
            model.load_image_gen_modules(
                package_root,
                torch_dtype=kwargs.get("torch_dtype", torch.float32),
                load_image_gen_diffusion=load_image_gen_diffusion,
                load_image_gen_others=load_image_gen_others,
                device=image_gen_device,
            )
        return model

    def load_image_gen_modules(
        self,
        inference_model_path,
        torch_dtype=torch.float32,
        load_image_gen_diffusion=True,
        load_image_gen_others=True,
        device=None,
    ):
        inference_model_path = str(resolve_model_directory(inference_model_path))
        if os.path.exists(os.path.join(inference_model_path, "byt5")):
            raise ValueError(
                "Ming Image inference does not support a byt5 component; "
                "the public checkpoint layout has no byt5/ directory."
            )
        self.inference_profile = load_checkpoint_capabilities(inference_model_path)
        if self.inference_profile.inference_profile == "layer_decompose":
            raise ValueError(
                "tinyv3 checkpoints do not declare layer-decompose capability"
            )
        if device is not None:
            device = torch.device(device)
        elif self.model is not None:
            device = self.model.device
        else:
            device = torch.device(torch.cuda.current_device())
        logger.info(f"load_image_gen_modules device={device}")

        from safetensors.torch import load_file

        temp_state_dict = load_file(
            os.path.join(inference_model_path, "mlp", "model.safetensors")
        )
        import json

        with open(os.path.join(inference_model_path, "mlp", "config.json"), "r") as f:
            metax_config = json.load(f)
        diffusion_c_input_dim = metax_config.get("diffusion_c_input_dim", 2560)
        self.img_gen_scales = metax_config.get("img_gen_scales", [16])
        self.connector_norm = bool(metax_config.get("connector_norm", False))
        self.use_vlm_directvlm_condition = bool(
            metax_config.get("use_vlm_directvlm_condition", True)
        )
        self.use_learnable_token_condition = bool(
            metax_config.get("use_learnable_token_condition", True)
        )
        self.selected_hidden_states_layers = metax_config.get(
            "selected_hidden_states_layers"
        )
        self.diffusion_inner_dim = metax_config.get("diffusion_inner_dim", 3840)

        hidden_size = self.config.text_config.hidden_size

        if load_image_gen_others:
            self.connector = None
            self.query_tokens_dict = nn.ParameterDict()
            self.scale_indices = []
            current_idx = 0
            for scale in self.img_gen_scales:
                num_tokens = scale * scale
                self.query_tokens_dict[f"{scale}x{scale}"] = nn.Parameter(
                    F.normalize(torch.randn(num_tokens, hidden_size), dim=-1)
                )
                current_idx += num_tokens
                self.scale_indices.append(current_idx)
            self.query_tokens_dict.to(torch_dtype).to(device)

            if self.use_learnable_token_condition:
                self.query_tokens_dict.load_state_dict(
                    {
                        f"{scale}x{scale}": temp_state_dict[
                            f"query_tokens_dict.{scale}x{scale}"
                        ]
                        for scale in self.img_gen_scales
                    },
                    strict=True,
                )
                from transformers import AutoModelForCausalLM

                self.connector = AutoModelForCausalLM.from_pretrained(
                    inference_model_path,
                    subfolder="connector",
                    torch_dtype=torch_dtype,
                )
                for layer in self.connector.model.layers:
                    layer.self_attn.is_causal = False
                self.connector.to(device)

                self.proj_in = nn.Linear(hidden_size, self.connector.config.hidden_size)
                self.proj_out = nn.Linear(
                    self.connector.config.hidden_size, diffusion_c_input_dim
                )
                self.proj_in.load_state_dict(
                    {
                        "weight": temp_state_dict["proj_in.weight"],
                        "bias": temp_state_dict["proj_in.bias"],
                    },
                    strict=True,
                )
                self.proj_out.load_state_dict(
                    {
                        "weight": temp_state_dict["proj_out.weight"],
                        "bias": temp_state_dict["proj_out.bias"],
                    },
                    strict=True,
                )
                self.proj_in.to(device=device, dtype=torch_dtype)
                self.proj_out.to(device=device, dtype=torch_dtype)

            self.proj_directvlm = None
            if self.use_vlm_directvlm_condition:
                directvlm_dim = hidden_size
                if self.selected_hidden_states_layers is not None:
                    directvlm_dim = directvlm_dim * len(
                        self.selected_hidden_states_layers
                    )
                self.proj_directvlm = nn.Sequential(
                    RMSNorm(directvlm_dim, eps=1e-5),
                    nn.Linear(directvlm_dim, self.diffusion_inner_dim, bias=True),
                )
                self.proj_directvlm.load_state_dict(
                    {
                        "0.weight": temp_state_dict["proj_directvlm.0.weight"],
                        "1.weight": temp_state_dict["proj_directvlm.1.weight"],
                        "1.bias": temp_state_dict["proj_directvlm.1.bias"],
                    },
                    strict=True,
                )
                self.proj_directvlm.to(device=device, dtype=torch_dtype)

        if load_image_gen_diffusion:
            diffusion_mlp_state_dict = {
                key[len("mlp."):]: temp_state_dict[key]
                for key in temp_state_dict
                if key.startswith("mlp.")
            }
            from diffusion.generator import ImageGenerator

            self.diffusion_loss = ImageGenerator(
                model_path=inference_model_path,
                scheduler_path=inference_model_path,
                vision_dim=diffusion_c_input_dim,
                mlp_state_dict=diffusion_mlp_state_dict,
                torch_dtype=torch_dtype,
                device=device,
                use_identity_mlp=metax_config.get("use_identity_mlp", False),
                text_encoder_norm=metax_config.get("text_encoder_norm", False),
                inference_profile=self.inference_profile,
            )
            self.diffusion_loss.to(device)
        self.loaded_image_gen_modules = True

    # ------------------------------------------------------------------ #
    # image features
    # ------------------------------------------------------------------ #
    def extract_image_feature(self, pixel_values, grid_thw):
        feats = self.get_image_features(pixel_values, image_grid_thw=grid_thw)
        if isinstance(feats, (tuple, list)):
            image_embeds = torch.cat(feats, dim=0)
        else:
            image_embeds = feats
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds

    def _image_gen_token_ids(self):
        vl_cfg = self.config
        text_cfg = vl_cfg.text_config
        patch = getattr(text_cfg, "image_patch_token", None) or vl_cfg.image_token_id
        start = getattr(text_cfg, "image_start_token", None) or vl_cfg.vision_start_token_id
        end = getattr(text_cfg, "image_end_token", None) or getattr(
            vl_cfg, "vision_end_token_id", patch
        )
        return int(start), int(end), int(patch)

    def _spatial_merge_size(self) -> int:
        return int(self.config.vision_config.spatial_merge_size)

    # ------------------------------------------------------------------ #
    # condition path (reference semantics)
    # ------------------------------------------------------------------ #
    def append_input_ids_with_multiscale_learnable_tokens(
        self,
        text_ids,
        attention_mask,
        scales,
        start_token_id,
        end_token_id,
        patch_token_id,
    ):
        default_scaled_tokens = []
        default_scaled_attn_masks = []
        default_gen_masks = []
        for scale in scales:
            default_scaled_tokens.append(start_token_id)
            default_scaled_tokens.extend([patch_token_id for _ in range(scale * scale)])
            default_scaled_tokens.append(end_token_id)
            default_scaled_attn_masks.extend([1 for _ in range(scale * scale + 2)])
            default_gen_masks.append(0)
            default_gen_masks.extend([1 for _ in range(scale * scale)])
            default_gen_masks.append(0)

        text_ids_list = text_ids.cpu().tolist()
        attention_mask_list = attention_mask.cpu().tolist()

        new_text_ids_list = []
        new_attention_mask_list = []
        gen_mask_list = []
        new_labels_list = []
        for text_ids_one_batch, attention_mask_one_batch in zip(
            text_ids_list, attention_mask_list
        ):
            padding_start = 0
            for value in attention_mask_one_batch:
                if value == 0:
                    break
                padding_start += 1

            new_text_ids_list.append(
                text_ids_one_batch[:padding_start]
                + deepcopy(default_scaled_tokens)
                + text_ids_one_batch[padding_start:]
            )
            new_labels_list.append(
                [-100 for _ in range(padding_start)]
                + [1 for _ in range(len(default_scaled_tokens))]
                + [-100 for _ in range(len(text_ids_one_batch[padding_start:]))]
            )
            new_attention_mask_list.append(
                attention_mask_one_batch[:padding_start]
                + deepcopy(default_scaled_attn_masks)
                + attention_mask_one_batch[padding_start:]
            )
            gen_mask_list.append(
                [0 for _ in range(padding_start)]
                + deepcopy(default_gen_masks)
                + [0 for _ in range(len(attention_mask_one_batch[padding_start:]))]
            )

        text_ids_append_lq = torch.tensor(
            new_text_ids_list, dtype=text_ids.dtype, device=text_ids.device
        )
        attention_mask_append_lq = torch.tensor(
            new_attention_mask_list,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        gen_mask = torch.tensor(
            gen_mask_list, dtype=attention_mask.dtype, device=attention_mask.device
        )
        labels = torch.tensor(
            new_labels_list, dtype=text_ids.dtype, device=text_ids.device
        )
        return text_ids_append_lq, labels, attention_mask_append_lq, gen_mask

    def appand_learnable_tokens(
        self,
        text_ids,
        gen_mask,
        image_embeds,
        image_grid_thw,
        patch_token_id,
    ):
        merge = self._spatial_merge_size()
        query_tokens_embeds = torch.cat(
            [self.query_tokens_dict[f"{scale}x{scale}"] for scale in self.img_gen_scales],
            dim=0,
        )
        ref_tensor = image_embeds if image_embeds is not None else text_ids
        query_tokens_embeds = query_tokens_embeds.to(
            dtype=torch.bfloat16, device=ref_tensor.device
        )
        if image_embeds is not None:
            query_tokens_embeds = query_tokens_embeds.to(dtype=image_embeds.dtype)

        text_ids_aslist = text_ids.cpu().view(-1).tolist()
        gen_mask_aslist = gen_mask.cpu().view(-1).tolist()
        is_patch_list = [1 if i == patch_token_id else 0 for i in text_ids_aslist]
        idxes_start_of_patch = find_first_index_of_consecutive_ones(is_patch_list)
        isgen_indicators = merge_consecutive_ones(
            [1 if gen_mask_aslist[i] else 0 for i in idxes_start_of_patch],
            len(self.img_gen_scales),
        )

        new_image_grid_thw = []
        new_image_embeds = []
        cum_image_token = 0
        cnt_input_image = 0

        for is_gen in isgen_indicators:
            if is_gen:
                for scale in self.img_gen_scales:
                    new_image_grid_thw.append([1, merge, scale * scale * merge])
                new_image_embeds.append(query_tokens_embeds)
            else:
                thw = image_grid_thw[cnt_input_image].tolist()
                n_image_token = int(thw[0] * thw[1] * thw[2] // (merge * merge))
                image_embed_one = image_embeds[
                    cum_image_token : cum_image_token + n_image_token, :
                ]
                new_image_embeds.append(image_embed_one)
                new_image_grid_thw.append(thw)
                cnt_input_image += 1
                cum_image_token += n_image_token

        if image_grid_thw is not None:
            assert cnt_input_image == image_grid_thw.shape[0]
            assert cum_image_token == image_embeds.shape[0]
        else:
            assert cnt_input_image == 0
            assert cum_image_token == 0

        new_image_grid_thw = torch.tensor(
            new_image_grid_thw, dtype=text_ids.dtype, device=text_ids.device
        )
        new_image_embeds = torch.cat(new_image_embeds, dim=0).to(text_ids.device)
        return new_image_grid_thw, new_image_embeds

    def get_condition_embeds_for_image_gen(
        self,
        input_ids,
        attention_mask,
        image_embeds,
        position_ids,
        use_cache,
        image_grid_thw,
        llm_hidden_states,
    ):
        del use_cache
        start_id, end_id, patch_id = self._image_gen_token_ids()
        (
            input_ids,
            labels,
            attention_mask,
            gen_mask,
        ) = self.append_input_ids_with_multiscale_learnable_tokens(
            input_ids,
            attention_mask,
            self.img_gen_scales,
            start_id,
            end_id,
            patch_id,
        )

        outputs = None
        if llm_hidden_states is None:
            image_grid_thw, image_embeds = self.appand_learnable_tokens(
                input_ids,
                gen_mask,
                image_embeds,
                image_grid_thw,
                patch_id,
            )
            vocab_size = self.get_input_embeddings().weight.shape[0]
            words_embeddings = self.get_input_embeddings()(
                input_ids.clamp(0, vocab_size - 1)
            )
            if image_embeds is not None:
                patch_mask = input_ids == patch_id
                expected = int(patch_mask.sum().item())
                if expected != image_embeds.shape[0]:
                    raise ValueError(
                        "image token / embed mismatch: "
                        f"{expected} patch tokens vs {image_embeds.shape[0]} embeds"
                    )
                words_embeddings = words_embeddings.masked_scatter(
                    patch_mask.unsqueeze(-1).expand_as(words_embeddings),
                    image_embeds.to(dtype=words_embeddings.dtype).reshape(-1),
                )

            if position_ids is None:
                position_ids, _ = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    inputs_embeds=words_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = llm_hidden_states

        directvlm_hidden_states = None
        if self.use_vlm_directvlm_condition:
            use_input_mask = torch.lt(labels, 0).int().to(attention_mask.dtype) * attention_mask
            directvlm_max_valid_ind = (
                use_input_mask.cumsum(-1).argmax(-1).max().item()
            )
            use_input_mask = use_input_mask[:, :directvlm_max_valid_ind]
            if self.selected_hidden_states_layers is not None:
                if outputs is None:
                    raise RuntimeError(
                        "selected_hidden_states_layers requires a live VL forward"
                    )
                directvlm_hidden_states = torch.cat(
                    [
                        outputs.hidden_states[layer_i].to(labels.device)[
                            :, :directvlm_max_valid_ind, :
                        ]
                        * use_input_mask.unsqueeze(-1)
                        for layer_i in self.selected_hidden_states_layers
                    ],
                    dim=-1,
                )
            else:
                src = (
                    outputs.hidden_states[-1]
                    if outputs is not None
                    else hidden_states
                )
                directvlm_hidden_states = src.to(labels.device)[
                    :, :directvlm_max_valid_ind, :
                ] * use_input_mask.unsqueeze(-1)
            directvlm_hidden_states = self.proj_directvlm(
                directvlm_hidden_states.detach()
            )

        scale_embeds = None
        if self.use_learnable_token_condition:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gen_mask_exp = (
                    gen_mask.unsqueeze(-1)
                    .expand(
                        gen_mask.shape[0], gen_mask.shape[1], hidden_states.shape[-1]
                    )
                    .to(hidden_states.device)
                    .bool()
                )
                hidden_states_gen = torch.masked_select(hidden_states, gen_mask_exp).view(
                    hidden_states.shape[0], -1, hidden_states.shape[-1]
                )
                scale_start_idxes = [0] + self.scale_indices[:-1]
                scale_end_idxes = self.scale_indices
                assert scale_end_idxes[-1] == hidden_states_gen.shape[1]
                scale, scale_start_idx, scale_end_idx = [
                    i
                    for i in zip(
                        self.img_gen_scales, scale_start_idxes, scale_end_idxes
                    )
                ][-1]
                scale_hidden = hidden_states_gen[
                    :, scale_start_idx:scale_end_idx, :
                ]
                scale_embeds = self.proj_in(scale_hidden)
                seq_shape = scale_embeds.shape
                scale_embeds = self.connector(
                    inputs_embeds=scale_embeds,
                    attention_mask=torch.ones(
                        seq_shape[0], 1, seq_shape[1], seq_shape[1],
                        device=scale_embeds.device,
                    ),
                    output_hidden_states=True,
                ).hidden_states[-1]
                scale_embeds = self.proj_out(scale_embeds)
                if self.connector_norm:
                    scale_embeds = F.normalize(scale_embeds, dim=-1)

        return scale_embeds, directvlm_hidden_states

    # ------------------------------------------------------------------ #
    # generation (same contract as BailingMM2NativeForConditionalGeneration)
    # ------------------------------------------------------------------ #
    def _sampling_autocast_dtype(self):
        """bf16 autocast for the diffusion sampling loop.

        tinyv3 transformer checkpoints store F32 weights; diffusers keeps the
        precision-sensitive ``t_embedder`` / ``cap_embedder`` in fp32 when
        loading with ``torch_dtype=bfloat16`` (``_skip_layerwise_casting_
        patterns``). The reference training/sampling stack runs the loop
        under bf16 autocast; this reproduces those numerics. With an all-fp32
        load no autocast is needed and none is applied.
        """
        if self.diffusion_loss is None:
            return None
        param = next(self.diffusion_loss.train_model.parameters())
        return torch.bfloat16 if param.dtype == torch.bfloat16 else None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_gen: Optional[bool] = False,
        image_gen_pixel_values_reference: Optional[torch.FloatTensor] = None,
        image_gen_steps: Optional[int] = None,
        image_gen_seed: Optional[int] = None,
        image_gen_cfg: Optional[float] = None,
        image_gen_image_cfg: Optional[float] = 1.0,
        image_gen_cfg_mode: Optional[int] = 1,
        image_gen_height: Optional[Union[int, List[int]]] = None,
        image_gen_width: Optional[Union[int, List[int]]] = None,
        image_gen_highres=1024,
        image_gen_llm_hidden_states=None,
        image_gen_condition_embeds=None,
        image_gen_negative_condition_embeds=None,
        image_gen_condition_embeds_2=None,
        image_gen_negative_condition_embeds_2=None,
        image_gen_only_extract_hidden_states: bool = False,
        image_gen_task=None,
        image_gen_return_batch: bool = False,
        num_frames_per_prompt=1,
        image_gen_latents=None,
        image_gen_step_callback=None,
        **generate_kwargs,
    ):
        if image_gen_image_cfg not in (None, 1.0):
            raise ValueError(
                "image_gen_image_cfg is not supported by this inference path; "
                "guidance is controlled by image_gen_cfg only"
            )
        if not image_gen:
            raise ValueError(
                "BailingMoeV3VLNativeForConditionalGeneration.generate requires "
                "image_gen=True (text-only VL generation is not part of the "
                "Ming Image public inference surface)"
            )
        if generate_kwargs:
            raise ValueError(
                "unsupported generate kwargs for tinyv3: "
                + ", ".join(sorted(generate_kwargs))
            )
        if not hasattr(self, "inference_profile") or self.inference_profile is None:
            raise RuntimeError(
                "image modules were loaded without a checkpoint inference profile"
            )
        self.inference_profile.validate_task(
            image_gen_task,
            has_reference_image=image_gen_pixel_values_reference is not None,
            num_layers=num_frames_per_prompt,
        )
        sampling = self.inference_profile.resolve_sampling_parameters(
            steps=image_gen_steps,
            cfg=image_gen_cfg,
            family=self.mllm_family,
        )
        image_gen_steps = sampling.steps
        image_gen_cfg = sampling.cfg
        if image_gen_pixel_values_reference is not None:
            input_channels = image_gen_pixel_values_reference.shape[1]
            expected_channels = self.inference_profile.vae_input_channels
            if input_channels % expected_channels != 0:
                raise ValueError(
                    "reference image channels do not match the checkpoint "
                    f"VAE contract: input={input_channels}, expected a "
                    f"multiple of {expected_channels}"
                )

        condition_embeds, negative_condition_embeds = None, None
        condition_embeds_2, negative_condition_embeds_2 = None, None
        if (image_gen_condition_embeds is not None) or (
            image_gen_condition_embeds_2 is not None
        ):
            if image_gen_condition_embeds is not None:
                condition_embeds = image_gen_condition_embeds
                negative_condition_embeds = (
                    condition_embeds * 0.0
                    if image_gen_negative_condition_embeds is None
                    else image_gen_negative_condition_embeds
                )
            if image_gen_condition_embeds_2 is not None:
                condition_embeds_2 = image_gen_condition_embeds_2
                negative_condition_embeds_2 = (
                    condition_embeds_2 * 0.0
                    if image_gen_negative_condition_embeds_2 is None
                    else image_gen_negative_condition_embeds_2
                )
        else:
            image_embeds = None
            if image_gen_llm_hidden_states is None and pixel_values is not None:
                image_embeds = self.extract_image_feature(
                    pixel_values, grid_thw=image_grid_thw
                )
            assert self.loaded_image_gen_modules is True, (
                "please add `load_image_gen=True` in from_pretrained() method"
            )
            assert position_ids is None
            condition_embeds, condition_embeds_2 = self.get_condition_embeds_for_image_gen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embeds=image_embeds,
                position_ids=position_ids,
                use_cache=False,
                image_grid_thw=image_grid_thw,
                llm_hidden_states=image_gen_llm_hidden_states,
            )
            if condition_embeds is not None:
                negative_condition_embeds = condition_embeds * 0.0
            if condition_embeds_2 is not None:
                negative_condition_embeds_2 = condition_embeds_2 * 0.0
            if image_gen_only_extract_hidden_states:
                return (
                    condition_embeds,
                    negative_condition_embeds,
                    condition_embeds_2,
                    negative_condition_embeds_2,
                )

        assert (condition_embeds is not None) or (condition_embeds_2 is not None)
        if (condition_embeds is not None) and (condition_embeds_2 is not None):
            assert condition_embeds.shape[0] == condition_embeds_2.shape[0]
        bsz = (
            condition_embeds.shape[0]
            if condition_embeds is not None
            else condition_embeds_2.shape[0]
        )

        if image_gen_height is None or image_gen_width is None:
            image_gen_height = [int(image_gen_highres)] * bsz
            image_gen_width = [int(image_gen_highres)] * bsz
        elif isinstance(image_gen_height, torch.Tensor) or isinstance(
            image_gen_width, torch.Tensor
        ):
            image_gen_height = image_gen_height.cpu().tolist()
            image_gen_width = image_gen_width.cpu().tolist()
        elif isinstance(image_gen_height, int) or isinstance(image_gen_width, int):
            image_gen_height = [int(image_gen_height)] * bsz
            image_gen_width = [int(image_gen_width)] * bsz
        assert len(image_gen_height) == bsz
        assert len(image_gen_width) == bsz
        assert all(h == image_gen_height[0] for h in image_gen_height)
        assert all(w == image_gen_width[0] for w in image_gen_width)
        image_gen_height = image_gen_height[0]
        image_gen_width = image_gen_width[0]

        # Reference image shape drives latent dimensions; CLI resolution
        # drives the final output size (pipeline resizes if needed).

        if image_gen_seed is None or image_gen_seed < 0:
            from datetime import datetime

            image_gen_seed = datetime.now().microsecond % 1000

        sample_kwargs = {
            "steps": image_gen_steps,
            "seed": image_gen_seed,
            "cfg": image_gen_cfg,
            "height": image_gen_height,
            "width": image_gen_width,
            "cfg_mode": image_gen_cfg_mode,
            "ref_x": image_gen_pixel_values_reference,
            "encoder_hidden_states": condition_embeds,
            "directvlm_hidden_states": condition_embeds_2,
            "num_frames_per_prompt": num_frames_per_prompt,
            # alignment-tool passthrough: replaying a captured initial noise
            # and/or collecting per-step latents. Both default to None and
            # leave normal sampling untouched.
            "latents": image_gen_latents,
            "callback_on_step_end": image_gen_step_callback,
        }

        autocast_dtype = self._sampling_autocast_dtype()
        # Seed a generator on the sampling device so initial-noise generation
        # follows the same device-specific RNG path on repeated runs.
        generator = torch.Generator(device=self.diffusion_loss.device).manual_seed(
            image_gen_seed
        )
        sample_kwargs["generator"] = generator
        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                image = self.diffusion_loss.sample(**sample_kwargs)
        else:
            image = self.diffusion_loss.sample(**sample_kwargs)

        if image_gen_task != "layer-decompose" and not image_gen_return_batch and len(image) == 1:
            image = image[0]
        return image



__all__ = [
    "BailingMoeV3VLNativeForConditionalGeneration",
    "MLLM_FAMILY",
]
