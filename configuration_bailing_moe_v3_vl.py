# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BailingMoE V3 VL model configuration"""

import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class BailingMoeV3Config(PretrainedConfig):
    model_type = "bailing_moe_v3"

    def __init__(
        self,
        vocab_size=157184,
        hidden_size=2048,
        intermediate_size=5120,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        use_qkv_bias=False,  # bailing only
        use_bias=False,  # bailing only
        rms_norm_eps=1e-06,
        tie_word_embeddings=False,  # PretrainedConfig key, here change default value.
        embedding_dropout=0.0,
        attention_dropout=0.0,
        output_dropout=0.0,
        initializer_range=0.02,
        max_position_embeddings=32768,
        rope_theta=600000.0,
        use_cache=True,
        max_window_layers=20,
        rope_scaling=None,
        pad_token_id=156892,
        eos_token_id=156892,
        num_experts=256,
        num_shared_experts=1,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        moe_intermediate_size=512,
        moe_shared_expert_intermediate_size=512,
        expert_swiglu_limit_list=None,
        share_expert_swiglu_limit_list=None,
        first_k_dense_replace=1,
        head_dim=128,
        output_router_logits=False,
        use_qk_norm=True,
        num_nextn_predict_layers=0,
        mtp_loss_scaling_factor=0,
        moe_router_enable_expert_bias=True,
        routed_scaling_factor=1.0,
        layer_group_size=5,
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        rope_interleave=True,
        score_function="sigmoid",
        scoring_func="sigmoid",
        seq_aux=True,
        topk_method="noaux_tc",
        router_dtype="fp32",
        gated_attention_proj_granularity_type=None,
        no_kda_lora=False,
        kda_safe_gate=False,
        kda_lower_bound=None,
        short_conv_kernel_size=4,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.routed_scaling_factor = routed_scaling_factor

        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.expert_swiglu_limit_list = expert_swiglu_limit_list
        self.share_expert_swiglu_limit_list = share_expert_swiglu_limit_list
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits

        # Linear configs
        self.layer_group_size = layer_group_size
        # mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        self.score_function = score_function
        self.scoring_func = scoring_func
        self.seq_aux = seq_aux
        self.topk_method = topk_method
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.rope_interleave = rope_interleave
        self.router_dtype = router_dtype
        self.gated_attention_proj_granularity_type = gated_attention_proj_granularity_type
        self.no_kda_lora = no_kda_lora
        self.kda_safe_gate = kda_safe_gate
        self.kda_lower_bound = kda_lower_bound if kda_safe_gate else None
        self.short_conv_kernel_size = short_conv_kernel_size

        if isinstance(layer_group_size, int):
            if layer_group_size <= 0:
                raise ValueError("layer_group_size must be positive")
            if num_hidden_layers % layer_group_size != 0:
                raise ValueError(
                    f"num_hidden_layers={num_hidden_layers} must be divisible by "
                    f"layer_group_size={layer_group_size}"
                )
        elif isinstance(layer_group_size, list):
            if len(layer_group_size) != num_hidden_layers:
                raise ValueError(
                    "When layer_group_size is a list, its length must equal num_hidden_layers"
                )
        else:
            raise TypeError("layer_group_size must be an int or a list")

        super().__init__(
            pad_token_id=pad_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )

    @property
    def layers_block_type(self):
        """Return the SGLang-compatible attention type for every decoder layer."""
        if isinstance(self.layer_group_size, int):
            return [
                "attention" if (layer_idx + 1) % self.layer_group_size == 0 else "linear_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        return [
            "linear_attention" if layer_type == 1 else "attention"
            for layer_type in self.layer_group_size
        ]

    @property
    def linear_layer_ids(self):
        return [i for i, layer_type in enumerate(self.layers_block_type) if layer_type == "linear_attention"]

    @property
    def full_attention_layer_ids(self):
        return [i for i, layer_type in enumerate(self.layers_block_type) if layer_type == "attention"]


class Qwen3VLMoeVisionConfig(PretrainedConfig):
    model_type = "qwen3_moe_vit"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
        initializer_range=0.02,
        disable_merger_proj=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.disable_merger_proj = disable_merger_proj

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'vision_config' in config_dict:
            config_dict = config_dict['vision_config']

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class BailingMoeV3VLConfig(PretrainedConfig):
    model_type = "bailing_moe_v3_vl"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        mrope_section=None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config = Qwen3VLMoeVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen3VLMoeVisionConfig()

        if isinstance(text_config, dict):
            text_config = BailingMoeV3Config(**text_config)
        elif text_config is None:
            text_config = BailingMoeV3Config()

        self.vision_config = vision_config
        self.text_config = text_config
        # SGLang names the language config `llm_config`; keep `text_config` for
        # HuggingFace compatibility and expose the SGLang alias as well.
        self.llm_config = text_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # M-RoPE section: split qk_rope_head_dim // 2 frequencies into [T, H, W].
        # Prefer an explicit top-level value; otherwise follow SGLang and read it
        # from the language rope config before falling back to the historical default.
        rope_parameters = getattr(text_config, "rope_parameters", None)
        if rope_parameters is None:
            rope_parameters = dict(getattr(text_config, "rope_scaling", None) or {})
        else:
            rope_parameters = dict(rope_parameters)

        if mrope_section is None:
            mrope_section = rope_parameters.get("mrope_section", [12, 10, 10])
        mrope_section = list(mrope_section)
        expected_mrope_dims = text_config.qk_rope_head_dim // 2
        if len(mrope_section) != 3 or sum(mrope_section) != expected_mrope_dims:
            raise ValueError(
                f"mrope_section must contain three entries summing to {expected_mrope_dims}, "
                f"got {mrope_section}"
            )
        self.mrope_section = mrope_section
        # The HF text rotary module reads mrope_section from the text config,
        # while SGLang reads it from llm_config.rope_parameters.  Set both.
        text_config.mrope_section = mrope_section

        # Keep both naming conventions in sync.  SGLang uses rope_parameters; the
        # HF modeling in this directory still understands rope_scaling/default RoPE.
        rope_parameters.update(
            rope_type=rope_parameters.get("rope_type", rope_parameters.get("type", "default")),
            mrope_section=mrope_section,
            video_rope=rope_parameters.get("video_rope", True),
        )
        text_config.rope_parameters = dict(rope_parameters)
        text_config.rope_scaling = dict(rope_parameters)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
