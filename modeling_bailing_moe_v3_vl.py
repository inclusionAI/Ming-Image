# coding=utf-8
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch BailingMoE V3 VL model."""

import math
import warnings
from typing import List, Optional, Tuple, Union, Callable
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

def _compute_bailing_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if layer_type is not None and layer_type in rope_parameters:
        rope_parameters = rope_parameters[layer_type]
    base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0)))
    # Prefer the explicit config attribute because BailingMoeV3VLRotaryEmbedding
    # overrides it to 1.0 for qk_rope_head_dim.  The checkpoint text config may
    # still carry partial_rotary_factor=0.5 in rope_parameters/rope_scaling for
    # other HF conventions; using that here would produce 32-dim cos/sin for a
    # 64-dim q_rot/k_rot tensor.
    partial_rotary_factor = getattr(config, "partial_rotary_factor", rope_parameters.get("partial_rotary_factor", 1.0))
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0

ROPE_INIT_FUNCTIONS.setdefault("default", _compute_bailing_default_rope_parameters)
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
try:
    from transformers.utils.import_utils import is_torch_fx_available
except ImportError:
    def is_torch_fx_available():
        return False
from configuration_bailing_moe_v3_vl import BailingMoeV3Config, BailingMoeV3VLConfig, Qwen3VLMoeVisionConfig
from transformers.generation.utils import GenerationMixin
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import DynamicLayer
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from einops import rearrange, repeat

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except ImportError:
    _flash_attn_varlen_func = None


def _bailing_flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
    seqused_q=None, seqused_k=None, page_table=None, softmax_scale=None,
    causal=False, qv=None, q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1), attention_chunk=0, softcap=0.0, num_splits=1,
    pack_gqa=None, sm_margin=0, return_softmax_lse=False, sinks=None, out=None,
):
    """Standalone flash-attn backend for the vision encoder.

    This intentionally has the former FA3 wrapper signature so the vision
    attention call site stays unchanged, but it has no kernels/sgl_kernel or
    SGLang runtime dependency.
    """
    if _flash_attn_varlen_func is None:
        raise ImportError("flash-attn is required for CUDA vision attention")
    result = _flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        return_attn_probs=return_softmax_lse,
        block_table=page_table,
    )
    if isinstance(result, tuple):
        result = result[0]
    if out is not None:
        out.copy_(result)
        return out
    return result

def roll_tensor(tensor, shifts=-1, dims=-1, fill_value=0):
    """Roll the tensor input along the given dimension(s).
    Inserted elements are set to be 0.0.
    """
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.select(dims, shifts).fill_(fill_value)
    return rolled_tensor, rolled_tensor.sum()


try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.simple_gla.chunk import chunk_simple_gla
    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
    from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
    from fla.utils import tensor_cache
except ImportError:
    # fla-core is an optional dependency required only by the tinyv3
    # (bailing_moe_v3_vl) MLLM; miniv2 checkpoints never import this module.
    FusedRMSNormGated = ShortConvolution = None
    chunk_kda = fused_recurrent_kda = None
    chunk_simple_gla = fused_recurrent_simple_gla = None
    prepare_cu_seqlens_from_mask = prepare_lens_from_mask = None

    def tensor_cache(*_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator


def require_fla_kernels():
    if chunk_kda is None:
        raise ImportError(
            "the tinyv3 (bailing_moe_v3_vl) MLLM requires the fla-core package "
            "(KDA linear-attention kernels, e.g. pip install fla-core); "
            "miniv2 checkpoints do not need it"
        )

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BailingMoeV3VLConfig"


def index_first_axis(x, indices):
    other_shape = x.shape[1:]
    second_dim = other_shape.numel()
    return torch.gather(
        rearrange(x, "b ... -> b (...)"),
        0,
        repeat(indices, "z -> z d", d=second_dim),
    ).reshape(-1, *other_shape)


def index_put_first_axis(x, indices, first_axis_dim):
    y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
    y[indices] = x
    # y.scatter_(0, repeat(indices, 'z -> z d', d=x.shape[1]), x)
    return y


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)


@tensor_cache
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.BailingMoeV3.modeling_BailingMoeV3._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.BailingMoeV3.modeling_BailingMoeV3._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.BailingMoeV3.modeling_BailingMoeV3.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class BailingMoeV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BailingMoeV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class BailingMoeV3GroupRMSNorm(nn.Module):
    def __init__(self, hidden_size, group_norm_size, eps=1e-6):
        """
        BailingMoeV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_norm_size = group_norm_size
        assert hidden_size % group_norm_size == 0, "hidden_size must be divisible by group_norm_size"
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.size()
        group_input_shape = input_shape[:-1] + (self.group_norm_size, input_shape[-1] // self.group_norm_size)
        hidden_states = hidden_states.view(group_input_shape)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype).view(input_shape)


ALL_LAYERNORM_LAYERS.append(BailingMoeV3RMSNorm)


class BailingMoeV3VLRotaryEmbedding(nn.Module):
    """M-RoPE aware rotary embedding for VL models.

    Handles 3D position IDs (temporal, height, width) and interleaves
    the frequency components according to mrope_section.
    """
    inv_freq: torch.Tensor

    def __init__(self, config: BailingMoeV3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = deepcopy(config)
        self.config.head_dim = config.qk_rope_head_dim
        self.config.partial_rotary_factor = 1.0
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        # These non-persistent buffers are constructed while Transformers may be
        # under an empty/meta-weight initialization context. Such buffers can be
        # materialized with uninitialized storage by low_cpu_mem_usage loading.
        # Recompute from the immutable RoPE config after model materialization
        # (see _ensure_inv_freq_initialized) rather than trusting that storage.
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self._inv_freq_initialized = False

        # M-RoPE section: split frequencies into [T, H, W] chunks
        self.mrope_section = getattr(config, "mrope_section", [12, 10, 10])

    def _ensure_inv_freq_initialized(self, device):
        if self._inv_freq_initialized and self.inv_freq.device == device:
            return
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.inv_freq = inv_freq
        self.original_inv_freq = inv_freq.clone()
        self._inv_freq_initialized = True

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        self._ensure_inv_freq_initialized(x.device)
        # Support 3D position IDs for M-RoPE (T, H, W)
        # position_ids: (3, batch, seq) or (batch, seq)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # inv_freq shape: (dim/2,) -> expand to (3, batch, dim/2, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # (3, batch, 1, seq)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            # freqs shape: (3, batch, seq, dim/2)
            emb = torch.cat((freqs, freqs), dim=-1)
            # emb shape: (3, batch, seq, dim)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            cos, sin = self.apply_video_rope(cos, sin, self.mrope_section)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_video_rope(self, cos, sin, mrope_section):
        """Apply video_rope layout to 3D rotary embeddings.

        Args:
            cos: (3, bs, seq_len, head_dim)  -- [T, H, W]
            sin: (3, bs, seq_len, head_dim)
            mrope_section: [T_section, H_section, W_section] e.g. [12, 10, 10]
        Returns:
            cos, sin: each (bs, seq_len, head_dim)
        """
        section = list(mrope_section)
        section = [section[0], section[1] + section[2]]  # [T, H+W]
        section = section * 2                             # [T, H+W, T, H+W]
        section = section[::-1]                           # [H+W, T, H+W, T]

        index = 0
        result_cos = []
        result_sin = []
        for i, sec in enumerate(section):
            if i % 2 == 0:  # H+W: interleave H and W dim-by-dim
                for j in range(sec):
                    row = 1 if j % 2 == 0 else 2
                    result_cos.append(cos[row, ..., index: index + 1])
                    result_sin.append(sin[row, ..., index: index + 1])
                    index += 1
            else:  # T: block copy
                result_cos.append(cos[0, ..., index: index + sec])
                result_sin.append(sin[0, ..., index: index + sec])
                index += sec

        return torch.cat(result_cos, dim=-1), torch.cat(result_sin, dim=-1)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class BailingMoeV3MLP(nn.Module):
    def __init__(
        self,
        config: BailingMoeV3Config,
        intermediate_size: int,
        swiglu_limit: Optional[float] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(gate * up)


class BailingMoeV3Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor

        self.expert_bias = nn.Parameter(
            torch.zeros((self.num_experts,), dtype=torch.float32),
            requires_grad=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = torch.sigmoid(logits.float()).type_as(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits


class BailingMoeV3SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    @staticmethod
    def _get_swiglu_limit(limit_list, layer_idx):
        try:
            if limit_list is None or len(limit_list) <= layer_idx or limit_list[layer_idx] == 0:
                return None
            return limit_list[layer_idx]
        except Exception:
            return None

    def __init__(self, config: BailingMoeV3Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_swiglu_limit = self._get_swiglu_limit(
            getattr(config, "expert_swiglu_limit_list", None), layer_idx
        )
        self.share_expert_swiglu_limit = self._get_swiglu_limit(
            getattr(config, "share_expert_swiglu_limit_list", None), layer_idx
        )
        self._setup_experts()
        self.gate = BailingMoeV3Gate(config)
        if config.num_shared_experts is not None:
            self.shared_experts = BailingMoeV3MLP(
                config=config,
                intermediate_size=config.moe_shared_expert_intermediate_size * config.num_shared_experts,
                swiglu_limit=self.share_expert_swiglu_limit,
            )

    def _setup_experts(self):
        self.experts = nn.ModuleList(
            [
                BailingMoeV3MLP(
                    config=self.config,
                    intermediate_size=self.config.moe_intermediate_size,
                    swiglu_limit=self.expert_swiglu_limit,
                )
                for _ in range(self.config.num_experts)
            ]
        )

    def forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out.to(x.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# =============================================================================
# Vision Encoder Components (from Qwen3 ViT, without DeepStack)
# =============================================================================


class Qwen3VLMoeVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))




def _conv3d_patch_embed_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], kernel_size):
    # Match SGLang Conv3dLayer's unfold+F.linear patch embedding path.
    k1, k2, k3 = tuple(kernel_size)
    x = x.unfold(2, k1, k1).unfold(3, k2, k2).unfold(4, k3, k3)
    n, dp, hp, wp = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
    x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(n, dp, hp, wp, -1)
    x = F.linear(x, weight.reshape(weight.shape[0], -1), bias)
    return x.permute(0, 4, 1, 2, 3)

class Qwen3VLMoeVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = hidden_states.to(dtype=target_dtype)
        # SGLang uses Conv3dLayer, which takes an unfold+F.linear fast path for
        # patch embedding.  This differs numerically from cuDNN conv3d in bf16;
        # use the same computation order for strict alignment.
        hidden_states = _conv3d_patch_embed_linear(
            hidden_states, self.proj.weight, self.proj.bias, self.proj.kernel_size
        ).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLMoeVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLMoeVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLMoeVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.disable_merger_proj = getattr(config, "disable_merger_proj", False)
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        if not self.disable_merger_proj:
            self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
            self.act_fn = nn.GELU()
            self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        if not self.disable_merger_proj:
            x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


# Keep this helper eager so inference does not require a compatible
# inductor/triton compilation stack on every target device.
def _bailing_vision_apply_rotary_pos_emb(q, k, cos, sin):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(1).float()
    sin = sin.unsqueeze(1).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)

def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.size(-1) * 2 == q.shape[-1]:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    return _bailing_vision_apply_rotary_pos_emb(q, k, cos, sin)


class Qwen3VLMoeVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLMoeVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** -0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Match SGLang VisionAttention(use_qkv_parallel=True, backend=triton_attn)
        # numerics/layout as closely as possible. Input is [S, 1, D] from the
        # HF vision block; SGLang rearranges it to [1, S, D] before qkv_proj.
        input_was_2d = hidden_states.dim() == 2
        if input_was_2d:
            hidden_states = hidden_states.unsqueeze(1)
        seq_length = hidden_states.shape[0]
        x = hidden_states.transpose(0, 1).contiguous()  # [1, S, D]
        qkv = self.qkv(x)
        query_states, key_states, value_states = qkv.split([self.dim, self.dim, self.dim], dim=-1)
        query_states = query_states.reshape(seq_length, self.num_heads, self.head_dim).contiguous()
        key_states = key_states.reshape(seq_length, self.num_heads, self.head_dim).contiguous()
        value_states = value_states.reshape(seq_length, self.num_heads, self.head_dim).contiguous()

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        if query_states.is_cuda:
            cu = cu_seqlens.to(device=query_states.device, dtype=torch.int32)
            seq_lens = cu[1:] - cu[:-1]
            max_seqlen = int(seq_lens.max().item())
            attn_output = _bailing_flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=None,
                causal=False,
            )
        else:
            # Portable fallback: per-sequence SDPA/eager attention.
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            q_splits = torch.split(query_states, lengths, dim=0)
            k_splits = torch.split(key_states, lengths, dim=0)
            v_splits = torch.split(value_states, lengths, dim=0)
            outs = []
            scale = self.scaling
            for q, k, v in zip(q_splits, k_splits, v_splits):
                q = q.transpose(0, 1).unsqueeze(0)
                k = k.transpose(0, 1).unsqueeze(0)
                v = v.transpose(0, 1).unsqueeze(0)
                o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=scale)
                outs.append(o.squeeze(0).transpose(0, 1))
            attn_output = torch.cat(outs, dim=0)

        attn_output = attn_output.reshape(1, seq_length, -1).contiguous()
        attn_output = self.proj(attn_output).reshape(seq_length, 1, -1).contiguous()
        if attn_output.shape[0] != seq_length or attn_output.shape[1] != 1:
            raise RuntimeError(f"bad vision attn output shape {tuple(attn_output.shape)} for seq_length={seq_length}")
        return attn_output.squeeze(1) if input_was_2d else attn_output


class Qwen3VLMoeVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLMoeVisionAttention(config=config)
        self.mlp = Qwen3VLMoeVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Match SGLang Qwen3_VisionBlock layout.  SGLang keeps a singleton
        # batch dim through LayerNorm/residuals: x [S,1,D] -> norm -> attn ->
        # residual -> norm2 -> mlp -> residual.  Running LayerNorm on [S,D] is
        # mathematically equivalent but can differ in bf16 kernel/layout.
        input_was_2d = hidden_states.dim() == 2
        if input_was_2d:
            hidden_states = hidden_states.unsqueeze(1)
        norm1 = self.norm1(hidden_states)
        attn = self.attn(
            norm1,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + attn
        norm2 = self.norm2(hidden_states)
        mlp = self.mlp(norm2)
        hidden_states = hidden_states + mlp
        return hidden_states.squeeze(1) if input_was_2d else hidden_states


class BailingMoeV3VLVisionModel(PreTrainedModel):
    """Vision encoder for BailingMoeV3 VL, based on Qwen3 ViT without DeepStack."""
    config_class = Qwen3VLMoeVisionConfig
    _no_split_modules = ["Qwen3VLMoeVisionBlock"]

    def __init__(self, config: Qwen3VLMoeVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLMoeVisionPatchEmbed(config=config)

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)

        head_dim = config.hidden_size // config.num_heads
        # Keep the module for compatibility, but rot_pos_emb below builds
        # the SGLang-equivalent cos/sin cache explicitly to avoid relying on
        # non-persistent buffers under low_cpu_mem_usage loading.
        self.rotary_pos_emb = Qwen3VLMoeVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLMoeVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLMoeVisionPatchMerger(config=config, use_postshuffle_norm=False)

        # This checkpoint has disable_merger_proj=True and no deepstack merger
        # weights.  SGLang may construct deepstack modules generically, but its
        # Bailing wrapper slices vision_embeds[..., :vision_out_dim] before the
        # language projector.  For HF strict inference, avoid randomly
        # initialized deepstack modules and keep the effective main stream.
        self.disable_merger_proj = getattr(config, "disable_merger_proj", False)
        self.deepstack_visual_indexes = list(getattr(config, "deepstack_visual_indexes", []) or [])
        self.use_deepstack = (not self.disable_merger_proj) and len(self.deepstack_visual_indexes) > 0
        if self.use_deepstack:
            self.deepstack_merger_list = nn.ModuleList(
                [
                    Qwen3VLMoeVisionPatchMerger(config=config, use_postshuffle_norm=True)
                    for _ in range(len(self.deepstack_visual_indexes))
                ]
            )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match SGLang Qwen3VLMoeVisionModel.rot_pos_emb exactly."""
        grid_list = grid_thw.tolist() if torch.is_tensor(grid_thw) else grid_thw
        pos_ids = []
        import numpy as _np
        for t, h, w in grid_list:
            h, w, t = int(h), int(w), int(t)
            m = int(self.spatial_merge_size)
            hpos_ids = _np.broadcast_to(_np.arange(h).reshape(h, 1), (h, w))
            h_div, w_div = h // m, w // m
            hpos_ids = hpos_ids.reshape(h_div, m, w_div, m).transpose(0, 2, 1, 3).flatten()
            wpos_ids = _np.broadcast_to(_np.arange(w).reshape(1, w), (h, w))
            wpos_ids = wpos_ids.reshape(h_div, m, w_div, m).transpose(0, 2, 1, 3).flatten()
            base = torch.from_numpy(_np.stack([hpos_ids, wpos_ids], axis=-1))
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0).to(self.pos_embed.weight.device, non_blocking=True)
        max_grid_size = max(max(int(h), int(w)) for _, h, w in grid_list)
        # Build the same cos/sin cache as SGLang RotaryEmbedding.get_cos_sin.
        # Do this explicitly instead of relying on the HF non-persistent
        # inv_freq buffer, which can be left uninitialized by low_cpu_mem_usage
        # loading for this remote-code module.
        head_dim = self.config.hidden_size // self.config.num_heads
        rotary_dim = head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=pos_ids.device) / rotary_dim))
        t = torch.arange(max_grid_size, dtype=torch.float32, device=pos_ids.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[pos_ids].flatten(1), sin[pos_ids].flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw):
        """Match SGLang fast_pos_embed_interpolate_from_list exactly."""
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim
        outputs = []
        for t, h, w in grid_thw.tolist():
            h_idxs = torch.linspace(0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.pos_embed.weight.device)
            w_idxs = torch.linspace(0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.pos_embed.weight.device)

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side
            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.pos_embed.weight.dtype)

            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)
            combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)
        return torch.cat(outputs, dim=0)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states: raw pixel values, shape (total_patches, C*temporal_patch*patch*patch)
            grid_thw: (num_images_or_videos, 3) - temporal, height, width of each image/video

        Returns:
            merged hidden_states after vision encoding
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        position_embeddings = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        num_deepstack_captured = 0
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if self.use_deepstack and layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[num_deepstack_captured](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
                num_deepstack_captured += 1

        hidden_states = self.merger(hidden_states)
        if deepstack_feature_lists:
            hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)

        return hidden_states


# =============================================================================
# Text Model Components
# =============================================================================


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int, head_first: bool = True) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). If head_first is True, the hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    if head_first:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    else:
        batch, slen, num_key_value_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
        return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def repeat_kv2(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv2(key, module.num_key_value_groups)
    value_states = repeat_kv2(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""
    TODO let's just use the original freqcis computation to not have the view
    transpose + reshape! This is not optimized!
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # q/k projection weights use pair-interleaved rotary channels:
    # [x0, y0, x1, y1, ...]. Keep this layout at the attention boundary.
    # The rotary module returns duplicated-half frequencies
    # [c0, c1, ..., cN, c0, c1, ..., cN], so one value per pair is
    # taken from the first half. This matches SGLang's is_neox_style=False
    # even/odd rotation and, importantly, makes position 0 an identity.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_pairs(x):
        half_dim = x.shape[-1] // 2
        x_pairs = x.reshape(*x.shape[:-1], half_dim, 2)
        x_even, x_odd = x_pairs.unbind(dim=-1)
        pair_cos = cos[..., :half_dim]
        pair_sin = sin[..., :half_dim]
        return torch.stack(
            (x_even * pair_cos - x_odd * pair_sin,
             x_odd * pair_cos + x_even * pair_sin),
            dim=-1,
        ).flatten(-2)

    return rotate_pairs(q), rotate_pairs(k)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class BailingMoeV3MultiLatentAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.gated_attention_proj_granularity_type = config.gated_attention_proj_granularity_type

        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.use_qkv_bias)
            self.q_a_layernorm = BailingMoeV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.use_qkv_bias,
        )
        self.kv_a_layernorm = BailingMoeV3RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        if self.gated_attention_proj_granularity_type is None:
            self.g_proj = None
        elif self.gated_attention_proj_granularity_type == "head_wise":
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        elif self.gated_attention_proj_granularity_type == "element_wise":
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads * self.v_head_dim, bias=False)

        self.dense = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.use_qkv_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling.get("factor", 1.0)
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings  # tptest
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        if self.g_proj is not None:
            gate = self.g_proj(hidden_states)
            gate = F.sigmoid(gate.float()).type_as(hidden_states)
            if self.gated_attention_proj_granularity_type == "head_wise":
                attn_output = attn_output * gate[:, :, :, None]
            else:
                attn_output = attn_output * gate.view(batch_size, seq_length, self.num_heads, self.v_head_dim)

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.dense(attn_output)
        return attn_output, attn_weights, past_key_values


class BailingMoeV3KimiDeltaAttention(nn.Module):
    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        require_fla_kernels()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.short_conv_kernel_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads
        self.no_kda_lora = config.no_kda_lora
        self.safe_gate = config.kda_safe_gate
        self.lower_bound = config.kda_lower_bound

        self.layer_idx = layer_idx

        assert self.mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu',
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu',
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation='silu',
        )

        self.A_log = torch.nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))

        if self.no_kda_lora:
            self.f_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        if self.no_kda_lora:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        attention_mask = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask")

            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here.",
                )
        use_cache = past_key_value is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get('cu_seqlens')
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None

        if past_key_value is not None and isinstance(past_key_value, Cache):
            # ensure the cache list is long enough
            while len(past_key_value.layers) <= self.layer_idx:
                past_key_value.layers.append(DynamicLayer())

            if past_key_value.layers[self.layer_idx].keys is not None:
                recurrent_state = past_key_value.layers[self.layer_idx].keys
                # ensure recurrent_state is on the same device as hidden_states
                if recurrent_state.device != hidden_states.device:
                    recurrent_state = recurrent_state.to(hidden_states.device).contiguous()

            if past_key_value.layers[self.layer_idx].values is not None:
                conv_state_q, conv_state_k, conv_state_v = past_key_value.layers[self.layer_idx].values

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        if self.no_kda_lora:
            g = self.f_proj(hidden_states)
        else:
            g = self.f_b_proj(self.f_a_proj(hidden_states))

        beta = self.b_proj(hidden_states).float().sigmoid()

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)

        if mode == 'chunk':
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            )

        if use_cache and past_key_value is not None and isinstance(past_key_value, Cache):
            target_device = None
            for cache in past_key_value.layers:
                if cache.keys is not None:
                    target_device = cache.keys.device
                    break
            if target_device is None:
                target_device = recurrent_state.device

            # move to target device
            if recurrent_state.device != target_device:
                recurrent_state = recurrent_state.to(target_device)

            past_key_value.layers[self.layer_idx].keys = recurrent_state
            past_key_value.layers[self.layer_idx].values = (conv_state_q, conv_state_k, conv_state_v)

        if self.no_kda_lora:
            g = self.g_proj(hidden_states)
        else:
            g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_value


class BailingMoeV3MTPLayer(nn.Module):
    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.post_attention_layernorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = BailingMoeV3MultiLatentAttention(config=config, layer_idx=layer_idx)
        self.mlp = BailingMoeV3SparseMoeBlock(config, layer_idx=layer_idx)

        self.hnorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layernorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        input_embeds = self.enorm(input_embeds)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = self.eh_proj(torch.cat([input_embeds, hidden_states], dim=-1))
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)
        hidden_states = self.final_layernorm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class BailingMoeV3DecoderLayer(nn.Module):
    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_layer_type = (
            "attention"
            if (layer_idx + 1) % config.layer_group_size == 0
            or layer_idx >= config.num_hidden_layers // config.layer_group_size * config.layer_group_size
            else "linear_attention"
        )

        if self.attention_layer_type == "attention":
            self.attention = BailingMoeV3MultiLatentAttention(config=config, layer_idx=layer_idx)
        else:
            self.attention = BailingMoeV3KimiDeltaAttention(config=config, layer_idx=layer_idx)

        self.mlp = (
            BailingMoeV3SparseMoeBlock(config, layer_idx=layer_idx)
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else BailingMoeV3MLP(config=config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.attention_layer_type == "attention":
            hidden_states, self_attn_weights, present_key_value = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,  #
                position_embeddings=position_embeddings,  #
                **kwargs,
            )
        else:
            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            device = hidden_states.device

            if attention_mask is None:
                # if attention_mask is None, create a full mask
                attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int32, device=device)
            elif attention_mask.dim() == 4 and attention_mask.shape[1] == 1:
                attention_mask = attention_mask[:, 0, -1, :].to(torch.int32)
                attention_mask = (attention_mask > -1e4).to(torch.int32)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask.to(torch.int32)
            else:
                raise ValueError(f"Unsupported mask dimension: {attention_mask.shape}")

            hidden_states, self_attn_weights, present_key_value = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                position_ids=position_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


BAILINGMOEV3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`BailingMoeV3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


class MoeV3ModelOutputWithPast(MoeModelOutputWithPast):
    """Backbone output extended with hidden states from optional MTP layers."""

    def __init__(self, mtp_hidden_states=None, **kwargs):
        super().__init__(**kwargs)
        self.mtp_hidden_states = mtp_hidden_states


@dataclass
class BailingMoeV3VLModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


@dataclass
class BailingMoeV3VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    "The bare BailingMoeV3 VL Model outputting raw hidden-states without any specific head on top.",
    BAILINGMOEV3_START_DOCSTRING,
)
class BailingMoeV3PreTrainedModel(PreTrainedModel):
    _keep_in_fp32_modules_strict = ["gate", "expert_bias", "A_log", "dt_bias"]  # 4.56.2 校验按 token 匹配, 不支持点号路径; "gate" 精确命中 mlp.gate(路由), 不会误伤 gate_proj
    config_class = BailingMoeV3VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingMoeV3DecoderLayer", "Qwen3VLMoeVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


BAILINGMOEV3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare BailingMoeV3 Model outputting raw hidden-states without any specific head on top.",
    BAILINGMOEV3_START_DOCSTRING,
)
class BailingMoeV3Model(BailingMoeV3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`BailingMoeV3DecoderLayer`]
    Args:
        config: BailingMoeV3Config
    """

    def __init__(self, config: BailingMoeV3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_nextn_predict_layers = config.num_nextn_predict_layers

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layers = []
        for layer_idx in range(config.num_hidden_layers + config.num_nextn_predict_layers):
            layer_cls = BailingMoeV3DecoderLayer if layer_idx < config.num_hidden_layers else BailingMoeV3MTPLayer
            layers.append(layer_cls(config, layer_idx))
        self.layers = nn.ModuleList(layers)

        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BailingMoeV3VLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(BAILINGMOEV3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mtp_input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, MoeV3ModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Handle VL 4D position_ids: (4, batch, seq) → text_position_ids + 3D for M-RoPE
        text_position_ids = None
        if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]  # for mask creation
            position_ids = position_ids[1:]  # (3, batch, seq) for M-RoPE
        elif position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        softmax_attention_layer_id = self.config.layer_group_size - 1
        past_seen_tokens = (
            past_key_values.get_seq_length(layer_idx=softmax_attention_layer_id) if past_key_values is not None else 0
        )

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_seen_tokens,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_seen_tokens
            )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        # Split main decoder layers and MTP layers
        main_layers = self.layers[: -self.num_nextn_predict_layers] if self.num_nextn_predict_layers > 0 else self.layers
        mtp_layers = self.layers[-self.num_nextn_predict_layers :] if self.num_nextn_predict_layers > 0 else None

        for decoder_layer in main_layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    text_position_ids if text_position_ids is not None else position_ids,
                    past_key_values,
                    cache_position,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids if text_position_ids is not None else position_ids,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)
        main_hidden_states = hidden_states

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (main_hidden_states,)

        mtp_hidden_states = None

        # Keep the MTP forward path identical to the standalone LLM: if MTP
        # layers are configured, execute them in both training and eval mode.
        # Their outputs are auxiliary and do not replace main_hidden_states.
        if mtp_layers:
            # The main VL path must use inputs_embeds after replacing image/video
            # placeholders, while MTP still needs token IDs to build shifted
            # next-token embeddings. Keep these two inputs independent.
            if mtp_input_ids is None:
                mtp_input_ids = input_ids
            if mtp_input_ids is None:
                raise ValueError("input_ids or mtp_input_ids are required when MTP layers are enabled")
            for decoder_layer in mtp_layers:
                mtp_input_ids, _ = roll_tensor(mtp_input_ids, shifts=-1, dims=-1)
                mtp_inputs_embeds = self.word_embeddings(mtp_input_ids)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        mtp_inputs_embeds,
                        hidden_states,
                        attention_mask,
                        text_position_ids if text_position_ids is not None else position_ids,
                        past_key_values,
                        output_attentions,
                        output_router_logits,
                        use_cache,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        mtp_inputs_embeds,
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids if text_position_ids is not None else position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                    )
                if mtp_hidden_states is None:
                    mtp_hidden_states = []
                hidden_states = layer_outputs[0]
                mtp_hidden_states.append(hidden_states)

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if output_router_logits and layer_outputs[-1] is not None:
                    all_router_logits += (layer_outputs[-1],)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [main_hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeV3ModelOutputWithPast(
            last_hidden_state=main_hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            mtp_hidden_states=mtp_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


# =============================================================================
# Composite VL Model
# =============================================================================


class BailingMoeV3VLModel(BailingMoeV3Model):
    """Composite vision-language model: visual encoder + text decoder.

    Inherits from BailingMoeV3Model so that LLM sub-modules (word_embeddings,
    layers, norm, rotary_emb) live directly on ``self``, matching the checkpoint
    key namespace (``model.layers.*`` etc.) produced by the conversion script.
    """

    def __init__(self, config: BailingMoeV3VLConfig):
        # Initialise LLM backbone with text_config
        # self.config remains text_config so BailingMoeV3Model.forward works correctly
        super().__init__(config.text_config)
        # Store full VL config for VL-specific access (image_token_id, vision_config, etc.)
        self._vl_config = config
        # VL-specific modules
        self.visual = BailingMoeV3VLVisionModel._from_config(config.vision_config)
        # NOTE: linear_proj is held by ForConditionalGeneration (key = linear_proj.*)
        self.rope_deltas = None

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device=None,
    ):
        """Compute 3D positional indices (T, H, W) for vision tokens from a single image/video."""
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size

        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
            llm_grid_h * llm_grid_t
        )
        position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
            llm_grid_w * llm_grid_t
        )
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
        position_temporal = position_temporal * time_interval
        vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)

        return vision_position_ids

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Bailing v3/Qwen3-VL 3D M-RoPE IDs using SGLang's token-driven layout.

        The computation is intentionally based on start/patch token ids and grid_thw, not on
        externally supplied ``mm_token_type_ids``.  This matches SGLang's
        ``bailing_3drope_get_input_positions_tensor`` for multimodal prefill while keeping the
        old argument for API compatibility.
        """
        spatial_merge_size = self._vl_config.vision_config.spatial_merge_size
        text_config = self._vl_config.text_config
        image_patch_id = getattr(text_config, "image_patch_token", self._vl_config.image_token_id)
        video_patch_id = getattr(text_config, "video_patch_token", self._vl_config.video_token_id)
        image_start_token_id = getattr(text_config, "image_start_token", getattr(self._vl_config, "vision_start_token_id", None))
        video_start_token_id = getattr(text_config, "video_start_token", getattr(self._vl_config, "vision_start_token_id", None))
        use_interleaved_frame_timestamp = getattr(text_config, "use_interleaved_frame_timestamp", False)

        # SGLang treats interleaved-frame videos as per-frame image-start segments.
        if video_grid_thw is not None and use_interleaved_frame_timestamp:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw = video_grid_thw.clone()
            video_grid_thw[:, 0] = 1

        batch_size, seq_len = input_ids.shape
        position_ids = torch.zeros(3, batch_size, seq_len, dtype=input_ids.dtype, device=input_ids.device)
        mrope_position_deltas = []

        for batch_idx, current_input_ids in enumerate(input_ids):
            if attention_mask is not None:
                valid_mask = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[valid_mask]
            else:
                valid_mask = None

            device = current_input_ids.device
            input_tokens = current_input_ids.tolist()
            llm_pos_ids_list = []
            image_index = 0
            video_index = 0

            image_nums = 0
            if image_grid_thw is not None and image_start_token_id is not None:
                image_start_indices = torch.argwhere(current_input_ids == image_start_token_id).flatten()
                if image_start_indices.numel() > 0:
                    image_start_indices = image_start_indices[image_start_indices + 1 < current_input_ids.numel()]
                    vision_tokens = current_input_ids[image_start_indices + 1]
                    image_nums = int((vision_tokens == image_patch_id).sum().item())

            video_nums = 0
            if video_grid_thw is not None:
                start_id = image_start_token_id if use_interleaved_frame_timestamp else video_start_token_id
                if start_id is not None:
                    video_start_indices = torch.argwhere(current_input_ids == start_id).flatten()
                    if video_start_indices.numel() > 0:
                        video_start_indices = video_start_indices[video_start_indices + 1 < current_input_ids.numel()]
                        vision_tokens = current_input_ids[video_start_indices + 1]
                        video_nums = int((vision_tokens == video_patch_id).sum().item())

            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if remain_images > 0 and image_patch_id in input_tokens[st:]:
                    ed_image = input_tokens.index(image_patch_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if remain_videos > 0 and video_patch_id in input_tokens[st:]:
                    ed_video = input_tokens.index(video_patch_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    if image_grid_thw is None or image_index >= image_grid_thw.shape[0]:
                        raise ValueError("Not enough image_grid_thw entries for image patch tokens")
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    if video_grid_thw is None or video_index >= video_grid_thw.shape[0]:
                        raise ValueError("Not enough video_grid_thw entries for video patch tokens")
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t = int(t.item() if isinstance(t, torch.Tensor) else t)
                llm_grid_h = int(h.item() if isinstance(h, torch.Tensor) else h) // spatial_merge_size
                llm_grid_w = int(w.item() if isinstance(w, torch.Tensor) else w) // spatial_merge_size
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1][0].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if text_len > 0:
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=device, dtype=torch.float32).view(1, -1).expand(3, -1) + st_idx
                    )

                # Body-diagonal symmetry, matching SGLang BailingMRotaryEmbedding.
                t_index = (
                    torch.arange(llm_grid_t, device=device, dtype=torch.float32)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                    * 2.0
                    + text_len
                    + st_idx
                )
                h_index = (
                    torch.arange(llm_grid_h, device=device, dtype=torch.float32)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                    - (llm_grid_h - 1) // 2
                    + t_index
                )
                w_index = (
                    torch.arange(llm_grid_w, device=device, dtype=torch.float32)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                    - (llm_grid_w - 1) // 2
                    + t_index
                )
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]))
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1][0].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device, dtype=torch.float32).view(1, -1).expand(3, -1) + st_idx
                )

            if len(llm_pos_ids_list) == 0:
                llm_positions = torch.arange(len(input_tokens), device=device, dtype=torch.float32).view(1, -1).expand(3, -1)
            else:
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            llm_positions = llm_positions.to(dtype=position_ids.dtype, device=position_ids.device)
            if valid_mask is not None:
                position_ids[:, batch_idx, valid_mask] = llm_positions
            else:
                position_ids[:, batch_idx, : llm_positions.shape[1]] = llm_positions
            mrope_position_deltas.append(llm_positions[0].max() + 1 - len(current_input_ids))

        mrope_position_deltas = torch.stack(mrope_position_deltas).to(device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def compute_3d_position_ids(
        self,
        input_ids,
        inputs_embeds,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        past_key_values=None,
        mm_token_type_ids=None,
    ):
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        can_compute_mrope = input_ids is not None and has_multimodal

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=inputs_embeds.device)
        else:
            position_ids = None
        return position_ids

    def get_image_features(self, pixel_values, image_grid_thw, projector=None, **kwargs):
        """Run vision encoder, optionally project, and split outputs by image."""
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        if projector is not None:
            vision_out_dim = getattr(projector[0], "in_features", None) if isinstance(projector, nn.Sequential) else None
            if vision_out_dim is not None and image_embeds.shape[-1] != vision_out_dim:
                if image_embeds.shape[-1] < vision_out_dim:
                    raise ValueError(
                        f"Vision embedding dim {image_embeds.shape[-1]} is smaller than projector input dim {vision_out_dim}"
                    )
                image_embeds = image_embeds[..., :vision_out_dim]
            image_embeds = projector(image_embeds)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def get_video_features(self, pixel_values_videos, video_grid_thw, projector=None, **kwargs):
        """Run vision encoder for video (same as images)."""
        return self.get_image_features(pixel_values_videos, video_grid_thw, projector=projector, **kwargs)

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        """Find image/video placeholder token positions in the input."""
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self._vl_config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self._vl_config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self._vl_config.image_token_id
            special_video_mask = input_ids == self._vl_config.video_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        projector: Optional[nn.Module] = None,
        mtp_input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BailingMoeV3VLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Process images
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, projector=projector)
            image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds_cat)
            expected = int(image_embeds_cat.numel())
            actual = int(image_mask.sum().item())
            if actual != expected:
                raise ValueError(
                    f"Image placeholder token count mismatch before scatter: expected {expected} values "
                    f"from image embeddings, but found {actual} masked input values "
                    f"({expected // inputs_embeds.shape[-1]} vs {actual // inputs_embeds.shape[-1]} tokens)."
                )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

        # Process videos
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, projector=projector)
            video_embeds_cat = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds_cat)
            expected = int(video_embeds_cat.numel())
            actual = int(video_mask.sum().item())
            if actual != expected:
                raise ValueError(
                    f"Video placeholder token count mismatch before scatter: expected {expected} values "
                    f"from video embeddings, but found {actual} masked input values "
                    f"({expected // inputs_embeds.shape[-1]} vs {actual // inputs_embeds.shape[-1]} tokens)."
                )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

        # Compute 3D M-RoPE position IDs
        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = super().forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mtp_input_ids=input_ids if mtp_input_ids is None else mtp_input_ids,
            **kwargs,
        )

        return BailingMoeV3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            rope_deltas=self.rope_deltas,
        )


class BailingMoeV3VLForConditionalGeneration(BailingMoeV3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False

    def __init__(self, config: BailingMoeV3VLConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BailingMoeV3VLModel(config)
        if getattr(config.vision_config, "disable_merger_proj", False):
            vision_out_dim = config.vision_config.hidden_size * (config.vision_config.spatial_merge_size ** 2)
        else:
            vision_out_dim = config.vision_config.out_hidden_size
        self.vision_out_dim = vision_out_dim
        self.linear_proj = nn.Sequential(
            nn.Linear(vision_out_dim, config.text_config.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True),
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, projector=self.linear_proj, **kwargs)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        return self.model.get_video_features(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, projector=self.linear_proj, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, BailingMoeV3VLCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            projector=self.linear_proj,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return BailingMoeV3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        # If continuing generation with cached rope_deltas
        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            position_ids = text_positions[None, ...] + self.model.rope_deltas
            return position_ids

        # Compute 3D position ids for vision tokens
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if (
            is_input_ids
            and model_kwargs.get("mm_token_type_ids") is not None
            and (model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None)
        ):
            model_kwargs_for_rope = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
            vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs_for_rope)
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        # Concatenate [text_pos, T, H, W] → [4, bs, seq_len]
        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)

        return position_ids

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the number of images and videos per sample for beam expansion."""
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, dict]:
        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            if video_grid_thw is not None:
                cumulative_frame_counts = torch.cumsum(video_grid_thw[:, 0], dim=0)
                cumulative_token_video_counts = torch.cumsum(video_nums, dim=0)
                video_boundary_indices = torch.searchsorted(cumulative_frame_counts, cumulative_token_video_counts)
                video_nums = torch.diff(torch.cat([-video_boundary_indices.new_ones(1), video_boundary_indices]))

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if key == "position_ids" and dict_to_expand[key].ndim == 3:
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=1)
                elif (
                    dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs
