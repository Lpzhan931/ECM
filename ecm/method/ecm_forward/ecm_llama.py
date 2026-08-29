import torch
from tqdm import tqdm
import json
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    repeat_kv,
    rotate_half,
    LLAMA_INPUTS_DOCSTRING,
    _prepare_4d_causal_attention_mask_with_cache_position
)
import types
import math
import torch.nn as nn
from transformers import DynamicCache, Cache, logging, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.get_logger(__name__)

from ecm.util.cache_utils import CompressCache, apply_rotary_pos_emb_single

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward

cumsum_pos = True

def compute_casual_mask(lens, q_len, start_pos):
    # lens: bs * head_num * seq_len
    lens = lens.to(torch.int32) # (bs, H, s_kv): number of logical KV vectors represented by each physical KV vector
    real_pos = (lens.cumsum(dim=-1) - lens[:,:,:1]).unsqueeze(-2) # Shape: (bs, H, 1, s_kv); absolute logical positions of key vectors
    pos_idx = torch.arange(q_len, device=lens.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1) # 1, 1, q_len, 1
    pos_idx += start_pos    # Absolute logical positions of query vectors
    # Broadcast to (bs, H, q_len, s_kv): use 0.0 where a query is at or after a key, and -1e20 otherwise.
    causal_mask = torch.where(real_pos <= pos_idx, torch.tensor(0.0, device=lens.device), torch.tensor(-1e20, device=lens.device))
    return causal_mask.to(torch.bfloat16)


def llama_pos_shift_attention_ecm_forward_442(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[CompressCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # full_len tracks the logical cache length without compression; the actual cache length is the physical length.
    full_len = key_states.shape[-2]     # q_len
    past_len = 0    # Logical length of past_key_values
    if past_key_value is not None and len(past_key_value.key_cache) > self.layer_idx:
        if cumsum_pos:
            # With cache compression, logical and physical lengths differ; summing len_cache yields the logical length.
            # past_key_value.len_cache records how many original tokens each compressed KV vector represents.
            past_len = int(past_key_value.len_cache[self.layer_idx][0,0,:].long().sum())
        else:
            # Without cache compression, the logical length equals the physical length.
            past_len = past_key_value.key_cache[self.layer_idx].shape[-2]
    else:
        past_len = 0
    
    # Logical cache length (past_len + q_len), i.e., the logical number of KV vectors.
    full_len += past_len

    # if full_len>4000:
    #     print("")

    logger.warning_once(
        "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
        "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
        "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
        "removed and `position_embeddings` will be mandatory."
    )

    # Use logical positions from past_len to full_len - 1 for the q_len new query vectors.
    query_position_ids = torch.arange(past_len, full_len, device=position_ids.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    
    # Compute positions for compressed past keys and q_len newly generated keys.
    # Each physical key vector may represent multiple logical pre-compression key vectors.
    if past_key_value is not None and len(past_key_value.key_cache) > self.layer_idx:
        if cumsum_pos:
            # New keys use the same positions as queries; several position schemes are possible for compressed past keys.
            # For example, four past keys with len_cache [2, 3, 1, 2] have cumulative lengths [2, 5, 6, 8], and new positions start at 8.
            # [Method 1] Subtract the first length from the cumulative values, yielding key positions [0, 3, 4, 6] (, 8).
            key_position_ids = torch.cat((past_key_value.len_cache[self.layer_idx][:,0,:].long().cumsum(dim=-1) - past_key_value.len_cache[self.layer_idx][:,0,0].long().unsqueeze(-1), query_position_ids), dim=1) # bs * seq_len
            
            # [Method 2] Use the minimum original logical position represented by each key: [0, 2, 5, 6] (, 8).
            # key_position_ids = torch.cat((torch.zeros_like(past_key_value.len_cache[self.layer_idx][:,0,:1],dtype=torch.long), past_key_value.len_cache[self.layer_idx][:,0,:-1].long().cumsum(dim=-1), query_position_ids), dim=1)
            
            # [Method 3] Use physical positions directly: [0, 1, 2, 3] (, 8).
            # key_position_ids = torch.arange(past_key_value.len_cache[self.layer_idx].shape[2]+q_len, device=position_ids.device).unsqueeze(0)
        else:
            key_position_ids = torch.arange(full_len, device=position_ids.device).unsqueeze(0)
    else:
        key_position_ids = torch.arange(full_len, device=position_ids.device).unsqueeze(0)

    # Compute rotary embeddings.
    # value_states is used only to provide the device and dtype.
    cos_query, sin_query = self.rotary_emb(value_states, query_position_ids)
    query_states = apply_rotary_pos_emb_single(query_states, cos_query, sin_query)
    # Each layer in a CompressCache stores (key, value, length).
    len_states = torch.ones((key_states.shape[0], key_states.shape[1], q_len), device=key_states.device, dtype=key_states.dtype)
    if past_key_value is not None:
        # Unlike the original implementation, store unrotated keys because positions may change after compression.
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        key_states, value_states, len_states = past_key_value.update(key_states, value_states, len_states, self.layer_idx)
    cos_key, sin_key = self.rotary_emb(value_states, key_position_ids)
    key_states = apply_rotary_pos_emb_single(key_states, cos_key, sin_key)
    # past_key_value.updatek(key_states, self.layer_idx)
    
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    len_states = repeat_kv(len_states.unsqueeze(-1), self.num_key_value_groups).squeeze(-1)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        # causal_mask_t = attention_mask[:, :, :, : key_states.shape[-2]]
        # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        causal_mask = compute_casual_mask(len_states, q_len, past_len)
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    # Implement softmax explicitly with exp and sum.
    attn_weights = torch.exp(attn_weights.to(torch.float32))
    # Correct the denominator with len_states so it remains unchanged by compression.
    # The numerator needs no correction because compressed values are sums, rather than averages, of the original values.
    attn_weights = attn_weights /(attn_weights* len_states.unsqueeze(-2)).sum(dim=-1, keepdim=True)
    attn_weights = attn_weights.to(query_states.dtype)

    # attn_weights = torch.exp(attn_weights) * len_states.unsqueeze(-2)
    # attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
    
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# # Unused; self.rotary_emb is used instead.
# def LlamaRotaryEmbedding_forward(self, x, position_ids):
#     # Accumulate over the last dimension of l.

#     this_len = int(position_ids.view(-1).max() + 1)
#     if this_len > self.max_seq_len_cached:
#         self._set_cos_sin_cache(seq_len=this_len, device=x.device, dtype=x.dtype)
    
#     # Core RoPE block
#     inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#     position_ids_expanded = position_ids[:, None, :].float()
#     # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
#     device_type = x.device.type
#     device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
#     with torch.autocast(device_type=device_type, enabled=False):
#         freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#         emb = torch.cat((freqs, freqs), dim=-1)
#         cos = emb.cos()
#         sin = emb.sin()

#     # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
#     cos = cos * self.attention_scaling
#     sin = sin * self.attention_scaling

#     return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def enable_llama_pos_shift_ecm_attention_recursive(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_pos_shift_ecm_attention_recursive(
                module,
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                llama_pos_shift_attention_ecm_forward_442, model._modules[name]
            )


def enable_llama_pos_shift_ecm_attention_442(model):
    model.model.forward = types.MethodType(LlamaModel_forward, model.model)
    enable_llama_pos_shift_ecm_attention_recursive(model)


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def LlamaModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if (
        use_cache and not isinstance(past_key_values, Cache) # and not self.training; this also initializes the cache during training
    ):  # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = True
        ###
        past_key_values = CompressCache.from_legacy_cache(past_key_values)
        ###
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0] 

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
