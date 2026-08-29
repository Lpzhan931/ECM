from ecm.streaming_llm.kv_cache import StartRecentKVCache
import torch
import math
import torch.nn.functional as F
from ecm.method.ecm_forward.ecm_mistral import enable_mistral_ecm_attention
from ecm.method.ecm_forward.ecm_llama import enable_llama_pos_shift_ecm_attention_442
from ecm.method.ecm_forward.ecm_qwen2 import enable_qwen2_ecm_attention


def enable_ecm_attention(model_name, model):
    if "llama" in model_name.lower():
        enable_llama_pos_shift_ecm_attention_442(model)
    elif "mistral" in model_name.lower():
        enable_mistral_ecm_attention(model)
    elif "qwen" in model_name.lower():
        enable_qwen2_ecm_attention(model)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def repeat_l(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    Hidden states go from (batch, num_key_value_heads, seqlen)
    to (batch, num_attention_heads, seqlen).
    """
    batch, num_key_value_heads, slen = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(batch, num_key_value_heads, n_rep, slen)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen)


def compress_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Compress hidden states from (batch, num_attention_heads, seqlen, head_dim)
    to (batch, num_key_value_heads, seqlen, head_dim) by taking the mean
    of every n_rep heads.
    """
    batch, num_attention_heads, slen, head_dim = hidden_states.shape
    num_key_value_heads = num_attention_heads // n_rep
    hidden_states = hidden_states.view(batch, num_key_value_heads, n_rep, slen, head_dim)
    compressed_hidden_states = hidden_states.mean(dim=2)
    return compressed_hidden_states


class ECMCache(StartRecentKVCache):
    """
    ECM (Expected Contribution Matching) KV Cache compression.

    Pair-level eviction selection combined with global semantic similarity
    matching and ECM closed-form value redistribution.
    """

    def __init__(self, *args, merge_num=16, lam=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.accum_attn = None
        self.merge_num = merge_num
        self.lam = lam

    def _ensure_kvl(self, past_key_values):
        if past_key_values is None:
            return None
        if len(past_key_values[0]) == 2:
            ret = []
            for k, v in past_key_values:
                l = torch.ones(k.shape[:-1], device=k.device, dtype=k.dtype)
                ret.append((k, v, l))
            return tuple(ret)
        return past_key_values

    def __call__(self, past_key_values, attns):
        if past_key_values is None:
            return None

        past_key_values = self._ensure_kvl(past_key_values)

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        if self.accum_attn is None:
            self.accum_attn = [None] * len(past_key_values)

        new_mid = []
        for i, (k, v, l) in enumerate(past_key_values):
            mid_k = self.k_slice(k, self.start_size, seq_len)
            mid_v = self.v_slice(v, self.start_size, seq_len)
            mid_l = l[:, :, self.start_size:seq_len]
            mid_len = mid_k.shape[2]

            cur_attn = attns[i][:, :, :, self.start_size:].sum(dim=-2)
            if self.accum_attn[i] is None:
                self.accum_attn[i] = cur_attn
            else:
                prev_accum = self.accum_attn[i]
                tmp = cur_attn[..., :prev_accum.shape[-1]] + prev_accum
                self.accum_attn[i] = torch.cat([tmp, cur_attn[..., prev_accum.shape[-1]:]], dim=-1)
            accum_attn_i = self.accum_attn[i]

            # 1. Pair-level eviction selection
            repeat_mid_l = repeat_l(mid_l, int(cur_attn.shape[1] / mid_l.shape[1]))
            cur_attn_weighted = cur_attn * repeat_mid_l[:, :, :cur_attn.shape[2]]
            cur_attn_weighted = cur_attn_weighted[:, :, :-1] + cur_attn_weighted[:, :, 1:]
            weight_i = cur_attn_weighted.sum(dim=1).squeeze(0)

            l_i = mid_l[:, :, :].sum(dim=-2)
            l_i = l_i[:, :-1] + l_i[:, 1:]
            l_i = l_i[0]

            gamma = 4096 / math.log(512)
            weight_idx = torch.arange(1, len(weight_i) + 1, dtype=torch.float, device=weight_i.device)
            sqrt_indices = torch.exp(weight_idx / gamma)

            weight_i = weight_i / sqrt_indices * l_i
            K_evict = seq_len - self.cache_size

            evict_indices = weight_i.topk(K_evict, dim=-1, largest=False).indices

            # 2. Global semantic similarity matching
            bsz, num_kv_heads, _, head_dim = mid_v.shape

            v_norm = F.normalize(mid_v, p=2, dim=-1)
            v_evict_norm = v_norm[:, :, evict_indices, :]

            sim_matrix = torch.matmul(v_evict_norm, v_norm.transpose(-1, -2))
            sim_matrix[:, :, :, evict_indices] = -float('inf')

            target_indices = sim_matrix.topk(self.merge_num, dim=-1, largest=True).indices

            # 3. ECM weight computation
            num_heads = accum_attn_i.shape[1]
            if num_heads != num_kv_heads:
                n_rep = num_heads // num_kv_heads
                accum_attn_kv = accum_attn_i.view(bsz, num_kv_heads, n_rep, mid_len).mean(dim=2)
            else:
                accum_attn_kv = accum_attn_i

            a_sum = accum_attn_kv.sum(dim=-1, keepdim=True) + 1e-8
            a_hat = accum_attn_kv / a_sum

            a_hat_i = a_hat[:, :, evict_indices].unsqueeze(-1)
            a_hat_expanded = a_hat.unsqueeze(2).expand(-1, -1, K_evict, -1)
            a_hat_targets = torch.gather(a_hat_expanded, dim=3, index=target_indices)

            numerator = a_hat_i * a_hat_targets
            denominator = (a_hat_targets ** 2).sum(dim=-1, keepdim=True) + self.lam
            w = numerator / denominator

            # 4. Parallel scatter value merging
            v_evict = mid_v[:, :, evict_indices, :]
            spread_v = v_evict.unsqueeze(3) * w.unsqueeze(-1)

            src = spread_v.view(bsz, num_kv_heads, K_evict * self.merge_num, head_dim)
            idx_flat = target_indices.view(bsz, num_kv_heads, K_evict * self.merge_num)
            idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            mid_v.scatter_add_(dim=2, index=idx_expanded, src=src)

            # 5. Remove evicted tokens
            keep_mask = torch.ones(mid_len, dtype=torch.bool, device=mid_k.device)
            keep_mask[evict_indices] = False

            new_mid_k = mid_k[:, :, keep_mask, :]
            new_mid_v = mid_v[:, :, keep_mask, :]
            new_mid_l = mid_l[:, :, keep_mask]

            self.accum_attn[i] = self.accum_attn[i][:, :, keep_mask]
            new_mid_l = torch.clip(new_mid_l, max=5)
            new_mid.append((new_mid_k, new_mid_v, new_mid_l))

        return [
            [
                torch.cat([self.k_slice(k, 0, self.start_size), new_k], dim=self.k_seq_dim),
                torch.cat([self.v_slice(v, 0, self.start_size), new_v], dim=self.v_seq_dim),
                torch.cat([l[:, :, :self.start_size], new_l], dim=2),
            ]
            for (k, v, l), (new_k, new_v, new_l) in zip(past_key_values, new_mid)
        ]
