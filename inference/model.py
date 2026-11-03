"""
EmbeddingGemma-2 High-Performance Neural Representation Model.
=============================================================
End-to-end model implementation integrating custom GPU kernels for:
  - RMSNorm (512-dim, 256-dim, with_scale and without_scale)
  - Fused GeGLU activation (2048-dim)
  - Fused PLE gate (512-dim)
  - Fused RoPE rotary position embedding (256-dim, 512-dim)
  - Fused token Mean-Pooling + L2 Normalization (768-dim)
"""

import os
import sys
import json
import struct
import math
from typing import List, Union, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

try:
    from .configuration import EmbeddingGemma2Config
    from .ops import RMSNorm, GeGLU, apply_rope, FusedMeanPoolNormalize, warmup_ops
except ImportError:
    from configuration import EmbeddingGemma2Config
    from ops import RMSNorm, GeGLU, apply_rope, FusedMeanPoolNormalize, warmup_ops


class EmbeddingGemma2RotaryEmbedding(nn.Module):
    def __init__(self, config: EmbeddingGemma2Config, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.freqs = {}

        for layer_type in ["sliding_attention", "full_attention"]:
            base = config.rope_parameters[layer_type]["rope_theta"]
            dim = 256 if layer_type == "sliding_attention" else 512
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            self.freqs[layer_type] = inv_freq

    def get_cos_sin(self, seq_len: int, layer_type: str, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.freqs[layer_type]
        t = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).to(dtype)
        return cos, sin


class EmbeddingGemma2Attention(nn.Module):
    def __init__(self, config: EmbeddingGemma2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = (self.layer_type == "sliding_attention")

        key_str = f"{layer_idx:02d}"
        plc = config.per_layer_config or {}
        if key_str in plc:
            layer_cfg = plc[key_str]
            self.head_dim = layer_cfg.get("head_dim", config.head_dim)
            self.num_kv_heads = layer_cfg.get("num_key_value_heads", config.num_key_value_heads)
        else:
            self.head_dim = config.head_dim
            self.num_kv_heads = config.num_key_value_heads

        self.num_heads = config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=True)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=True)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_norm(v).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=1.0
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


class EmbeddingGemma2MLP(nn.Module):
    def __init__(self, config: EmbeddingGemma2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.geglu = GeGLU(config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        act = self.geglu(gate, up)
        return self.down_proj(act)


class EmbeddingGemma2PLEBlock(nn.Module):
    def __init__(self, config: EmbeddingGemma2Config):
        super().__init__()
        dim = config.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(config.hidden_size, dim, bias=False)
        self.ple_gate = GeGLU(dim)
        self.per_layer_projection = nn.Linear(dim, config.hidden_size, bias=False)
        self.post_per_layer_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, per_layer_input: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        gate = self.per_layer_input_gate(hidden_states)
        mixed = self.ple_gate(gate, per_layer_input)
        proj = self.per_layer_projection(mixed)
        normed = self.post_per_layer_input_norm(proj)
        return residual + normed


class EmbeddingGemma2DecoderLayer(nn.Module):
    def __init__(self, config: EmbeddingGemma2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = (self.layer_type == "sliding_attention")

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = EmbeddingGemma2Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = EmbeddingGemma2MLP(config)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.ple_block = EmbeddingGemma2PLEBlock(config)
        self.layer_scalar = nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, cos, sin, attn_mask)
        attn_out = self.post_attention_layernorm(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        normed = self.pre_feedforward_layernorm(hidden_states)
        mlp_out = self.mlp(normed)
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        hidden_states = residual + mlp_out

        hidden_states = self.ple_block(hidden_states, per_layer_input)

        if self.layer_scalar.item() != 1.0:
            hidden_states = hidden_states * self.layer_scalar

        return hidden_states


class EmbeddingGemma2(nn.Module):
    def __init__(self, config: EmbeddingGemma2Config, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.config = config
        self.device = device
        self.tokenizer = None

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embed_scale = math.sqrt(config.hidden_size)

        self.ple_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.ple_scale = 1.0 / math.sqrt(config.hidden_size)
        self.ple_norm = RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        self.rotary_emb = EmbeddingGemma2RotaryEmbedding(config, device=device)

        self.layers = nn.ModuleList([
            EmbeddingGemma2DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_projection = nn.Linear(config.hidden_size, config.embedding_dim, bias=False)
        self.pool_and_norm = FusedMeanPoolNormalize(config.embedding_dim)

        self.to(device=device, dtype=torch.bfloat16)

    def compute_ple(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = inputs_embeds.shape
        ple = self.ple_model_projection(inputs_embeds) * self.ple_scale
        ple = ple.view(batch, seq_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input)
        return self.ple_norm(ple)

    def create_masks(self, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = attention_mask.shape
        key_valid = attention_mask[:, None, None, :].bool()
        query_valid = attention_mask[:, None, :, None].bool()
        valid_pair = key_valid & query_valid

        full_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bfloat16, device=attention_mask.device)
        full_mask = full_mask.masked_fill(~valid_pair, float("-inf"))

        idx = torch.arange(seq_len, device=attention_mask.device)
        dist = (idx[:, None] - idx[None, :]).abs()
        in_window = (dist <= self.config.sliding_window)[None, None, :, :]
        sliding_valid = valid_pair & in_window
        sliding_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bfloat16, device=attention_mask.device)
        sliding_mask = sliding_mask.masked_fill(~sliding_valid, float("-inf"))

        return full_mask, sliding_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_token_embeddings: bool = False,
        truncate_dim: Optional[int] = None,
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        x = self.embed_tokens(input_ids) * self.embed_scale
        per_layer_inputs = self.compute_ple(x)

        cos_sliding, sin_sliding = self.rotary_emb.get_cos_sin(seq_len, "sliding_attention", torch.bfloat16)
        cos_full, sin_full = self.rotary_emb.get_cos_sin(seq_len, "full_attention", torch.bfloat16)
        full_mask, sliding_mask = self.create_masks(attention_mask)

        for i, layer in enumerate(self.layers):
            ple_slice = per_layer_inputs[:, :, i, :]
            if layer.is_sliding:
                cos, sin, mask = cos_sliding, sin_sliding, sliding_mask
            else:
                cos, sin, mask = cos_full, sin_full, full_mask
            x = layer(x, ple_slice, cos, sin, mask)

        x = self.norm(x)
        token_embs = self.embedding_projection(x)
        pooled = self.pool_and_norm(token_embs, attention_mask)

        if truncate_dim is not None and truncate_dim < self.config.embedding_dim:
            pooled = pooled[:, :truncate_dim]
            pooled = F.normalize(pooled.float(), p=2, dim=-1).bfloat16()

        if return_token_embeddings:
            return pooled, token_embs
        return pooled

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[str, List[str]],
        prompt_name: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        batch_size: int = 32,
        max_length: int = 1024,
    ) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Call load_tokenizer() first.")

        if isinstance(sentences, str):
            sentences = [sentences]

        if prompt_name and prompt_name in self.config.prompts:
            prefix = self.config.prompts[prompt_name]
            sentences = [prefix + s for s in sentences]

        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            encoded = [self.tokenizer.encode(t) for t in batch_texts]
            lens = [len(e.ids) for e in encoded]
            max_len = min(max_length, max(lens))

            input_ids = torch.zeros((len(batch_texts), max_len), dtype=torch.int64, device=self.device)
            attention_mask = torch.zeros((len(batch_texts), max_len), dtype=torch.int64, device=self.device)

            for b_idx, e in enumerate(encoded):
                cur_len = min(len(e.ids), max_len)
                input_ids[b_idx, :cur_len] = torch.tensor(e.ids[:cur_len], dtype=torch.int64, device=self.device)
                attention_mask[b_idx, :cur_len] = 1

            embs = self.forward(input_ids, attention_mask=attention_mask, truncate_dim=truncate_dim)
            all_embeddings.append(embs)

        return torch.cat(all_embeddings, dim=0)

    def load_tokenizer(self, tokenizer_path: str):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "The 'tokenizers' library is required to tokenize text input. "
                "Please install it with: pip install tokenizers"
            )

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        config = EmbeddingGemma2Config.from_model_dir(model_dir)
        dev = torch.device(device)
        model = cls(config, device=dev)

        print(f"[EmbeddingGemma-2] Loading weights from {safetensors_path}...")
        _load_safetensors_weights(model, safetensors_path, dev)
        print("[EmbeddingGemma-2] Model weights loaded successfully!")

        if os.path.exists(tokenizer_path):
            try:
                model.load_tokenizer(tokenizer_path)
                print("[EmbeddingGemma-2] Tokenizer loaded successfully!")
            except Exception as e:
                print(f"[EmbeddingGemma-2] Warning: Could not load tokenizer: {e}")

        return model


def _load_safetensors_weights(model: EmbeddingGemma2, path: str, device: torch.device):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))

        dtype_map = {
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
            "I32": torch.int32,
            "I64": torch.int64,
        }

        for key, info in header.items():
            if not key.startswith("language_model."):
                continue

            rel_key = key[len("language_model."):]
            start, end = info["data_offsets"]
            shape = info["shape"]
            dt = dtype_map[info["dtype"]]

            f.seek(8 + header_len + start)
            data = bytearray(f.read(end - start))
            tensor = torch.frombuffer(data, dtype=dt).reshape(shape).to(device)

            if rel_key == "embed_tokens.weight":
                model.embed_tokens.weight.data.copy_(tensor)
            elif rel_key == "embedding_projection.weight":
                model.embedding_projection.weight.data.copy_(tensor)
            elif rel_key == "norm.weight":
                model.norm.weight.data.copy_(tensor)
            elif rel_key == "ple.per_layer_model_projection.weight":
                model.ple_model_projection.weight.data.copy_(tensor)
            elif rel_key == "ple.per_layer_projection_norm.weight":
                model.ple_norm.weight.data.copy_(tensor)
            elif rel_key.startswith("layers."):
                parts = rel_key.split(".")
                layer_idx = int(parts[1])
                sub = ".".join(parts[2:])
                layer = model.layers[layer_idx]

                if sub == "input_layernorm.weight":
                    layer.input_layernorm.weight.data.copy_(tensor)
                elif sub == "post_attention_layernorm.weight":
                    layer.post_attention_layernorm.weight.data.copy_(tensor)
                elif sub == "pre_feedforward_layernorm.weight":
                    layer.pre_feedforward_layernorm.weight.data.copy_(tensor)
                elif sub == "post_feedforward_layernorm.weight":
                    layer.post_feedforward_layernorm.weight.data.copy_(tensor)
                elif sub == "layer_scalar":
                    layer.layer_scalar.data.copy_(tensor)
                elif sub == "self_attn.q_proj.weight":
                    layer.self_attn.q_proj.weight.data.copy_(tensor)
                elif sub == "self_attn.k_proj.weight":
                    layer.self_attn.k_proj.weight.data.copy_(tensor)
                elif sub == "self_attn.v_proj.weight":
                    layer.self_attn.v_proj.weight.data.copy_(tensor)
                elif sub == "self_attn.o_proj.weight":
                    layer.self_attn.o_proj.weight.data.copy_(tensor)
                elif sub == "self_attn.q_norm.weight":
                    layer.self_attn.q_norm.weight.data.copy_(tensor)
                elif sub == "self_attn.k_norm.weight":
                    layer.self_attn.k_norm.weight.data.copy_(tensor)
                elif sub == "mlp.gate_proj.weight":
                    layer.mlp.gate_proj.weight.data.copy_(tensor)
                elif sub == "mlp.up_proj.weight":
                    layer.mlp.up_proj.weight.data.copy_(tensor)
                elif sub == "mlp.down_proj.weight":
                    layer.mlp.down_proj.weight.data.copy_(tensor)
                elif sub == "ple_block.per_layer_input_gate.weight":
                    layer.ple_block.per_layer_input_gate.weight.data.copy_(tensor)
                elif sub == "ple_block.per_layer_projection.weight":
                    layer.ple_block.per_layer_projection.weight.data.copy_(tensor)
                elif sub == "ple_block.post_per_layer_input_norm.weight":
                    layer.ple_block.post_per_layer_input_norm.weight.data.copy_(tensor)
