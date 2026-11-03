"""
Configuration class for EmbeddingGemma-2.
"""

import os
import json
from typing import List, Optional, Dict, Any


class EmbeddingGemma2Config:
    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        embedding_dim: int = 768,
        hidden_size_per_layer_input: int = 512,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 512,
        max_position_embeddings: int = 262144,
        pad_token_id: int = 0,
        per_layer_config: Optional[Dict] = None,
        layer_types: Optional[List[str]] = None,
        rope_parameters: Optional[Dict] = None,
        prompts: Optional[Dict[str, str]] = None,
        text_config: Optional[Dict[str, Any]] = None,
    ):
        if text_config is not None:
            vocab_size = text_config.get("vocab_size", vocab_size)
            hidden_size = text_config.get("hidden_size", hidden_size)
            intermediate_size = text_config.get("intermediate_size", intermediate_size)
            num_hidden_layers = text_config.get("num_hidden_layers", num_hidden_layers)
            num_attention_heads = text_config.get("num_attention_heads", num_attention_heads)
            num_key_value_heads = text_config.get("num_key_value_heads", num_key_value_heads)
            head_dim = text_config.get("head_dim", head_dim)
            embedding_dim = text_config.get("embedding_dim", embedding_dim)
            hidden_size_per_layer_input = text_config.get("hidden_size_per_layer_input", hidden_size_per_layer_input)
            rms_norm_eps = text_config.get("rms_norm_eps", rms_norm_eps)
            sliding_window = text_config.get("sliding_window", sliding_window)
            max_position_embeddings = text_config.get("max_position_embeddings", max_position_embeddings)
            pad_token_id = text_config.get("pad_token_id", pad_token_id)
            if per_layer_config is None:
                per_layer_config = text_config.get("per_layer_config", None)
            if layer_types is None:
                layer_types = text_config.get("layer_types", None)
            if rope_parameters is None:
                rope_parameters = text_config.get("rope_parameters", None)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.prompts = prompts or {}

        if layer_types is None:
            self.layer_types = [
                "sliding_attention" if ((i + 1) % 6 != 0) else "full_attention"
                for i in range(num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        if per_layer_config is None:
            self.per_layer_config = {
                "05": {"head_dim": 512, "num_key_value_heads": 1},
                "11": {"head_dim": 512, "num_key_value_heads": 1},
                "17": {"head_dim": 512, "num_key_value_heads": 1},
                "23": {"head_dim": 512, "num_key_value_heads": 1},
            }
        else:
            self.per_layer_config = per_layer_config

        if rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {"rope_theta": 1000000.0},
            }
        else:
            self.rope_parameters = rope_parameters

    @classmethod
    def from_model_dir(cls, model_dir: str):
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tc = data.get("text_config", data)

        prompts = {}
        st_config_path = os.path.join(model_dir, "config_sentence_transformers.json")
        if os.path.exists(st_config_path):
            with open(st_config_path, "r", encoding="utf-8") as f:
                st_data = json.load(f)
                prompts = st_data.get("prompts", {})

        return cls(
            vocab_size=tc.get("vocab_size", 262144),
            hidden_size=tc.get("hidden_size", 512),
            intermediate_size=tc.get("intermediate_size", 2048),
            num_hidden_layers=tc.get("num_hidden_layers", 24),
            num_attention_heads=tc.get("num_attention_heads", 4),
            num_key_value_heads=tc.get("num_key_value_heads", 2),
            head_dim=tc.get("head_dim", 256),
            embedding_dim=tc.get("embedding_dim", 768),
            hidden_size_per_layer_input=tc.get("hidden_size_per_layer_input", 512),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            sliding_window=tc.get("sliding_window", 512),
            max_position_embeddings=tc.get("max_position_embeddings", 262144),
            pad_token_id=tc.get("pad_token_id", 0),
            per_layer_config=tc.get("per_layer_config", None),
            layer_types=tc.get("layer_types", None),
            rope_parameters=tc.get("rope_parameters", None),
            prompts=prompts,
        )
