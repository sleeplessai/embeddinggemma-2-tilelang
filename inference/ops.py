"""
Fused GPU Operators for EmbeddingGemma-2.
=========================================
Self-contained kernel implementations using TileLang for:
  - RMSNorm (512-dim, 256-dim, with/without scale)
  - Fused GeGLU activation (2048-dim)
  - Fused RoPE rotary embedding (256-dim, 512-dim)
  - Fused MeanPool + L2Norm (768-dim, collaborative SRAM reduction)
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import tilelang
import tilelang.language as T

_KERNEL_CACHE = {}

SQRT_2_OVER_PI = 0.7978845608028654
GELU_COEFF = 0.044715


def get_target_device() -> str:
    return "cuda"


# ==============================================================================
# 1. RMSNorm
# ==============================================================================

def get_rmsnorm_kernel(dim: int, eps: float = 1e-6, dtype: str = "bfloat16"):
    key = ("rmsnorm", dim, eps, dtype)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    threads = 128

    @T.prim_func
    def rms_norm_kernel(
        X: T.Buffer((T.dynamic("batch"), dim), dtype),
        Weight: T.Buffer((dim,), dtype),
        Y: T.Buffer((T.dynamic("batch"), dim), dtype),
    ):
        num_rows = X.shape[0]
        with T.Kernel(num_rows, threads=threads) as bx:
            X_shared = T.alloc_shared((dim,), dtype)
            W_shared = T.alloc_shared((dim,), dtype)
            T.copy(X[bx, :], X_shared)
            T.copy(Weight[:], W_shared)

            acc = T.alloc_fragment((1,), "float32")
            T.clear(acc)
            for i in T.serial(dim):
                val = T.Cast("float32", X_shared[i])
                acc[0] += val * val

            rsqrt_val = T.rsqrt(acc[0] / float(dim) + eps)

            for i in T.Parallel(dim):
                normed = T.Cast("float32", X_shared[i]) * rsqrt_val
                scaled = normed * T.Cast("float32", W_shared[i])
                Y[bx, i] = T.Cast(dtype, scaled)

    compiled = tilelang.compile(rms_norm_kernel, target=get_target_device())
    _KERNEL_CACHE[key] = compiled
    return compiled


def get_rmsnorm_no_weight_kernel(dim: int, eps: float = 1e-6, dtype: str = "bfloat16"):
    key = ("rmsnorm_no_weight", dim, eps, dtype)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    threads = 128

    @T.prim_func
    def rms_norm_no_weight_kernel(
        X: T.Buffer((T.dynamic("batch"), dim), dtype),
        Y: T.Buffer((T.dynamic("batch"), dim), dtype),
    ):
        num_rows = X.shape[0]
        with T.Kernel(num_rows, threads=threads) as bx:
            X_shared = T.alloc_shared((dim,), dtype)
            T.copy(X[bx, :], X_shared)

            acc = T.alloc_fragment((1,), "float32")
            T.clear(acc)
            for i in T.serial(dim):
                val = T.Cast("float32", X_shared[i])
                acc[0] += val * val

            rsqrt_val = T.rsqrt(acc[0] / float(dim) + eps)

            for i in T.Parallel(dim):
                normed = T.Cast("float32", X_shared[i]) * rsqrt_val
                Y[bx, i] = T.Cast(dtype, normed)

    compiled = tilelang.compile(rms_norm_no_weight_kernel, target=get_target_device())
    _KERNEL_CACHE[key] = compiled
    return compiled


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.with_scale = with_scale
        self.dtype_str = "float32" if dtype == torch.float32 else "bfloat16"

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
            self._kernel = get_rmsnorm_kernel(dim, eps, self.dtype_str)
        else:
            self.register_parameter("weight", None)
            self._kernel = get_rmsnorm_no_weight_kernel(dim, eps, self.dtype_str)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.dim).contiguous()
        y_2d = torch.empty_like(x_2d)
        if self.with_scale:
            w = self.weight.to(x.device) if self.weight.device != x.device else self.weight
            self._kernel(x_2d, w.contiguous(), y_2d)
        else:
            self._kernel(x_2d, y_2d)
        return y_2d.view(orig_shape)


# ==============================================================================
# 2. GeGLU
# ==============================================================================

def get_geglu_kernel(dim: int, dtype: str = "bfloat16", threads: int = 512):
    key = ("geglu", dim, dtype, threads)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    @T.prim_func
    def geglu_kernel(
        Gate: T.Buffer((T.dynamic("batch"), dim), dtype),
        Up: T.Buffer((T.dynamic("batch"), dim), dtype),
        Out: T.Buffer((T.dynamic("batch"), dim), dtype),
    ):
        num_rows = Gate.shape[0]
        with T.Kernel(num_rows, threads=threads) as bx:
            for i in T.Parallel(dim):
                g = T.Cast("float32", Gate[bx, i])
                u = T.Cast("float32", Up[bx, i])
                inner = SQRT_2_OVER_PI * (g + GELU_COEFF * g * g * g)
                act = 0.5 * g * (1.0 + T.tanh(inner))
                res = act * u
                Out[bx, i] = T.Cast(dtype, res)

    compiled = tilelang.compile(geglu_kernel, target=get_target_device())
    _KERNEL_CACHE[key] = compiled
    return compiled


class GeGLU(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.dtype_str = "float32" if dtype == torch.float32 else "bfloat16"
        self._kernel = get_geglu_kernel(dim, self.dtype_str)

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        orig_shape = gate.shape
        g_2d = gate.reshape(-1, self.dim).contiguous()
        u_2d = up.reshape(-1, self.dim).contiguous()
        out_2d = torch.empty_like(g_2d)
        self._kernel(g_2d, u_2d, out_2d)
        return out_2d.view(orig_shape)


# ==============================================================================
# 3. RoPE
# ==============================================================================

def get_rope_kernel(head_dim: int, dtype: str = "bfloat16"):
    key = ("rope", head_dim, dtype)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    half_dim = head_dim // 2
    threads = 128

    @T.prim_func
    def rope_kernel(
        Q: T.Buffer((T.dynamic("batch"), T.dynamic("heads"), T.dynamic("seq"), head_dim), dtype),
        Cos: T.Buffer((T.dynamic("batch"), T.dynamic("seq"), head_dim), dtype),
        Sin: T.Buffer((T.dynamic("batch"), T.dynamic("seq"), head_dim), dtype),
        Q_out: T.Buffer((T.dynamic("batch"), T.dynamic("heads"), T.dynamic("seq"), head_dim), dtype),
    ):
        batch = Q.shape[0]
        heads = Q.shape[1]
        seq = Q.shape[2]
        total_tokens = batch * heads * seq
        with T.Kernel(total_tokens, threads=threads) as tid:
            s_idx = tid % seq
            rem = tid // seq
            h_idx = rem % heads
            b_idx = rem // heads

            q_sh = T.alloc_shared((head_dim,), dtype)
            cos_sh = T.alloc_shared((head_dim,), dtype)
            sin_sh = T.alloc_shared((head_dim,), dtype)

            T.copy(Q[b_idx, h_idx, s_idx, :], q_sh)
            T.copy(Cos[b_idx, s_idx, :], cos_sh)
            T.copy(Sin[b_idx, s_idx, :], sin_sh)

            for i in T.Parallel(half_dim):
                q1 = T.Cast("float32", q_sh[i])
                q2 = T.Cast("float32", q_sh[i + half_dim])
                c1 = T.Cast("float32", cos_sh[i])
                s1 = T.Cast("float32", sin_sh[i])
                c2 = T.Cast("float32", cos_sh[i + half_dim])
                s2 = T.Cast("float32", sin_sh[i + half_dim])

                rot1 = q1 * c1 - q2 * s1
                rot2 = q2 * c2 + q1 * s2

                Q_out[b_idx, h_idx, s_idx, i] = T.Cast(dtype, rot1)
                Q_out[b_idx, h_idx, s_idx, i + half_dim] = T.Cast(dtype, rot2)

    compiled = tilelang.compile(rope_kernel, target=get_target_device())
    _KERNEL_CACHE[key] = compiled
    return compiled


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    dtype_str = "float32" if q.dtype == torch.float32 else "bfloat16"
    kernel = get_rope_kernel(head_dim, dtype_str)

    batch = q.shape[0]
    if cos.shape[0] != batch:
        cos = cos.expand(batch, -1, -1)
        sin = sin.expand(batch, -1, -1)

    q_c = q.contiguous()
    k_c = k.contiguous()
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    q_out = torch.empty_like(q_c)
    k_out = torch.empty_like(k_c)

    kernel(q_c, cos_c, sin_c, q_out)
    kernel(k_c, cos_c, sin_c, k_out)
    return q_out, k_out


# ==============================================================================
# 4. Fused MeanPool + L2Norm
# ==============================================================================

def get_fused_pool_norm_kernel(dim: int = 768, dtype: str = "bfloat16"):
    key = ("pool_norm_sram", dim, dtype)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    threads = 128

    @T.prim_func
    def pool_norm_sram_kernel(
        H: T.Buffer((T.dynamic("batch"), T.dynamic("seq"), dim), dtype),
        Mask: T.Buffer((T.dynamic("batch"), T.dynamic("seq")), "int64"),
        Out: T.Buffer((T.dynamic("batch"), dim), dtype),
    ):
        num_rows = H.shape[0]
        seq_len = H.shape[1]
        with T.Kernel(num_rows, threads=threads) as bx:
            mean_s = T.alloc_shared((dim,), "float32")
            for d in T.Parallel(dim):
                mean_s[d] = 0.0

            mask_sum = T.alloc_fragment((1,), "float32")
            mask_sum[0] = 0.0

            for s in T.serial(seq_len):
                m_val = T.Cast("float32", Mask[bx, s])
                mask_sum[0] += m_val
                if m_val > 0.0:
                    for d in T.Parallel(dim):
                        mean_s[d] += T.Cast("float32", H[bx, s, d])

            denom = T.max(mask_sum[0], 1.0)
            for d in T.Parallel(dim):
                mean_s[d] = mean_s[d] / denom

            l2_acc = T.alloc_fragment((1,), "float32")
            T.clear(l2_acc)
            for d in T.serial(dim):
                val = mean_s[d]
                l2_acc[0] += val * val

            inv_l2 = T.rsqrt(T.max(l2_acc[0], 1e-12))
            for d in T.Parallel(dim):
                Out[bx, d] = T.Cast(dtype, mean_s[d] * inv_l2)

    compiled = tilelang.compile(pool_norm_sram_kernel, target=get_target_device())
    _KERNEL_CACHE[key] = compiled
    return compiled


class FusedMeanPoolNormalize(nn.Module):
    def __init__(self, dim: int = 768, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.dtype_str = "float32" if dtype == torch.float32 else "bfloat16"
        self._kernel = get_fused_pool_norm_kernel(dim, self.dtype_str)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=hidden_states.device)

        out = torch.empty(
            (batch_size, self.dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._kernel(hidden_states.contiguous(), attention_mask.contiguous(), out)
        return out


def warmup_ops(dtype: str = "bfloat16"):
    get_rmsnorm_kernel(512, dtype=dtype)
    get_rmsnorm_kernel(256, dtype=dtype)
    get_rmsnorm_no_weight_kernel(512, dtype=dtype)
    get_rmsnorm_no_weight_kernel(256, dtype=dtype)
    get_geglu_kernel(2048, dtype=dtype)
    get_rope_kernel(256, dtype=dtype)
    get_rope_kernel(512, dtype=dtype)
    get_fused_pool_norm_kernel(768, dtype=dtype)
