# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant online weight quantization for vLLM.

3-4 bit weight compression via WHT rotation + Lloyd-Max codebook.
Load any BF16 checkpoint, compress weights at startup, serve with
~4x smaller GPU memory. Zero calibration data needed.

Algorithm: scalar case of HIGGS (Malinovskii et al., NAACL 2025,
aclanthology.org/2025.naacl-long.543; preprint arXiv:2411.17525) —
Random Hadamard Transform + MSE-optimal Lloyd-Max grid + per-group
normalization. The implementation was originally based on TurboQuant
(Zandieh et al., ICLR 2026, arXiv:2504.19874), which targets online
KV-cache and ANN vector search; engineering simplifications (scalar
over vector, WHT over general random rotations) converged the weight
path onto the HIGGS scalar case. The ``turboquant`` name is kept
for API and plugin-package compatibility.

Usage:
    vllm serve <model> --quantization turboquant
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.model_executor.layers.quantization.turboquant.quantizer import (
    generate_wht_signs,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.utils.math_utils import next_power_of_2, round_up

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform
# ---------------------------------------------------------------------------


def _fast_wht_batch(x: torch.Tensor) -> torch.Tensor:
    """Batched fast WHT. x: (batch, n) where n is power of 2."""
    n = x.shape[1]
    h = 1
    while h < n:
        x_view = x.view(x.shape[0], n // (h * 2), 2, h)
        a = x_view[:, :, 0, :].clone()
        b = x_view[:, :, 1, :].clone()
        x_view[:, :, 0, :] = a + b
        x_view[:, :, 1, :] = a - b
        h *= 2
    return x.div_(math.sqrt(n))


def _fast_wht_batch_blocked(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Batched fast WHT applied independently to each block of width ``block_size``.

    Equivalent to ``block_diag(H, H, ..., H) @ x`` where each ``H`` is a
    ``block_size × block_size`` Walsh-Hadamard matrix. Used for partial-rotary
    models (MiniMax M2.5/M2.7, Qwen3.6-A3B, etc.) so the RoPE-rotated prefix
    and the content-only suffix are kept under separate rotations and don't
    mix inside a group (MidasMining, PR #39970 comment, 2026-04-21).

    ``x.shape[1]`` must be a multiple of ``block_size``; ``block_size`` must
    be a power of two.
    """
    n = x.shape[1]
    assert n % block_size == 0, (
        f"dim {n} not divisible by block_size {block_size}"
    )
    assert block_size & (block_size - 1) == 0, (
        f"block_size {block_size} must be a power of two"
    )
    num_blocks = n // block_size
    x_blocks = x.reshape(x.shape[0] * num_blocks, block_size)
    x_blocks = _fast_wht_batch(x_blocks)
    return x_blocks.reshape(x.shape[0], n)




# ---------------------------------------------------------------------------
# PolarQuant quantizer
# ---------------------------------------------------------------------------

# Cache: (group_size, bits, device_str, rotary_dim) → PolarQuant instance
_quantizers: dict[tuple[int, int, str, "int | None"], "_PolarQuant"] = {}


def _get_quantizer(
    group_size: int,
    bits: int,
    device: str,
    rotary_dim: "int | None" = None,
) -> "_PolarQuant":
    """Return a cached ``_PolarQuant``.

    ``rotary_dim`` selects between a full-width WHT (``None``) and a block-
    diagonal WHT (each block ``rotary_dim`` wide). Only meaningful when
    ``rotary_dim < group_size``; otherwise folded to ``None`` so full-width
    quantizers are shared.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    effective_rotary_dim = (
        rotary_dim
        if (rotary_dim is not None and rotary_dim < group_size)
        else None
    )
    key = (group_size, bits, str(dev), effective_rotary_dim)
    if key not in _quantizers:
        _quantizers[key] = _PolarQuant(
            group_size, bits, device=str(dev),
            rotary_dim=effective_rotary_dim,
        )
    return _quantizers[key]


def _derive_rotary_dim(model_config, head_dim: int) -> "int | None":
    """Extract a block-diag rotary_dim from an HF text config, or None.

    Returns ``None`` (→ full-width WHT) unless ``partial_rotary_factor``
    is set to a value in ``(0, 1)``. When set, returns the largest power
    of two that divides ``head_dim`` and is ≤ ``head_dim * factor``.

    Examples:
        Qwen3.6-35B-A3B (head_dim=256, factor=0.25) → 64
        MiniMax M2.7   (head_dim=128, factor=0.5)  → 64
        Qwen3-8B       (head_dim=128, factor=1.0)  → None
    """
    if model_config is None:
        return None
    cfg = (
        getattr(model_config, "hf_text_config", None)
        or getattr(model_config, "hf_config", None)
        or model_config
    )
    if cfg is None:
        return None
    factor = getattr(cfg, "partial_rotary_factor", None)
    if factor is None:
        rope_params = getattr(cfg, "rope_parameters", None)
        if isinstance(rope_params, dict):
            factor = rope_params.get("partial_rotary_factor")
    if factor is None or factor >= 1.0 or factor <= 0.0:
        return None
    rd_target = int(head_dim * factor)
    if rd_target <= 0 or rd_target >= head_dim:
        return None
    rd_p2 = 1 << (rd_target.bit_length() - 1)
    if rd_p2 <= 0 or head_dim % rd_p2 != 0:
        return None
    return rd_p2


class _PolarQuant:
    """WHT rotation + Gaussian Lloyd-Max codebook quantizer.

    ``rotary_dim`` selects full-width WHT (``None``) vs block-diagonal WHT
    with independent ``rotary_dim``-wide blocks. See ``_fast_wht_batch_blocked``.
    """

    def __init__(
        self,
        dim: int,
        bits: int,
        seed: int = 42,
        device: str = "cuda",
        rotary_dim: "int | None" = None,
    ):
        self.dim = dim
        self.bits = bits
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev

        # Reuse the shared sign-vector helper (added in #38479). Two
        # independent seeds keep signs1 and signs2 uncorrelated — both
        # are needed by the randomized Walsh-Hadamard rotation.
        self.padded_dim = next_power_of_2(dim)
        self.signs1 = generate_wht_signs(self.padded_dim, seed=seed, device=dev)
        self.signs2 = generate_wht_signs(self.padded_dim, seed=seed + 1, device=dev)

        if rotary_dim is not None and 0 < rotary_dim < dim:
            assert rotary_dim & (rotary_dim - 1) == 0, (
                f"rotary_dim {rotary_dim} must be a power of two"
            )
            assert self.padded_dim % rotary_dim == 0, (
                f"rotary_dim {rotary_dim} must divide padded_dim "
                f"{self.padded_dim}"
            )
            self.rotary_dim = rotary_dim
        else:
            self.rotary_dim = None

        # Reuse the Lloyd-Max codebook from the KV-cache TurboQuant module
        # (added in #38479). It's scipy-free (trapezoid integration) and
        # cached via lru_cache on (dim, bits).
        self.centroids = get_centroids(dim, bits).to(dev)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2

    def _apply_wht(self, padded: torch.Tensor) -> torch.Tensor:
        if self.rotary_dim is None:
            return _fast_wht_batch(padded)
        return _fast_wht_batch_blocked(padded, block_size=self.rotary_dim)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.padded_dim > self.dim:
            padded = torch.zeros(batch, self.padded_dim, device=x.device, dtype=x.dtype)
            padded[:, : self.dim] = x
        else:
            padded = x.clone()
        padded *= self.signs1.unsqueeze(0)
        padded = self._apply_wht(padded)
        padded *= self.signs2.unsqueeze(0)
        return padded[:, : self.dim]

    def _rotate_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation. NOTE: mutates `y` in place — pass a
        fresh tensor (centroid lookups already allocate new storage)."""
        batch = y.shape[0]
        if self.padded_dim > self.dim:
            padded = torch.zeros(batch, self.padded_dim, device=y.device, dtype=y.dtype)
            padded[:, : self.dim] = y
        else:
            padded = y
        padded *= self.signs2.unsqueeze(0)
        padded = self._apply_wht(padded)
        padded *= self.signs1.unsqueeze(0)
        return padded[:, : self.dim]

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize grouped vectors. x: (n_groups, group_size). Returns (indices, norms)."""
        x = x.to(device=self.device, dtype=torch.float32)
        norms = torch.linalg.norm(x, dim=1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        x_unit = x / safe_norms.unsqueeze(1)
        y = self._rotate(x_unit)
        indices = torch.searchsorted(self.boundaries, y.contiguous())
        # Shape-gain decomposition (classical VQ technique, Gray 1984):
        # store original_norm / reconstruction_norm instead of raw L2 norm.
        # Accounts for the reconstruction-norm shrinkage from the quantization
        # step itself; ~2x lower 3-bit reconstruction error vs raw L2.
        y_hat = self.centroids[indices]
        x_hat_unit = self._rotate_inverse(y_hat)
        recon_norm = torch.linalg.norm(x_hat_unit, dim=1)
        safe_recon = torch.where(recon_norm > 0, recon_norm, torch.ones_like(recon_norm))
        norms = norms / safe_recon
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize. indices: (n_groups, group_size). Returns (n_groups, group_size)."""
        if indices.device != self.device:
            indices = indices.to(device=self.device)
        # `norms` is already fp32 from quantize; avoid a redundant alloc.
        if norms.dtype != torch.float32 or norms.device != self.device:
            norms = norms.to(device=self.device, dtype=torch.float32)
        y_hat = self.centroids[indices]
        x_hat_unit = self._rotate_inverse(y_hat)
        x_hat_unit.mul_(norms.unsqueeze(1))
        return x_hat_unit


# ---------------------------------------------------------------------------
# Bit packing: pack/unpack quantization indices into uint8
# ---------------------------------------------------------------------------


def _padded_size(dim: int, group_size: int) -> tuple[int, int]:
    """Return (padded_dim, n_groups) for group quantization."""
    padded = round_up(dim, group_size)
    return padded, padded // group_size


def _compress_2d(
    flat: torch.Tensor,
    bits: int,
    group_size: int,
    rotary_dim: "int | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Compress a 2D (rows, in_dim) tensor via PolarQuant + bit-packing.

    Shared between the Linear and MoE paths. ``rotary_dim`` is forwarded to
    ``_get_quantizer`` to select full-width vs block-diagonal WHT; pass
    ``None`` (default) for everything except attention q/k/qkv projections
    on partial-rotary models. Returns (packed, norms_per_row, padded_in,
    n_groups).
    """
    rows, in_dim = flat.shape
    padded_in, n_groups = _padded_size(in_dim, group_size)
    if padded_in > in_dim:
        flat = torch.nn.functional.pad(flat, (0, padded_in - in_dim))

    grouped = flat.reshape(-1, group_size)
    quantizer = _get_quantizer(
        group_size, bits, str(flat.device), rotary_dim=rotary_dim,
    )
    indices, norms_raw = quantizer.quantize(grouped)
    packed = _pack_indices(indices, bits)
    norms = norms_raw.reshape(rows, n_groups)
    return packed, norms, padded_in, n_groups


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack quantization indices into uint8."""
    if bits == 4:
        flat = indices.reshape(-1, indices.shape[-1])
        lo = flat[:, 0::2].to(torch.uint8)
        hi = flat[:, 1::2].to(torch.uint8)
        return (lo | (hi << 4)).reshape(indices.shape[0], -1)
    elif bits == 3:
        n_rows, n_cols = indices.shape[0], indices.shape[-1]
        flat = indices.reshape(n_rows, -1).to(torch.uint8)
        pad = (8 - n_cols % 8) % 8
        if pad > 0:
            flat = torch.nn.functional.pad(flat, (0, pad))
        n_packed_cols = flat.shape[1] // 8 * 3
        packed = torch.zeros(n_rows, n_packed_cols, dtype=torch.uint8, device=indices.device)
        for i in range(flat.shape[1] // 8):
            v = flat[:, i * 8 : (i + 1) * 8]
            packed[:, i * 3] = v[:, 0] | (v[:, 1] << 3) | ((v[:, 2] & 0x3) << 6)
            packed[:, i * 3 + 1] = (v[:, 2] >> 2) | (v[:, 3] << 1) | (v[:, 4] << 4) | ((v[:, 5] & 0x1) << 7)
            packed[:, i * 3 + 2] = (v[:, 5] >> 1) | (v[:, 6] << 2) | (v[:, 7] << 5)
        return packed
    elif bits == 2:
        flat = indices.reshape(-1, indices.shape[-1])
        shifts = torch.tensor([0, 2, 4, 6], device=indices.device, dtype=torch.uint8)
        parts = torch.stack([flat[:, i::4].to(torch.uint8) for i in range(4)], dim=-1)
        return (parts << shifts).sum(dim=-1).to(torch.uint8).reshape(indices.shape[0], -1)
    return indices.to(torch.uint8)


def _unpack_indices(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Unpack uint8 packed indices back to int32.

    int32 (vs int64) halves the hot-path memory spike in MoE decompress.
    Downstream consumers (``centroids[indices]``, Triton kernels) all
    accept any integer dtype.
    """
    if bits == 4:
        flat = packed.reshape(-1, packed.shape[-1])
        n_rows, n_cols = flat.shape
        unpacked = torch.empty(n_rows, n_cols * 2, dtype=torch.int32, device=packed.device)
        unpacked[:, 0::2] = (flat & 0x0F).to(torch.int32)
        unpacked[:, 1::2] = ((flat >> 4) & 0x0F).to(torch.int32)
        return unpacked[:, :dim]
    elif bits == 3:
        flat = packed.reshape(-1, packed.shape[-1])
        n_rows, n_cols = flat.shape
        n_groups_of_3 = n_cols // 3
        g3 = flat[:, : n_groups_of_3 * 3].reshape(n_rows, n_groups_of_3, 3).to(torch.int32)
        b0, b1, b2 = g3[..., 0], g3[..., 1], g3[..., 2]
        out = torch.empty(n_rows, n_groups_of_3, 8, dtype=torch.int32, device=packed.device)
        out[..., 0] = b0 & 0x7
        out[..., 1] = (b0 >> 3) & 0x7
        out[..., 2] = ((b0 >> 6) | (b1 << 2)) & 0x7
        out[..., 3] = (b1 >> 1) & 0x7
        out[..., 4] = (b1 >> 4) & 0x7
        out[..., 5] = ((b1 >> 7) | (b2 << 1)) & 0x7
        out[..., 6] = (b2 >> 2) & 0x7
        out[..., 7] = (b2 >> 5) & 0x7
        return out.reshape(n_rows, n_groups_of_3 * 8)[:, :dim]
    elif bits == 2:
        flat = packed.reshape(-1, packed.shape[-1])
        n_rows, n_cols = flat.shape
        unpacked = torch.empty(n_rows, n_cols * 4, dtype=torch.int32, device=packed.device)
        for i in range(4):
            unpacked[:, i::4] = ((flat >> (i * 2)) & 0x03).to(torch.int32)
        return unpacked[:, :dim]
    return packed.to(torch.int32)


# ---------------------------------------------------------------------------
# Triton kernels (FWHT-on-input GEMM + fused dequant-GEMM)
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# Rotation matrix cache: (id(signs1), id(signs2), group_size) → tensor
_rotation_matrix_cache: dict[tuple, torch.Tensor] = {}


def _build_rotation_matrix(
    signs1: torch.Tensor, signs2: torch.Tensor, group_size: int,
) -> torch.Tensor:
    """Pre-compute inverse rotation matrix W_rot = H @ D2 @ D1 / sqrt(n)."""
    n = group_size
    eye = torch.eye(n, device=signs1.device, dtype=torch.float32)
    rotated = eye * signs2.unsqueeze(0)
    rotated = _fast_wht_batch(rotated)
    rotated = rotated * signs1.unsqueeze(0)
    return rotated


def _get_cached_rotation_matrix(
    signs1: torch.Tensor, signs2: torch.Tensor, group_size: int,
) -> torch.Tensor:
    """Get or build cached rotation matrix."""
    key = (id(signs1), id(signs2), group_size)
    if key not in _rotation_matrix_cache:
        _rotation_matrix_cache[key] = _build_rotation_matrix(
            signs1, signs2, group_size,
        ).contiguous()
    return _rotation_matrix_cache[key]


def _rotate_input(
    x: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Apply forward rotation to input, grouped by group_size."""
    batch = x.shape[0]
    K = x.shape[1]
    padded_K = ((K + group_size - 1) // group_size) * group_size
    if padded_K > K:
        x = torch.nn.functional.pad(x, (0, padded_K - K))
    w_rot = _get_cached_rotation_matrix(signs1, signs2, group_size)
    x_grouped = x.reshape(-1, group_size)
    x_grouped = torch.matmul(x_grouped, w_rot.T)
    return x_grouped.reshape(batch, padded_K)


if _HAS_TRITON:

    @triton.jit
    def _tq_fused_gemm_kernel(
        a_ptr, stride_am, stride_ak,
        packed_ptr, norms_ptr,
        stride_packed_n, stride_packed_k, stride_norms_n, stride_norms_g,
        w_rot_ptr, centroids_ptr,
        c_ptr, stride_cm, stride_cn,
        bias_ptr,
        M, N, K, n_groups,
        GROUP_SIZE: tl.constexpr, BITS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Fused TQ dequant-GEMM: unpack + codebook + rotate + scale + accumulate.

        Note: 3-bit unpacking logic is duplicated in _polar_fused_gemm_kernel
        (Triton JIT kernels cannot share helper functions).
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, GROUP_SIZE)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        rot_offs = offs_k[:, None] * GROUP_SIZE + offs_k[None, :]
        w_rot = tl.load(w_rot_ptr + rot_offs)
        for g in range(n_groups):
            k_start = g * GROUP_SIZE
            a_offs = offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
            a_mask = (offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K)
            a_tile = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0).to(tl.float32)
            packed_row = offs_n * n_groups + g
            if BITS == 4:
                byte_idx = offs_k // 2
                is_hi = (offs_k % 2).to(tl.int32)
                pe = packed_row[:, None] * stride_packed_n + byte_idx[None, :] * stride_packed_k
                pb = tl.load(packed_ptr + pe, mask=offs_n[:, None] < N, other=0).to(tl.int32)
                indices = tl.where(is_hi[None, :] > 0, (pb >> 4) & 0xF, pb & 0xF)
            elif BITS == 3:
                # Bulk-load all 48 packed bytes per row in one coalesced
                # transaction (padded to 64 = pow-of-2 for Triton tile
                # constraint, with the tail 16 masked out). Then gather the
                # per-k bytes from the in-register buffer. Replaces two
                # uncoalesced per-thread scatter loads whose stride pattern
                # (bi0[k]=[0,0,0,1,1,1,2,2,...]) could not be vectorized
                # by Triton and cost ~50x at bs=1 on real models.
                g8 = offs_k // 8
                p8 = offs_k % 8
                bo = p8 * 3
                fb = bo // 8
                bib = (bo % 8).to(tl.int32)
                crosses = bib > 5
                bi0 = g8 * 3 + fb
                bi1 = tl.minimum(bi0 + 1, 47)
                byte_offs = tl.arange(0, 64)
                valid_byte = byte_offs < 48
                bulk_p = packed_row[:, None] * stride_packed_n + byte_offs[None, :] * stride_packed_k
                bulk = tl.load(
                    packed_ptr + bulk_p,
                    mask=(offs_n[:, None] < N) & valid_byte[None, :],
                    other=0,
                ).to(tl.int32)
                bi0_bc = tl.broadcast_to(bi0[None, :], (BLOCK_N, GROUP_SIZE))
                bi1_bc = tl.broadcast_to(bi1[None, :], (BLOCK_N, GROUP_SIZE))
                b0 = tl.gather(bulk, bi0_bc, axis=1)
                b1 = tl.gather(bulk, bi1_bc, axis=1)
                single = (b0 >> bib[None, :]) & 0x7
                cross = ((b0 >> bib[None, :]) | (b1 << (8 - bib[None, :]))) & 0x7
                indices = tl.where(crosses[None, :], cross, single)
            elif BITS == 2:
                byte_idx = offs_k // 4
                shift = (offs_k % 4).to(tl.int32) * 2
                pe = packed_row[:, None] * stride_packed_n + byte_idx[None, :] * stride_packed_k
                pb = tl.load(packed_ptr + pe, mask=offs_n[:, None] < N, other=0).to(tl.int32)
                indices = (pb >> shift[None, :]) & 0x3
            else:
                pe = packed_row[:, None] * stride_packed_n + offs_k[None, :] * stride_packed_k
                indices = tl.load(packed_ptr + pe, mask=offs_n[:, None] < N, other=0).to(tl.int32)
            centroid_vec = tl.load(centroids_ptr + indices)
            w_deq = tl.dot(centroid_vec, w_rot)
            norm_offs = offs_n * stride_norms_n + g * stride_norms_g
            norms = tl.load(norms_ptr + norm_offs, mask=offs_n < N, other=0.0)
            w_deq = w_deq * norms[:, None]
            # Cast to output dtype for tensor core GEMM (bf16/fp16 ~2x faster)
            out_dt = c_ptr.type.element_ty
            acc += tl.dot(a_tile.to(out_dt), tl.trans(w_deq).to(out_dt))
        if bias_ptr:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias[None, :]
        c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptr + c_offs, acc.to(c_ptr.type.element_ty), mask=c_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        ],
        key=["out_f", "in_f_padded"],
    )
    @triton.jit
    def _polar_fused_gemm_kernel(
        x_rot_ptr, stride_xm, stride_xk,
        codes_ptr, stride_cn, stride_ck,
        norms_ptr, stride_nn, stride_ng,
        ct_ptr,
        out_ptr, stride_om, stride_on,
        bias_ptr,
        batch_size, out_f, in_f_padded, n_groups,
        BLOCK_K: tl.constexpr, BITS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """FWHT-on-input: codebook dot product with pre-rotated input.

        Note: 3-bit unpacking logic is duplicated in _tq_fused_gemm_kernel
        (Triton JIT kernels cannot share helper functions).
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < batch_size
        mask_n = offs_n < out_f
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(n_groups):
            offs_k = tl.arange(0, BLOCK_K)
            x_ptrs = offs_m[:, None] * stride_xm + (g * BLOCK_K + offs_k)[None, :] * stride_xk
            x_tile = tl.load(x_rot_ptr + x_ptrs, mask=mask_m[:, None], other=0.0)
            packed_row = offs_n * n_groups + g
            if BITS == 4:
                bi = offs_k // 2
                ih = offs_k % 2
                cp = packed_row[:, None] * stride_cn + bi[None, :] * stride_ck
                pk = tl.load(codes_ptr + cp, mask=mask_n[:, None], other=0).to(tl.int32)
                codes = tl.where(ih[None, :] > 0, (pk >> 4) & 0xF, pk & 0xF)
            elif BITS == 3:
                # Bulk-load all 48 packed bytes per row in one coalesced
                # transaction, then gather per-k bytes from the in-register
                # buffer. See identical fix in _tq_fused_gemm_kernel.
                g8 = offs_k // 8
                p8 = offs_k % 8
                bo = p8 * 3
                fb = bo // 8
                bib = (bo % 8).to(tl.int32)
                crosses = bib > 5
                bi0 = g8 * 3 + fb
                bi1 = tl.minimum(bi0 + 1, 47)
                byte_offs = tl.arange(0, 64)
                valid_byte = byte_offs < 48
                bulk_p = packed_row[:, None] * stride_cn + byte_offs[None, :] * stride_ck
                bulk = tl.load(
                    codes_ptr + bulk_p,
                    mask=mask_n[:, None] & valid_byte[None, :],
                    other=0,
                ).to(tl.int32)
                bi0_bc = tl.broadcast_to(bi0[None, :], (BLOCK_N, BLOCK_K))
                bi1_bc = tl.broadcast_to(bi1[None, :], (BLOCK_N, BLOCK_K))
                b0 = tl.gather(bulk, bi0_bc, axis=1)
                b1 = tl.gather(bulk, bi1_bc, axis=1)
                single = (b0 >> bib[None, :]) & 0x7
                cross = ((b0 >> bib[None, :]) | (b1 << (8 - bib[None, :]))) & 0x7
                codes = tl.where(crosses[None, :], cross, single)
            elif BITS == 2:
                bi = offs_k // 4
                sh = (offs_k % 4).to(tl.int32) * 2
                cp = packed_row[:, None] * stride_cn + bi[None, :] * stride_ck
                pk = tl.load(codes_ptr + cp, mask=mask_n[:, None], other=0).to(tl.int32)
                codes = (pk >> sh[None, :]) & 0x3
            else:
                cp = packed_row[:, None] * stride_cn + offs_k[None, :] * stride_ck
                codes = tl.load(codes_ptr + cp, mask=mask_n[:, None], other=0).to(tl.int32)
            values = tl.load(ct_ptr + codes)
            norm_ptrs = offs_n * stride_nn + g * stride_ng
            norms = tl.load(norms_ptr + norm_ptrs, mask=mask_n, other=0.0)
            values = values * norms[:, None]
            out_dt = out_ptr.type.element_ty
            acc += tl.dot(x_tile.to(out_dt), tl.trans(values).to(out_dt))
        if bias_ptr:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
            acc += bias[None, :]
        out_ptrs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(out_ptr + out_ptrs, acc.to(out_ptr.type.element_ty), mask=out_mask)


def _tq_fused_gemm_launcher(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    norms: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    centroids: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused TQ dequant + GEMM launcher."""
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    M, K = x.shape
    N = norms.shape[0]
    if M == 0:
        return torch.empty((*orig_shape[:-1], N), dtype=x.dtype, device=x.device)
    n_groups = norms.shape[1]
    if K % group_size != 0 or K // group_size != n_groups:
        raise ValueError(f"K={K} not aligned with group_size={group_size}")
    w_rot = _get_cached_rotation_matrix(signs1, signs2, group_size)
    output = torch.empty(M, N, dtype=x.dtype, device=x.device)
    # Rotation matrix uses GROUP_SIZE^2 * 4 bytes of shared memory;
    # cap block sizes to stay within hardware limits (~100 KB on Ada).
    max_block = 16 if group_size >= 128 else 32
    BLOCK_M = min(max_block, triton.next_power_of_2(M))
    BLOCK_N = min(max_block, triton.next_power_of_2(N))
    if not packed_weight.is_contiguous():
        packed_weight = packed_weight.contiguous()
    if not norms.is_contiguous():
        norms = norms.contiguous()
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _tq_fused_gemm_kernel[grid](
        x, x.stride(0), x.stride(1),
        packed_weight, norms,
        packed_weight.stride(0), packed_weight.stride(1),
        norms.stride(0), norms.stride(1),
        w_rot, centroids,
        output, output.stride(0), output.stride(1),
        bias, M, N, K, n_groups,
        GROUP_SIZE=group_size, BITS=bits,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    if len(orig_shape) > 2:
        output = output.reshape(*orig_shape[:-1], N)
    return output


def _tq_fwht_input_gemm_launcher(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    norms: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    centroids: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """FWHT-on-input GEMM launcher. Rotates input once, then codebook dot."""
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    M, K = x.shape
    N = norms.shape[0]
    if M == 0:
        return torch.empty((*orig_shape[:-1], N), dtype=x.dtype, device=x.device)
    n_groups = norms.shape[1]
    padded_K = n_groups * group_size
    if K != padded_K:
        raise ValueError(f"K={K} != padded_K={padded_K}")
    x_rot = _rotate_input(x.float(), signs1, signs2, group_size)
    output = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BLOCK_K = group_size
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    _polar_fused_gemm_kernel[grid](
        x_rot, x_rot.stride(0), x_rot.stride(1),
        packed_weight, packed_weight.stride(0), packed_weight.stride(1),
        norms, norms.stride(0), norms.stride(1),
        centroids,
        output, output.stride(0), output.stride(1),
        bias, M, N, padded_K, n_groups,
        BLOCK_K=BLOCK_K, BITS=bits,
    )
    if len(orig_shape) > 2:
        output = output.reshape(*orig_shape[:-1], N)
    return output


# Register as torch.library.custom_op for fullgraph compatibility.
# Dynamo treats custom ops as opaque (no tracing into kernel body).
try:

    @torch.library.custom_op(
        "turboquant::tq_fused_gemm", mutates_args=(), device_types=("cuda",),
    )
    def _tq_fused_gemm_op(
        x: torch.Tensor, packed_weight: torch.Tensor, norms: torch.Tensor,
        signs1: torch.Tensor, signs2: torch.Tensor, centroids: torch.Tensor,
        group_size: int, bits: int, bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return _tq_fused_gemm_launcher(
            x, packed_weight, norms, signs1, signs2, centroids,
            group_size=group_size, bits=bits, bias=bias,
        )

    @_tq_fused_gemm_op.register_fake
    def _(x, packed_weight, norms, signs1, signs2, centroids, group_size, bits, bias):
        N = norms.shape[0]
        return x.new_empty((*x.shape[:-1], N))

    @torch.library.custom_op(
        "turboquant::tq_fwht_input_gemm", mutates_args=(), device_types=("cuda",),
    )
    def _tq_fwht_input_gemm_op(
        x: torch.Tensor, packed_weight: torch.Tensor, norms: torch.Tensor,
        signs1: torch.Tensor, signs2: torch.Tensor, centroids: torch.Tensor,
        group_size: int, bits: int, bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return _tq_fwht_input_gemm_launcher(
            x, packed_weight, norms, signs1, signs2, centroids,
            group_size=group_size, bits=bits, bias=bias,
        )

    @_tq_fwht_input_gemm_op.register_fake
    def _(x, packed_weight, norms, signs1, signs2, centroids, group_size, bits, bias):
        N = norms.shape[0]
        return x.new_empty((*x.shape[:-1], N))

    def tq_fused_gemm(x, packed_weight, norms, signs1, signs2, centroids,
                      group_size=128, bits=4, bias=None):
        return torch.ops.turboquant.tq_fused_gemm(
            x, packed_weight, norms, signs1, signs2, centroids,
            group_size, bits, bias,
        )

    def tq_fwht_input_gemm(x, packed_weight, norms, signs1, signs2, centroids,
                           group_size=128, bits=4, bias=None):
        return torch.ops.turboquant.tq_fwht_input_gemm(
            x, packed_weight, norms, signs1, signs2, centroids,
            group_size, bits, bias,
        )
except (AttributeError, RuntimeError, NameError):
    # Triton not available — tq_fused_gemm / tq_fwht_input_gemm undefined
    tq_fused_gemm = None  # type: ignore[assignment]
    tq_fwht_input_gemm = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# TurboQuantOnlineLinearMethod
# ---------------------------------------------------------------------------


class TurboQuantOnlineLinearMethod(LinearMethodBase):
    """Online TQ3/TQ4 weight compression for Linear layers.

    Allocates bf16 weight on meta device (zero GPU at init). After
    weight loading materializes the bf16 on GPU, compresses to TQ
    packed format. Forward pass uses Triton dequant-GEMM kernels.
    """

    uses_meta_device: bool = True

    def __init__(
        self,
        bits: int = 3,
        group_size: int = 128,
        rotary_dim: "int | None" = None,
    ):
        if bits not in (2, 3, 4):
            raise ValueError(f"turboquant bits must be 2, 3, or 4; got {bits}")
        if group_size <= 0 or group_size % 8 != 0:
            raise ValueError(
                "turboquant group_size must be a positive multiple of 8; "
                f"got {group_size}"
            )
        self.bits = bits
        self.group_size = group_size
        self.rotary_dim = rotary_dim

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        weight = layer.weight.data
        bits = self.bits
        group_size = self.group_size

        out_dim, in_dim = weight.shape
        packed, norms, padded_in, n_groups = _compress_2d(
            weight, bits, group_size, rotary_dim=self.rotary_dim,
        )
        quantizer = _get_quantizer(
            group_size, bits, str(weight.device), rotary_dim=self.rotary_dim,
        )

        # Keep weight attr for vLLM's MLA post-processing (expects it to exist)
        layer.weight.data = torch.empty(0, device=weight.device, dtype=weight.dtype)
        layer.register_buffer("tq_packed_weight", packed)
        layer.register_buffer("tq_norms", norms)
        layer.register_buffer("tq_signs1", quantizer.signs1)
        layer.register_buffer("tq_signs2", quantizer.signs2)
        layer.register_buffer("tq_centroids", quantizer.centroids)
        layer.tq_in_features = in_dim
        layer.tq_out_features = out_dim
        layer.tq_padded_in = padded_in

        # Triton kernels only implement full-width WHT. When we compressed
        # with a block-diagonal WHT (partial-rotary models), the kernel's
        # forward WHT wouldn't match and would produce garbage; fall back
        # to the PyTorch path which honours ``self.rotary_dim``.
        if tq_fused_gemm is not None and self.rotary_dim is None:
            layer._tq_primary_fn = tq_fwht_input_gemm if out_dim >= 4096 else tq_fused_gemm
            layer._tq_fallback_fn = tq_fused_gemm if out_dim >= 4096 else tq_fwht_input_gemm
        else:
            layer._tq_primary_fn = None

        layer._already_called_process_weights_after_loading = True
        del weight

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pad input if in_dim was not a multiple of group_size
        if x.shape[-1] != layer.tq_padded_in:
            x = torch.nn.functional.pad(x, (0, layer.tq_padded_in - x.shape[-1]))

        if layer._tq_primary_fn is not None:
            args = (
                x,
                layer.tq_packed_weight,
                layer.tq_norms,
                layer.tq_signs1,
                layer.tq_signs2,
                layer.tq_centroids,
            )
            try:
                return layer._tq_primary_fn(*args, group_size=self.group_size, bits=self.bits, bias=bias)
            except (ValueError, RuntimeError) as e:
                logger.warning_once("TurboQuant primary kernel failed, using fallback: %s", e)
                return layer._tq_fallback_fn(*args, group_size=self.group_size, bits=self.bits, bias=bias)

        # PyTorch fallback (no Triton)
        indices = _unpack_indices(layer.tq_packed_weight, self.bits, self.group_size)
        norms_flat = layer.tq_norms.reshape(-1)
        quantizer = _get_quantizer(
            self.group_size, self.bits, str(x.device),
            rotary_dim=self.rotary_dim,
        )
        w_groups = quantizer.dequantize(indices, norms_flat)
        w_deq = w_groups.reshape(layer.tq_out_features, layer.tq_padded_in).to(x.dtype)
        output = torch.matmul(x, w_deq.t())
        if bias is not None:
            output = output + bias
        return output
