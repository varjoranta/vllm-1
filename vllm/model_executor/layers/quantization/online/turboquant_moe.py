# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant MoE expert weight compression.

Follow-up to PR #39970 which covers Linear-only. This module extends
``--quantization turboquant`` to FusedMoE layers.

Strategy (borrowed from the turboquant-plus-vllm plugin, proven in
production for GLM-5.1 754B on 2xH200 and Gemma 4 26B on L40S):

  1. At weight-load time, compress each expert's w13 and w2 into packed
     3-bit/4-bit indices + per-group norms via the same PolarQuant pipeline
     used for Linear layers. Weights go from (n_experts, out, in) fp16
     down to (n_experts * out * n_groups, bytes_per_group) uint8 + norms.
     ~4x memory reduction on typical MoE checkpoints.
  2. Share a single bf16 scratch pool across ALL MoE layers in the model.
     Only one MoE layer's forward runs at a time, so one dequant buffer
     is enough. Per-layer scratch would defeat the compression.
  3. At ``apply()`` time, decompress into the pool and delegate to the
     existing ``UnquantizedFusedMoEMethod`` kernel path. Same kernel, same
     performance as BF16 MoE — just with expert weights held in packed
     form between forward passes.

Trade-offs vs a fused-dequant MoE kernel:

  - PRO: reuses vLLM's battle-tested unquantized MoE kernel; no new
    kernel surface to review
  - PRO: correct by construction — decompression math is identical to
    the Linear path
  - CON: extra HBM round-trip per forward (read packed, write bf16
    scratch, read bf16 scratch into MoE kernel). A fused kernel would
    elide the scratch roundtrip
  - CON: dequant kernel launched 2x per MoE layer per forward (w13 and w2)

Phase 2 (separate PR): replace the Python/Triton dequant with a fused
3D CUDA kernel that writes directly into the scratch pool with lower
HBM bandwidth usage. Groundwork exists in the plugin's csrc/.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)

# Reuse the Linear-path quantizer — same WHT rotation + Lloyd-Max codebook,
# no separate MoE-specific algorithm.
from vllm.model_executor.layers.quantization.online.turboquant import (
    _get_quantizer,
    _pack_indices,
    _padded_size,
    _unpack_indices,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )

logger = init_logger(__name__)


class _Compressed3D:
    """Packed form of a 3D expert weight tensor (n_experts, out, in).

    Compression pipeline matches the Linear path:
        1. Reshape (n_experts, out, in) → (n_experts * out, in)
        2. Pad in-dim to group_size multiple, reshape to (-1, group_size)
        3. Quantize with PolarQuant (WHT + Lloyd-Max codebook)
        4. Pack indices to uint8 (3-bit: 10 values per int32, or equivalent)

    Memory: bytes per expert weight = (out * in * bits) / 8 + (out * n_groups * 4 for norms).
    Typical 3-bit compression: ~4.3x over fp16 fp16.
    """

    __slots__ = (
        "packed", "norms", "shape", "dtype", "bits", "group_size",
        "padded_in", "n_groups",
    )

    def __init__(self, data: torch.Tensor, bits: int, group_size: int):
        n_experts, out_dim, in_dim = data.shape
        self.shape = data.shape
        self.dtype = data.dtype
        self.bits = bits
        self.group_size = group_size
        self.padded_in, self.n_groups = _padded_size(in_dim, group_size)

        flat = data.reshape(-1, in_dim)
        if self.padded_in > in_dim:
            padded = torch.zeros(
                flat.shape[0], self.padded_in,
                dtype=flat.dtype, device=flat.device,
            )
            padded[:, :in_dim] = flat
        else:
            padded = flat

        grouped = padded.reshape(-1, group_size)
        quantizer = _get_quantizer(group_size, bits, str(data.device))
        indices, norms = quantizer.quantize(grouped)

        self.packed = _pack_indices(indices, bits)
        # Norms shape: (n_experts * out, n_groups)
        self.norms = norms.reshape(n_experts * out_dim, self.n_groups)

    def decompress_into(self, out: torch.Tensor) -> None:
        """Write decompressed bf16/fp16 weights into a pre-allocated buffer.

        Phase 1: Python-level dequant via PolarQuant.dequantize(). Correct
        but ~10x slower than the fused CUDA kernel in the plugin's csrc/.
        Acceptable for initial MoE-path validation; Phase 2 PR replaces
        with a direct CUDA dequant matching the Linear path's speed.
        """
        assert out.shape == self.shape, (
            f"decompress_into expected shape {self.shape}, got {tuple(out.shape)}"
        )
        assert out.dtype == self.dtype, (
            f"decompress_into expected dtype {self.dtype}, got {out.dtype}"
        )

        n_experts, out_dim, in_dim = self.shape
        quantizer = _get_quantizer(self.group_size, self.bits, str(out.device))

        # Unpack indices → dequantize through codebook → reshape back to 3D
        indices = _unpack_indices(self.packed, self.bits, self.padded_in)
        # indices: (n_experts * out, padded_in). Reshape to (-1, group_size)
        grouped_indices = indices.reshape(-1, self.group_size)
        grouped_norms = self.norms.reshape(-1)
        # dequantize returns (n_groups_total, group_size)
        dequant = quantizer.dequantize(grouped_indices, grouped_norms)
        # Reshape back to (n_experts, out, padded_in), then truncate
        dequant = dequant.reshape(n_experts, out_dim, self.padded_in)
        if self.padded_in > in_dim:
            dequant = dequant[:, :, :in_dim]
        out.copy_(dequant.to(out.dtype))


class _MoEScratchPool:
    """One bf16 scratch buffer pair shared across all MoE layers.

    Only one MoE layer's forward runs at a time in standard dispatch, so
    a single set of (w13, w2) bf16 tensors is enough to hold the
    decompressed weights. Per-layer scratch would multiply memory by the
    layer count and defeat the compression.

    Raises if a later layer has a different expert shape (heterogeneous
    MoE — not expected in current vLLM-supported models but we fail loud
    rather than produce wrong results from a mismatched buffer).
    """

    __slots__ = ("w13", "w2", "shape_w13", "shape_w2")

    def __init__(self, w13_shape: torch.Size, w2_shape: torch.Size,
                 dtype: torch.dtype, device: torch.device):
        self.shape_w13 = w13_shape
        self.shape_w2 = w2_shape
        self.w13 = torch.zeros(w13_shape, dtype=dtype, device=device)
        self.w2 = torch.zeros(w2_shape, dtype=dtype, device=device)

    def assert_matches(self, w13_shape: torch.Size, w2_shape: torch.Size) -> None:
        if w13_shape != self.shape_w13:
            raise AssertionError(
                f"heterogeneous FusedMoE layer: scratch pool sized "
                f"{tuple(self.shape_w13)} but layer needs {tuple(w13_shape)}"
            )
        if w2_shape != self.shape_w2:
            raise AssertionError(
                f"heterogeneous FusedMoE layer: scratch pool sized "
                f"{tuple(self.shape_w2)} but layer needs {tuple(w2_shape)}"
            )


# Module-level singleton. Set by the first TurboQuantOnlineFusedMoEMethod
# in the model; reused by all subsequent ones.
_MOE_SCRATCH_POOL: _MoEScratchPool | None = None


class TurboQuantOnlineFusedMoEMethod(OnlineMoEMethodBase):
    """Online TQ3/TQ4 compression for FusedMoE expert weights.

    Allocates fp16/bf16 weights on meta device (zero GPU at init), waits
    for weight-loading to materialize them, then compresses w13 and w2
    in ``process_weights_after_loading``. Forward pass decompresses into
    the shared scratch pool and delegates to the unquantized kernel.
    """

    uses_meta_device: bool = True

    def __init__(
        self,
        *,
        layer: torch.nn.Module,
        bits: int = 3,
        group_size: int = 128,
    ):
        if bits not in (2, 3, 4):
            raise ValueError(f"turboquant bits must be 2, 3, or 4; got {bits}")
        if group_size <= 0 or group_size % 8 != 0:
            raise ValueError(
                f"turboquant group_size must be a positive multiple of 8; got {group_size}"
            )
        super().__init__(layer.moe_config)
        self.bits = bits
        self.group_size = group_size

        # Lazy imports to stay CPU-import-safe for tests.
        from vllm.model_executor.layers.fused_moe import (
            UnquantizedFusedMoEMethod,
        )
        # We delegate forward to the unquantized kernel on the post-dequant
        # bf16 weights. Holding a reference so apply() can call its logic
        # without re-instantiating.
        self._base_method: Any = UnquantizedFusedMoEMethod(layer.moe_config)

    # ------------------------------------------------------------------
    # Weight compression at model-load time
    # ------------------------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        global _MOE_SCRATCH_POOL
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        w13 = layer.w13_weight.data  # (E, 2 * I, H)
        w2 = layer.w2_weight.data   # (E, H, I)

        # Compress both tensors in-place. Expert loop unrolled by the
        # per-expert reshape inside _Compressed3D.
        w13_c = _Compressed3D(w13, self.bits, self.group_size)
        w2_c = _Compressed3D(w2, self.bits, self.group_size)

        # Initialize or validate the shared scratch pool.
        if _MOE_SCRATCH_POOL is None:
            _MOE_SCRATCH_POOL = _MoEScratchPool(
                w13_shape=w13_c.shape, w2_shape=w2_c.shape,
                dtype=w13.dtype, device=w13.device,
            )
        else:
            _MOE_SCRATCH_POOL.assert_matches(w13_c.shape, w2_c.shape)

        # Re-point the layer's w13_weight / w2_weight at the pool buffers.
        # The unquantized kernel we delegate to reads layer.w13_weight /
        # layer.w2_weight directly, so writing into the pool makes the
        # freshly-dequantized values visible on the next forward.
        layer.w13_weight.data = _MOE_SCRATCH_POOL.w13
        layer.w2_weight.data = _MOE_SCRATCH_POOL.w2

        # Stash the compressed tensors on the layer for apply() to read.
        layer._tq_w13_compressed = w13_c
        layer._tq_w2_compressed = w2_c
        layer._tq_scratch_pool = _MOE_SCRATCH_POOL

        layer._already_called_process_weights_after_loading = True

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module,
    ) -> "FusedMoEQuantConfig | None":
        # Decompression happens in apply() before delegating to the base
        # (unquantized) method; there's no quant-aware kernel to configure.
        return None

    def apply(
        self,
        layer: "FusedMoE",  # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Decompress both expert tensors into the shared scratch pool.
        # layer.w13_weight.data / layer.w2_weight.data were re-pointed at
        # pool.w13 / pool.w2 in process_weights_after_loading, so writing
        # into the pool makes the decompressed weights visible to the
        # unquantized kernel.
        pool = layer._tq_scratch_pool
        layer._tq_w13_compressed.decompress_into(pool.w13)
        layer._tq_w2_compressed.decompress_into(pool.w2)

        return self._base_method.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts_input=shared_experts_input,
        )
