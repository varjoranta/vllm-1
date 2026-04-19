# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant MoE expert weight compression.

Follow-up to PR #39970 which covers Linear-only. This module extends
``--quantization turboquant`` to FusedMoE layers.

Strategy (borrowed from the turboquant-plus-vllm plugin, proven in
production on GLM-5.1 754B and Gemma 4 26B): compress each expert's
w13 and w2 via the same PolarQuant pipeline used for Linear layers,
share a single bf16 scratch pool across all MoE layers (only one runs
at a time during forward), decompress into the pool per-forward and
delegate to the existing ``UnquantizedFusedMoEMethod`` kernel.

Trade-offs vs a fused-dequant MoE kernel:

  - PRO: reuses vLLM's battle-tested unquantized MoE kernel; no new
    kernel surface to review
  - PRO: decompression math matches the Linear path exactly
  - CON: extra HBM round-trip per forward; dequant launched 2x per
    MoE layer per forward

Phase 2 (separate PR): replace the Python-level dequant with a fused
3D CUDA kernel. Groundwork exists in the turboquant-plus-vllm plugin's
``csrc/tq_weight_dequant.cu``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.online.turboquant import (
    _compress_2d,
    _get_quantizer,
    _unpack_indices,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import FusedMoE

logger = init_logger(__name__)


class _Compressed3D:
    """Packed form of a 3D expert weight tensor (n_experts, out, in).

    Flattens experts into rows, runs the same 2D compression pipeline
    the Linear path uses, then remembers the original shape for
    decompression. Typical 3-bit compression: ~4.3x over fp16.
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

        flat = data.reshape(-1, in_dim)
        self.packed, self.norms, self.padded_in, self.n_groups = _compress_2d(
            flat, bits, group_size,
        )

    def decompress_into(self, out: torch.Tensor) -> None:
        """Write decompressed bf16/fp16 weights into a pre-allocated buffer.

        Phase 1 uses the Python-level dequant path — correct but ~10x
        slower than the fused CUDA kernel in the turboquant-plus-vllm
        plugin's csrc/. A Phase 2 PR replaces this with a direct CUDA
        dequant matching the Linear path's speed.
        """
        assert out.shape == self.shape, (
            f"decompress_into expected shape {self.shape}, got {tuple(out.shape)}"
        )
        assert out.dtype == self.dtype, (
            f"decompress_into expected dtype {self.dtype}, got {out.dtype}"
        )

        n_experts, out_dim, in_dim = self.shape
        quantizer = _get_quantizer(self.group_size, self.bits, str(out.device))

        indices = _unpack_indices(self.packed, self.bits, self.padded_in)
        grouped_indices = indices.reshape(-1, self.group_size)
        grouped_norms = self.norms.reshape(-1)
        dequant = quantizer.dequantize(grouped_indices, grouped_norms)
        dequant = dequant.reshape(n_experts, out_dim, self.padded_in)
        if self.padded_in > in_dim:
            dequant = dequant[:, :, :in_dim]
        out.copy_(dequant.to(out.dtype))


class _MoEScratchPool:
    """One bf16 scratch buffer pair shared across all MoE layers.

    Only one MoE layer runs at a time during forward, so a single set
    of (w13, w2) bf16 tensors is enough. Per-layer scratch would
    multiply memory by the layer count and defeat the compression.
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


# Module-level singleton. TODO(phase-2): move to per-model-config scope
# so multi-model-per-process cases (speculative draft models, LoRA
# routers) don't share a pool sized for the first-loaded model.
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

        # Delegation over inheritance for apply(): hold an
        # UnquantizedFusedMoEMethod and call its apply() after we've
        # populated the scratch pool. We still inherit OnlineMoEMethodBase
        # for its create_weights + initialize_online_processing setup,
        # but override apply() to insert the dequant step.
        from vllm.model_executor.layers.fused_moe import (
            UnquantizedFusedMoEMethod,
        )
        self._base_method = UnquantizedFusedMoEMethod(layer.moe_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        global _MOE_SCRATCH_POOL
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data

        w13_c = _Compressed3D(w13, self.bits, self.group_size)
        w2_c = _Compressed3D(w2, self.bits, self.group_size)

        if _MOE_SCRATCH_POOL is None:
            _MOE_SCRATCH_POOL = _MoEScratchPool(
                w13_shape=w13_c.shape, w2_shape=w2_c.shape,
                dtype=w13.dtype, device=w13.device,
            )
        else:
            _MOE_SCRATCH_POOL.assert_matches(w13_c.shape, w2_c.shape)

        # Re-point layer's weight parameters at the pool buffers so the
        # unquantized kernel (which reads layer.w13_weight / w2_weight)
        # sees freshly-dequantized values on each forward.
        layer.w13_weight.data = _MOE_SCRATCH_POOL.w13
        layer.w2_weight.data = _MOE_SCRATCH_POOL.w2

        layer._tq_w13_compressed = w13_c
        layer._tq_w2_compressed = w2_c
        layer._tq_scratch_pool = _MOE_SCRATCH_POOL

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: "FusedMoE",  # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
