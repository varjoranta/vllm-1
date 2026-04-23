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
delegate to the existing unquantized MoE kernel.

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
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    convert_to_unquantized_kernel_format,
    make_unquantized_moe_kernel,
    select_unquantized_moe_backend,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.online.turboquant import (
    _compress_2d,
    _get_quantizer,
    _unpack_indices,
)
from vllm.model_executor.utils import replace_parameter

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

logger = init_logger(__name__)


class _Compressed3D:
    """Packed form of a 3D expert weight tensor (n_experts, out, in).

    Flattens experts into rows, runs the same 2D compression pipeline
    the Linear path uses, then remembers the original shape for
    decompression. Typical 3-bit compression: ~4.3x over fp16.
    """

    __slots__ = (
        "packed", "norms_flat", "shape", "dtype", "bits", "group_size", "padded_in",
    )

    def __init__(self, data: torch.Tensor, bits: int, group_size: int):
        self.shape = data.shape
        self.dtype = data.dtype
        self.bits = bits
        self.group_size = group_size

        flat = data.reshape(-1, data.shape[-1])
        packed, norms, self.padded_in, _ = _compress_2d(flat, bits, group_size)
        self.packed = packed
        # Keep the flat view the hot path actually consumes, not the
        # (rows, n_groups) shape — one fewer view per forward.
        self.norms_flat = norms.reshape(-1)

    def decompress_into(self, out: torch.Tensor) -> None:
        """Write decompressed bf16/fp16 weights into a pre-allocated buffer.

        Phase 1 uses the Python-level dequant path — correct but ~10x
        slower than the fused CUDA kernel in the turboquant-plus-vllm
        plugin's csrc/. A Phase 2 PR replaces this with a direct CUDA
        dequant matching the Linear path's speed.
        """
        n_experts, out_dim, in_dim = self.shape
        quantizer = _get_quantizer(self.group_size, self.bits, str(out.device))
        indices = _unpack_indices(self.packed, self.bits, self.padded_in)
        dequant = quantizer.dequantize(
            indices.reshape(-1, self.group_size), self.norms_flat,
        )
        dequant = dequant.reshape(n_experts, out_dim, self.padded_in)
        out.copy_(dequant[:, :, :in_dim])

    def decompress_experts_into(
        self, out: torch.Tensor, active_experts: torch.Tensor,
    ) -> None:
        """Decompress only ``active_experts`` into ``out``.

        Output at listed indices is bit-identical to the full decompress;
        other slots are left untouched (the downstream ``fused_moe``
        routes via ``topk_ids`` and doesn't read them).

        Requires ``enforce_eager=True`` on vLLM in this Phase-1 form —
        the Python loop breaks CUDA graph capture. A follow-up PR will
        add a CUDA kernel variant that takes ``topk_ids`` directly.
        """
        if active_experts.numel() == 0:
            return

        n_experts, out_dim, in_dim = self.shape

        # At prefill, batch*top_k quickly covers every expert (bs=16 top-8
        # already = 128 = all of Qwen3-30B-A3B). Sparse iteration isn't a
        # win when we're touching every expert anyway; full decompress is
        # strictly faster + avoids the risk of a future CUDA sparse kernel
        # overflowing grid.x.
        if active_experts.numel() >= n_experts:
            self.decompress_into(out)
            return

        unique_experts = torch.unique(active_experts).tolist()
        n_groups = self.padded_in // self.group_size
        groups_per_expert = out_dim * n_groups
        quantizer = _get_quantizer(self.group_size, self.bits, str(out.device))

        for e in unique_experts:
            if e < 0 or e >= n_experts:
                continue
            expert_packed = self.packed[e * groups_per_expert : (e + 1) * groups_per_expert]
            expert_norms = self.norms_flat[e * groups_per_expert : (e + 1) * groups_per_expert]
            indices = _unpack_indices(expert_packed, self.bits, self.padded_in)
            dequant = quantizer.dequantize(
                indices.reshape(-1, self.group_size), expert_norms,
            )
            dequant = dequant.reshape(1, out_dim, self.padded_in)
            out[e : e + 1].copy_(dequant[:, :, :in_dim])


# Module-level singleton. Only one MoE layer runs at a time during forward,
# so one (w13, w2) scratch pair is enough across every MoE layer in the
# loaded model. Multi-model processes (speculative draft + target, LoRA
# routers) will share this pool; if they load models with mismatched MoE
# shapes, the second load raises via ``_assert_shape_matches``.
# TODO: key by VllmConfig once that has a clean lifetime handle, so
# multi-model serving gets independent pools.
_MOE_SCRATCH_POOL: "_MoEScratchPool | None" = None


class _MoEScratchPool:
    __slots__ = ("w13", "w2")

    def __init__(self, w13_shape: torch.Size, w2_shape: torch.Size,
                 dtype: torch.dtype, device: torch.device):
        self.w13 = torch.empty(w13_shape, dtype=dtype, device=device)
        self.w2 = torch.empty(w2_shape, dtype=dtype, device=device)

    def assert_matches(self, w13_shape: torch.Size, w2_shape: torch.Size) -> None:
        if self.w13.shape != w13_shape:
            raise ValueError(
                f"turboquant MoE scratch pool shape mismatch: pool is "
                f"{tuple(self.w13.shape)} but layer needs {tuple(w13_shape)}"
            )
        if self.w2.shape != w2_shape:
            raise ValueError(
                f"turboquant MoE scratch pool shape mismatch: pool is "
                f"{tuple(self.w2.shape)} but layer needs {tuple(w2_shape)}"
            )


def _get_or_create_pool(
    w13_shape: torch.Size, w2_shape: torch.Size,
    dtype: torch.dtype, device: torch.device,
) -> _MoEScratchPool:
    global _MOE_SCRATCH_POOL
    if _MOE_SCRATCH_POOL is None:
        _MOE_SCRATCH_POOL = _MoEScratchPool(w13_shape, w2_shape, dtype, device)
    else:
        _MOE_SCRATCH_POOL.assert_matches(w13_shape, w2_shape)
    return _MOE_SCRATCH_POOL


class TurboQuantOnlineFusedMoEMethod(OnlineMoEMethodBase):
    """Online TQ3/TQ4 compression for FusedMoE expert weights.

    Allocates fp16/bf16 weights on meta device (zero GPU at init), waits
    for weight-loading to materialize them, then compresses w13 and w2
    in ``process_weights_after_loading``. Forward pass decompresses into
    the shared scratch pool and delegates to the unquantized MoE kernel.
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
        # Mirror Fp8/Int8 online MoE: select backend now, build kernel in
        # process_weights_after_loading. OnlineMoEMethodBase.apply then
        # reads self.moe_kernel / layer.w13_weight / layer.w2_weight.
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(
            moe_config=self.moe,
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module,
    ) -> "FusedMoEQuantConfig":
        if self.moe.has_bias:
            return biased_moe_quant_config(layer.w13_bias, layer.w2_bias)
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data

        w13_c = _Compressed3D(w13, self.bits, self.group_size)
        w2_c = _Compressed3D(w2, self.bits, self.group_size)

        pool = _get_or_create_pool(w13.shape, w2.shape, w13.dtype, w13.device)

        # ``convert_to_unquantized_kernel_format`` calls ``.contiguous()``
        # on the plain-CUDA backend — a storage-preserving no-op on our
        # already-contiguous pool buffers. Backends that permute weights
        # (FlashInfer, AITER) would break the pool invariant; guard below.
        w13_k, w2_k = convert_to_unquantized_kernel_format(
            self.unquantized_backend, layer=layer,
            w13_weight=pool.w13, w2_weight=pool.w2,
        )
        if w13_k.data_ptr() != pool.w13.data_ptr() or w2_k.data_ptr() != pool.w2.data_ptr():
            raise ValueError(
                f"turboquant MoE does not yet support the "
                f"{self.unquantized_backend.name} backend "
                "(it permutes weight storage during setup)"
            )

        # Re-register layer weights onto the pool buffers so the
        # unquantized kernel reads from the pool on each forward.
        replace_parameter(layer, "w13_weight", w13_k)
        replace_parameter(layer, "w2_weight", w2_k)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_unquantized_moe_kernel(
            quant_config=self.moe_quant_config,
            moe_config=self.moe,
            backend=self.unquantized_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._maybe_init_expert_routing_tables(),
            shared_experts=layer.shared_experts,
        )

        layer._tq_w13_compressed = w13_c
        layer._tq_w2_compressed = w2_c
        layer._tq_scratch_pool = pool

        layer._already_called_process_weights_after_loading = True

    def _dequant_into_pool(
        self, layer: torch.nn.Module,
        active_experts: torch.Tensor | None = None,
    ) -> None:
        pool = layer._tq_scratch_pool
        if active_experts is None:
            layer._tq_w13_compressed.decompress_into(pool.w13)
            layer._tq_w2_compressed.decompress_into(pool.w2)
        else:
            layer._tq_w13_compressed.decompress_experts_into(pool.w13, active_experts)
            layer._tq_w2_compressed.decompress_experts_into(pool.w2, active_experts)

    def apply(
        self,
        layer: "FusedMoE",  # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # SPARSE DEQUANT: dequant only the experts ``topk_ids`` routes to.
        # Decompressing all N experts when the downstream ``fused_moe``
        # kernel only reads top-k is 93.75% wasted work on typical configs
        # (e.g. Qwen3-30B-A3B: 128 experts, top-8). Validated 8.4x decode
        # speedup at bs=1 on H100 in the companion plugin PR
        # (varjoranta/turboquant-vllm#33). Requires enforce_eager=True on
        # vLLM — Phase 2 will add a GPU-resident sparse kernel.
        self._dequant_into_pool(layer, active_experts=topk_ids.flatten())
        return super().apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: "FusedMoE",  # noqa: F821
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._dequant_into_pool(layer)
        return super().apply_monolithic(
            layer=layer, x=x, router_logits=router_logits,
        )
