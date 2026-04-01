# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for compressed KV cache.

Wraps FlashAttention, intercepting KV cache writes and reads to
compress/decompress using TurboQuant+ (PolarQuant + QJL).

Usage: vllm serve model --kv-cache-dtype tq4
       vllm serve model --kv-cache-dtype tq_k4v3  (asymmetric)
"""

import logging
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.turboquant.config import TurboQuantConfig
from vllm.turboquant.quantizer import TurboQuantQuantizer
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

logger = logging.getLogger(__name__)

# Per-head_dim quantizers (shared across layers)
_k_quantizers: dict[int, TurboQuantQuantizer] = {}
_v_quantizers: dict[int, TurboQuantQuantizer] = {}
_tq_config: TurboQuantConfig | None = None

# Sidecar compressed cache: layer_id -> {(block, offset, head) -> (ck, cv)}
_compressed_cache: dict[int, dict[tuple[int, int, int],
                                   tuple[tuple[torch.Tensor, torch.Tensor],
                                         tuple[torch.Tensor, torch.Tensor]]]] = {}


def _get_quantizers(head_dim: int, device: str):
    """Get or create K and V quantizers for a given head_dim."""
    global _tq_config
    if _tq_config is None:
        raise RuntimeError("TurboQuant config not initialized. "
                          "This should not happen.")

    if head_dim not in _k_quantizers:
        _k_quantizers[head_dim] = TurboQuantQuantizer.from_config(
            _tq_config, head_dim, is_key=True, device=device)
        _v_quantizers[head_dim] = TurboQuantQuantizer.from_config(
            _tq_config, head_dim, is_key=False, device=device)
        logger.info(
            "TurboQuant quantizers created: head_dim=%d K=%d-bit V=%d-bit",
            head_dim, _tq_config.k_bits, _tq_config.v_bits)

    return _k_quantizers[head_dim], _v_quantizers[head_dim]


class TurboQuantAttentionBackend(FlashAttentionBackend):
    """TurboQuant attention backend.

    Extends FlashAttention with compressed KV cache. The actual attention
    computation still uses FlashAttention kernels after decompression.
    """
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "tq3", "tq4", "tq_k4v3",
    ]

    impl_cls = None  # Set below after class definition

    @classmethod
    def get_name(cls) -> str:
        return "TURBOQUANT"


class TurboQuantAttentionImpl(FlashAttentionImpl):
    """TurboQuant attention implementation.

    Overrides do_kv_cache_update to compress K/V before storing,
    and forward to decompress before attention computation.

    Known limitations (monkey-patch era carryover):
    - Per-token Python loop in compress path (CUDA fused kernel needed)
    - Full cache decompression every forward call
    - Sidecar dict grows with sequence length
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parse TQ config from kv_cache_dtype
        global _tq_config
        if _tq_config is None:
            _tq_config = TurboQuantConfig.from_kv_cache_dtype(
                self.kv_cache_dtype)
            logger.info("TurboQuant initialized: %s", _tq_config)

        # Override kv_cache_dtype to "auto" so FlashAttention doesn't
        # try to apply FP8 quantization on top
        self.kv_cache_dtype = "auto"

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Call parent to write uncompressed K/V to paged cache
        # (for memory accounting and FlashAttention compatibility)
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

        # Now compress and store in sidecar
        layer_id = id(layer)
        if layer_id not in _compressed_cache:
            _compressed_cache[layer_id] = {}

        head_dim = key.shape[-1]
        num_kv_heads = key.shape[1] if key.dim() == 3 else 1
        device = str(key.device)
        k_quant, v_quant = _get_quantizers(head_dim, device)

        key_cache, _ = kv_cache.unbind(0)
        block_size = key_cache.shape[1]
        num_tokens = slot_mapping.shape[0]

        for t in range(num_tokens):
            slot = slot_mapping[t].item()
            block_idx = slot // block_size
            offset = slot % block_size

            for h in range(num_kv_heads):
                k_vec = key[t, h].unsqueeze(0)
                v_vec = value[t, h].unsqueeze(0)

                k_indices, k_norms = k_quant.quantize(k_vec)
                v_indices, v_norms = v_quant.quantize(v_vec)

                _compressed_cache[layer_id][(block_idx, offset, h)] = (
                    (k_indices, k_norms), (v_indices, v_norms))

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_cache is None:
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale,
                                   output_block_scale)

        # Decompress all cached K/V back into the paged cache
        layer_id = id(layer)
        store = _compressed_cache.get(layer_id)
        if store:
            head_dim = query.shape[-1]
            device = str(query.device)
            k_quant, v_quant = _get_quantizers(head_dim, device)
            key_cache, value_cache = kv_cache.unbind(0)

            for (block_idx, offset, head_idx), (ck, cv) in store.items():
                k_dec = k_quant.dequantize(*ck).squeeze(0).to(key_cache.dtype)
                v_dec = v_quant.dequantize(*cv).squeeze(0).to(value_cache.dtype)
                key_cache[block_idx, offset, head_idx] = k_dec
                value_cache[block_idx, offset, head_idx] = v_dec

        # Now run normal FlashAttention with decompressed cache
        return super().forward(layer, query, key, value, kv_cache,
                              attn_metadata, output, output_scale,
                              output_block_scale)


# Set the implementation class on the backend
TurboQuantAttentionBackend.impl_cls = TurboQuantAttentionImpl
TurboQuantAttentionBackend.metadata_cls = FlashAttentionMetadata
TurboQuantAttentionBackend.metadata_builder_cls = FlashAttentionMetadataBuilder
