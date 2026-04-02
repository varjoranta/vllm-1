# SPDX-License-Identifier: Apache-2.0
"""TurboQuant configuration."""

import math
import os
from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV-cache quantization.

    Args:
        head_dim: Attention head dimension (e.g. 64, 96, 128).
        total_bits: Total bits per coordinate (3 or 4).
            MSE stage uses (total_bits - 1) bits, QJL uses 1 bit.
        value_quant_bits: Bits per value dimension for quantization.
            8 = FP8 E4M3 (default, lossless quality, ~2x compression).
            4 = 16 levels uniform (higher compression, quality loss).
            2 = 4 levels uniform (aggressive, severe quality loss).
        asymmetric: If True, K and V use different bit widths.
            K uses total_bits, V uses v_total_bits.
        v_total_bits: Total bits for V when asymmetric=True.
        seed: Base seed for deterministic random matrix generation.
            Actual seed per layer = seed + layer_idx * 1337.
    """
    head_dim: int = 128
    total_bits: int = 3
    value_quant_bits: int = 8  # FP8 by default — lossless quality
    asymmetric: bool = False
    v_total_bits: int = 3  # only used when asymmetric=True
    seed: int = 42

    @property
    def mse_bits(self) -> int:
        return max(self.total_bits - 1, 1)

    @property
    def v_mse_bits(self) -> int:
        """MSE bits for V (same as K unless asymmetric)."""
        if self.asymmetric:
            return max(self.v_total_bits - 1, 1)
        return self.mse_bits

    @property
    def n_centroids(self) -> int:
        return 2 ** self.mse_bits

    @property
    def key_packed_size(self) -> int:
        """Packed bytes for a single compressed KEY vector.

        Layout:
          - MSE indices: ceil(head_dim * mse_bits / 8) bytes
          - QJL signs:   ceil(head_dim / 8) bytes
          - vec_norm:     2 bytes (float16)
          - res_norm:     2 bytes (float16)
        """
        mse_bytes = math.ceil(self.head_dim * self.mse_bits / 8)
        qjl_bytes = math.ceil(self.head_dim / 8)
        norm_bytes = 4  # 2x float16
        return mse_bytes + qjl_bytes + norm_bytes

    @property
    def effective_value_quant_bits(self) -> int:
        """Actual bits used for value storage."""
        return self.value_quant_bits

    @property
    def value_fp8(self) -> bool:
        """Whether values are stored as FP8 (E4M3)."""
        return self.effective_value_quant_bits == 8

    @property
    def value_packed_size(self) -> int:
        """Packed bytes for a single VALUE vector.

        FP8 mode: head_dim bytes (1 byte per element, no scale/zero).
        Uniform mode: ceil(head_dim * bits / 8) + 4 bytes (scale + zero fp16).
        """
        if self.value_fp8:
            return self.head_dim
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4  # +2 scale(fp16) +2 zero(fp16)

    @property
    def slot_size(self) -> int:
        """Total packed bytes per head per position (key + value combined)."""
        return self.key_packed_size + self.value_packed_size

    @property
    def padded_slot_size(self) -> int:
        """Slot size rounded up to next power of 2 for page alignment."""
        raw = self.slot_size
        s = 1
        while s < raw:
            s <<= 1
        return s

    @property
    def packed_size(self) -> int:
        """Alias for slot_size (backward compat)."""
        return self.slot_size

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int,
                         value_quant_bits: int = 8) -> "TurboQuantConfig":
        """Create config from cache dtype string.

        Supports: tq3, tq4, tq_k4v3 (asymmetric K=4bit, V=3bit).
        Values default to FP8 (8-bit) for quality. Override with
        TQ_VALUE_BITS env var or value_quant_bits parameter.
        """
        vqb_env = os.environ.get("TQ_VALUE_BITS")
        if vqb_env is not None:
            value_quant_bits = int(vqb_env)

        if cache_dtype == "tq3":
            return TurboQuantConfig(head_dim=head_dim, total_bits=3,
                                    value_quant_bits=value_quant_bits)
        elif cache_dtype == "tq4":
            return TurboQuantConfig(head_dim=head_dim, total_bits=4,
                                    value_quant_bits=value_quant_bits)
        elif cache_dtype == "tq_k4v3":
            return TurboQuantConfig(head_dim=head_dim, total_bits=4,
                                    asymmetric=True, v_total_bits=3,
                                    value_quant_bits=value_quant_bits)
        else:
            raise ValueError(f"Unknown TurboQuant cache dtype: {cache_dtype}")

    @classmethod
    def from_kv_cache_dtype(cls, dtype_str: str,
                            head_dim: int = 128) -> "TurboQuantConfig":
        """Alias for from_cache_dtype (backward compat)."""
        return cls.from_cache_dtype(dtype_str, head_dim)
