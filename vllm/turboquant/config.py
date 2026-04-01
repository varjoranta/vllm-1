# SPDX-License-Identifier: Apache-2.0
"""TurboQuant configuration."""

from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression.

    Supports asymmetric K/V bit widths: K precision dominates quality
    because it controls softmax attention routing. V can be compressed
    more aggressively.
    """
    k_bits: int = 4
    v_bits: int = 4
    group_size: int = 128  # WHT rotation group (matches head_dim)
    seed: int = 42

    @classmethod
    def from_kv_cache_dtype(cls, dtype_str: str) -> "TurboQuantConfig":
        """Parse kv_cache_dtype string into config.

        Supported: "tq3", "tq4", "tq_k4v3"
        """
        if dtype_str == "tq3":
            return cls(k_bits=3, v_bits=3)
        elif dtype_str == "tq4":
            return cls(k_bits=4, v_bits=4)
        elif dtype_str == "tq_k4v3":
            return cls(k_bits=4, v_bits=3)
        else:
            raise ValueError(f"Unknown TurboQuant dtype: {dtype_str}. "
                           "Use tq3, tq4, or tq_k4v3.")
