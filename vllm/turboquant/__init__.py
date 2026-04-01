# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache compression for vLLM.

Implements TurboQuant+ (Zandieh et al., 2025) for KV cache compression:
- PolarQuant with Walsh-Hadamard Transform rotation
- Lloyd-Max optimal codebook (data-oblivious, no calibration)
- Asymmetric K/V support (K4/V3)
- QJL residual correction for K cache inner product preservation

Reference: https://arxiv.org/abs/2504.19874
Extended by: https://github.com/TheTom/turboquant_plus
"""

from vllm.turboquant.config import TurboQuantConfig
from vllm.turboquant.quantizer import TurboQuantQuantizer

__all__ = ["TurboQuantConfig", "TurboQuantQuantizer"]
