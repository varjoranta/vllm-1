# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy loader for the bs=1 CUDA GEMV kernel.

``maybe_load()`` returns an object exposing ``tq3_gemv_bs1`` and
``tq3_decode_only`` if the CUDA extension builds, else ``None`` (caller
falls back to the Triton path).

Arch requirement: sm_80+ (uses ``__nv_bfloat16`` and warp shuffle).
"""

from __future__ import annotations

from functools import cache


@cache
def maybe_load():
    try:
        from vllm.platforms import current_platform
        if not current_platform.is_cuda():
            return None
        if not current_platform.has_device_capability(80):
            return None
        from .build import compile as _compile
        return _compile()
    except Exception:
        # Build failures (missing nvcc, headers, etc.) always fall back
        # to the Triton path — callers never need to know why.
        return None


__all__ = ["maybe_load"]
