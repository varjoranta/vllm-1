# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT-build the bs=1 GEMV kernel extension via torch's CUDA extension loader.

Run on a GPU machine — requires nvcc, CUDA toolkit, and torch. On a host
without CUDA this file imports fine but ``compile()`` will raise.

    from .build import compile
    ext = compile()                 # returns the loaded extension
    indices = ext.tq3_decode_only(packed_tensor)

The first call blocks for a few seconds while nvcc compiles; subsequent
imports hit torch's extension cache in ~/.cache/torch_extensions/.
"""

from __future__ import annotations

from pathlib import Path


def compile():  # noqa: A001 (shadows builtin deliberately; this is the API name)
    """Load the kernel.cu extension via torch.utils.cpp_extension."""
    from torch.utils.cpp_extension import load

    here = Path(__file__).resolve().parent
    return load(
        name="turboquant_gemv_ext",
        sources=[str(here / "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
