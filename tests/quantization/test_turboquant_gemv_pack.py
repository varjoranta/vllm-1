# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-equivalence tests for the bs=1 GEMV kernel's pack format.

The CUDA GEMV kernel in
``vllm/model_executor/layers/quantization/online/turboquant_gemv/kernel.cu``
must be byte-for-byte compatible with the 3-bit pack format produced by
``_pack_indices`` in ``turboquant.py``. The Python reference in
``turboquant_gemv/pack.py`` is the readable spec; this file pins that
reference against regressions.

CPU-only — no GPU or nvcc required.
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm.model_executor.layers.quantization.online.turboquant_gemv.pack import (
    decode_per_thread,
    decode_row,
    pack_reference,
)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_decode_row_roundtrip_random(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for trial in range(32):
        rows = int(rng.integers(1, 32))
        indices = rng.integers(0, 8, size=(rows, 128), dtype=np.int64)
        packed = pack_reference(indices)
        for n in range(rows):
            decoded = decode_row(packed[n])
            assert np.array_equal(decoded, indices[n]), (
                f"seed {seed} trial {trial} row {n} mismatch"
            )


def test_decode_row_edge_cases() -> None:
    for fill in (0, 7):
        arr = np.full((4, 128), fill, dtype=np.int64)
        assert np.array_equal(decode_row(pack_reference(arr)[0]), arr[0])
    stripes = np.tile(np.arange(8, dtype=np.int64), (1, 16))
    assert np.array_equal(decode_row(pack_reference(stripes)[0]), stripes[0])


def test_per_thread_consumes_12_bytes() -> None:
    """One thread decodes 32 indices from exactly 3 u32s = 12 bytes."""
    rng = np.random.default_rng(42)
    indices = rng.integers(0, 8, size=(1, 128), dtype=np.int64)[0]
    packed = pack_reference(indices.reshape(1, 128))[0]
    reassembled = np.zeros(128, dtype=np.int64)
    for t in range(4):
        base = t * 12
        pk = packed[base : base + 12].astype(np.uint32)
        u0 = int(pk[0]) | (int(pk[1]) << 8) | (int(pk[2]) << 16) | (int(pk[3]) << 24)
        u1 = int(pk[4]) | (int(pk[5]) << 8) | (int(pk[6]) << 16) | (int(pk[7]) << 24)
        u2 = int(pk[8]) | (int(pk[9]) << 8) | (int(pk[10]) << 16) | (int(pk[11]) << 24)
        reassembled[t * 32 : (t + 1) * 32] = decode_per_thread((u0, u1, u2))
    assert np.array_equal(reassembled, indices)


def test_decoded_values_bounded() -> None:
    rng = np.random.default_rng(1)
    indices = rng.integers(0, 8, size=(5, 128), dtype=np.int64)
    packed = pack_reference(indices)
    for n in range(5):
        decoded = decode_row(packed[n])
        assert decoded.min() >= 0 and decoded.max() <= 7
