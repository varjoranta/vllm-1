# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-equivalence tests for the bs=1 GEMV kernel's thread-level decode path.

These tests verify that the Python reference in ../pack.py produces
byte-for-byte identical output to the existing TurboQuant pack format.
The CUDA kernel in ../kernel.cu must match this reference.

Run: pytest tests/test_pack.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running standalone (no vllm install needed) by importing pack.py
# directly from the sibling directory instead of via the package path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pack import decode_per_thread, decode_row, pack_reference  # noqa: E402


def test_decode_row_roundtrip_random():
    rng = np.random.default_rng(0)
    for trial in range(100):
        rows = int(rng.integers(1, 32))
        indices = rng.integers(0, 8, size=(rows, 128), dtype=np.int64)
        packed = pack_reference(indices)
        for n in range(rows):
            decoded = decode_row(packed[n])
            assert np.array_equal(decoded, indices[n]), (
                f"trial {trial} row {n} mismatch:\n"
                f" expected {indices[n][:16]}\n"
                f" got      {decoded[:16]}"
            )


def test_decode_row_edge_cases():
    # All zeros
    zeros = np.zeros((4, 128), dtype=np.int64)
    assert np.array_equal(decode_row(pack_reference(zeros)[0]), zeros[0])
    # All sevens
    sevens = np.full((4, 128), 7, dtype=np.int64)
    assert np.array_equal(decode_row(pack_reference(sevens)[0]), sevens[0])
    # Stripes
    stripes = np.tile(np.arange(8, dtype=np.int64), (1, 16))
    assert np.array_equal(decode_row(pack_reference(stripes)[0]), stripes[0])


def test_per_thread_consumes_12_bytes():
    """One thread decodes 32 indices from exactly 3 u32s.

    Sanity: calling decode_per_thread four times on non-overlapping
    byte windows must tile the row exactly.
    """
    rng = np.random.default_rng(42)
    indices = rng.integers(0, 8, size=(1, 128), dtype=np.int64)[0]
    packed = pack_reference(indices.reshape(1, 128))[0]  # (48,)
    reassembled = np.zeros(128, dtype=np.int64)
    for t in range(4):
        base = t * 12
        pk = packed[base : base + 12].astype(np.uint32)
        u0 = int(pk[0]) | (int(pk[1]) << 8) | (int(pk[2]) << 16) | (int(pk[3]) << 24)
        u1 = int(pk[4]) | (int(pk[5]) << 8) | (int(pk[6]) << 16) | (int(pk[7]) << 24)
        u2 = int(pk[8]) | (int(pk[9]) << 8) | (int(pk[10]) << 16) | (int(pk[11]) << 24)
        reassembled[t * 32 : (t + 1) * 32] = decode_per_thread((u0, u1, u2))
    assert np.array_equal(reassembled, indices)


def test_values_bounded():
    """Decoded values must be in [0, 7]."""
    rng = np.random.default_rng(1)
    indices = rng.integers(0, 8, size=(5, 128), dtype=np.int64)
    packed = pack_reference(indices)
    for n in range(5):
        decoded = decode_row(packed[n])
        assert decoded.min() >= 0 and decoded.max() <= 7, (decoded.min(), decoded.max())


if __name__ == "__main__":
    test_decode_row_roundtrip_random()
    test_decode_row_edge_cases()
    test_per_thread_consumes_12_bytes()
    test_values_bounded()
    print("PASS: all 4 test functions")
