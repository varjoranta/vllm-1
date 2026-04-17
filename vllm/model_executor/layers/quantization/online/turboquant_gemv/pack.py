# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python reference for the TurboQuant bs=1 GEMV kernel's decode path.

The CUDA kernel in ../kernel.cu must be bit-equivalent to the functions
defined here. This file is the readable specification; the CUDA code is
the fast implementation. Run `pytest tests/test_pack.py` to verify.

Pack format (unchanged from existing TurboQuant): each "group" of 128
3-bit indices is stored in 48 contiguous bytes. Within a group the
layout is 16 triplets × 3 bytes, each triplet encoding 8 indices:

    byte b0: indices 0,1,2 (low bits of 2 cross to b1)
    byte b1: indices 2,3,4,5 (low bit of 5 crosses to b2)
    byte b2: indices 5,6,7 (high bits)

For a bs=1 GEMV kernel, we tile the K dimension in blocks of 32 indices
per thread (4 threads per group). Each thread reads 12 contiguous bytes
(3 × uint32 = 96 bits) and decodes 32 indices.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Pack side — matches vllm._pack_indices for bits=3. Included here for
# self-contained testing.
# ---------------------------------------------------------------------------


def pack_reference(indices: np.ndarray) -> np.ndarray:
    """Pack (rows, n_cols) int indices (values 0-7) into (rows, n_cols*3//8) uint8.

    `n_cols` must be a multiple of 8.
    """
    rows, n_cols = indices.shape
    assert n_cols % 8 == 0, f"n_cols={n_cols} not multiple of 8"
    flat = indices.astype(np.uint8)
    n_triplets = n_cols // 8
    packed = np.zeros((rows, n_triplets * 3), dtype=np.uint8)
    for i in range(n_triplets):
        v = flat[:, i * 8 : (i + 1) * 8]
        packed[:, i * 3 + 0] = v[:, 0] | (v[:, 1] << 3) | ((v[:, 2] & 0x3) << 6)
        packed[:, i * 3 + 1] = (
            (v[:, 2] >> 2)
            | (v[:, 3] << 1)
            | (v[:, 4] << 4)
            | ((v[:, 5] & 0x1) << 7)
        )
        packed[:, i * 3 + 2] = (v[:, 5] >> 1) | (v[:, 6] << 2) | (v[:, 7] << 5)
    return packed


# ---------------------------------------------------------------------------
# Thread-level decode — what ONE thread in the CUDA kernel will compute.
# Each thread is given 12 contiguous bytes (3 × uint32) and produces
# 32 decoded 3-bit indices (values 0..7).
# ---------------------------------------------------------------------------


def decode_per_thread(u32_triplet: tuple[int, int, int]) -> np.ndarray:
    """Decode 32 indices from three uint32 values (96 bits packed).

    Models what the CUDA kernel does with:
        uint3 packed = *reinterpret_cast<const uint3*>(packed_ptr + thread_offset)

    The 12 bytes cover four consecutive 3-byte triplets (= 32 indices):
        bytes [ 0,  1,  2] = triplet A → indices  0..7
        bytes [ 3,  4,  5] = triplet B → indices  8..15
        bytes [ 6,  7,  8] = triplet C → indices 16..23
        bytes [ 9, 10, 11] = triplet D → indices 24..31

    A triplet's three bytes live in adjacent byte positions; depending
    on the thread's byte-offset into the u32s, a triplet may straddle
    u32 boundaries. The CUDA kernel handles all four positions with a
    fixed switch on the thread-lane's alignment.

    For reference purposes we simply pack the three uint32s back into
    12 bytes little-endian and index directly.
    """
    u0, u1, u2 = u32_triplet
    bytes12 = np.array(
        [
            (u0 >>  0) & 0xFF, (u0 >>  8) & 0xFF, (u0 >> 16) & 0xFF, (u0 >> 24) & 0xFF,
            (u1 >>  0) & 0xFF, (u1 >>  8) & 0xFF, (u1 >> 16) & 0xFF, (u1 >> 24) & 0xFF,
            (u2 >>  0) & 0xFF, (u2 >>  8) & 0xFF, (u2 >> 16) & 0xFF, (u2 >> 24) & 0xFF,
        ],
        dtype=np.int64,
    )
    out = np.zeros(32, dtype=np.int64)
    for t in range(4):  # 4 triplets per thread
        b0, b1, b2 = bytes12[t * 3], bytes12[t * 3 + 1], bytes12[t * 3 + 2]
        out[t * 8 + 0] = b0 & 0x7
        out[t * 8 + 1] = (b0 >> 3) & 0x7
        out[t * 8 + 2] = ((b0 >> 6) | (b1 << 2)) & 0x7
        out[t * 8 + 3] = (b1 >> 1) & 0x7
        out[t * 8 + 4] = (b1 >> 4) & 0x7
        out[t * 8 + 5] = ((b1 >> 7) | (b2 << 1)) & 0x7
        out[t * 8 + 6] = (b2 >> 2) & 0x7
        out[t * 8 + 7] = (b2 >> 5) & 0x7
    return out


def decode_row(packed_row: np.ndarray) -> np.ndarray:
    """Decode all 128 indices in one 48-byte row by applying the per-
    thread decoder four times (4 threads × 32 indices = 128).
    """
    assert packed_row.shape == (48,), f"expected 48 bytes, got {packed_row.shape}"
    p = packed_row.astype(np.uint32)
    out = np.zeros(128, dtype=np.int64)
    for t in range(4):
        # Thread t reads bytes [t*12 : (t+1)*12]
        base = t * 12
        u0 = int(p[base + 0]) | (int(p[base + 1]) << 8) | (int(p[base + 2]) << 16) | (int(p[base + 3]) << 24)
        u1 = int(p[base + 4]) | (int(p[base + 5]) << 8) | (int(p[base + 6]) << 16) | (int(p[base + 7]) << 24)
        u2 = int(p[base + 8]) | (int(p[base + 9]) << 8) | (int(p[base + 10]) << 16) | (int(p[base + 11]) << 24)
        out[t * 32 : (t + 1) * 32] = decode_per_thread((u0, u1, u2))
    return out
