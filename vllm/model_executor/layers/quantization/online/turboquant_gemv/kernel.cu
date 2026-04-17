// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboQuant bs=1 GEMV kernel — SKELETON (not compilable yet).
//
// Purpose: hand-written CUDA kernel for 3-bit quantized Linear at
// batch size 1. The existing Triton kernel is ~10× slower than BF16
// at bs=1 on Qwen3-8B (A100). The bottleneck is structural: Triton's
// 2D-tile compilation model and tl.gather's "naive codegen" on sm_80
// saturate the ALUs regardless of the decode approach.
//
// Raw CUDA with a 1D grid + warp-shuffle reductions is the path past
// the ceiling — Marlin, AWQ, FLUTE, QuIP# all went this route.
//
// This file is a scaffold: function signature + planned structure.
// Fill in during Phase 3. See ./README.md for the design rationale,
// references, and work items.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace vllm {
namespace turboquant {

// --------------------------------------------------------------------------
// Pack format (TBD — pick in work-item 2 of README.md)
//
// Option A: current on-disk format (48 bytes per 128-value group).
//   Pros: no checkpoint migration. Cons: cross-byte 3-bit logic.
//
// Option B: 10-values-per-int32 (30 bits used, 2 padding).
//   Pros: single ldg.u32 → 10 indices via static shifts, no cross-byte.
//   ~8% storage overhead. Cons: checkpoint migration needed.
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Kernel signature (final boss GEMV)
//
// Grid: (cdiv(out_features, BLOCK_N),)
// Block: BLOCK_N threads (64 or 128). Each thread owns one output column.
// Shared memory: x broadcast buffer (in_features × 2 B for bf16).
// No tensor cores. Scalar FMA + __shfl_xor_sync reductions.
// --------------------------------------------------------------------------
template <typename ActT, int BLOCK_N, int GROUP_SIZE>
__global__ void tq_gemv_bs1_kernel(
    const ActT* __restrict__ x,              // (in_features,) bf16/fp16
    const uint8_t* __restrict__ packed,      // (out_features, packed_bytes_per_row) — format per README
    const float* __restrict__ norms,         // (out_features, n_groups) fp32
    const float* __restrict__ centroids,     // (8,) fp32
    const float* __restrict__ signs1,        // (GROUP_SIZE,) WHT signs, pre-rotation input
    const float* __restrict__ signs2,        // (GROUP_SIZE,) WHT signs, post-rotation
    ActT* __restrict__ out,                  // (out_features,) bf16/fp16
    const ActT* __restrict__ bias,           // (out_features,) optional
    int in_features,
    int out_features,
    int n_groups)
{
    // ---- Stage 1: broadcast x into shared memory ----
    // TODO: cooperative load of x into __shared__ ActT x_smem[in_features]
    //       one coalesced load per warp iteration.

    // ---- Stage 2: initialize per-thread state ----
    // TODO: int n = blockIdx.x * BLOCK_N + threadIdx.x;
    //       if (n >= out_features) return;
    //       float acc = 0.0f;

    // ---- Stage 3: loop over groups ----
    // TODO: for (int g = 0; g < n_groups; ++g) {
    //         // Load packed weights for this (n, g) — 48 bytes (or 12 u32 if Option B)
    //         // Decode 128 3-bit indices via static shifts (lop3.b32)
    //         // Lookup centroid values from 8-entry constant-memory codebook
    //         // Apply WHT rotation (absorbed offline into pre-rotated codebook ideally)
    //         // Multiply by norm (one fp32 scalar per group per thread)
    //         // FMA accumulate with x_smem[g*GROUP_SIZE : (g+1)*GROUP_SIZE]
    //       }

    // ---- Stage 4: add bias, cast, store ----
    // TODO: if (bias) acc += static_cast<float>(bias[n]);
    //       out[n] = static_cast<ActT>(acc);
}

// --------------------------------------------------------------------------
// Launcher — wraps kernel launch for torch.library.custom_op
// --------------------------------------------------------------------------
// TODO: extern "C" function that selects BLOCK_N via cuda occupancy calc,
// sets grid/block, launches kernel. Register as torch.library.custom_op
// in launcher.py similar to existing _tq_fwht_input_gemm_op.

}  // namespace turboquant
}  // namespace vllm
