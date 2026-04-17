// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboQuant bs=1 GEMV kernel — step 1: decode-only.
//
// This file currently contains only the DECODE path (packed 3-bit bytes →
// int32 indices), exposed via a torch extension so we can verify it is
// bit-equivalent to the Python reference in ./pack.py on real GPU hardware
// before adding GEMV accumulation / warp reduction / codebook lookup.
//
// Pack format (inherited from the existing TurboQuant pack — see
// vllm/model_executor/layers/quantization/online/turboquant.py::_pack_indices):
//   48 bytes per 128-value group, laid out as 16 × 3-byte triplets.
//   Each triplet encodes 8 values with the standard cross-byte layout
//   (see decode loop below).
//
// Per-thread work: 4 threads per row cover one group of 128 values.
// Each thread reads 12 contiguous bytes = 3 × uint32 = 96 bits, and
// decodes 32 indices via static shifts. The boundaries between the
// three uint32 values show up as t=1 and t=2 in the inner loop
// (positions 3/4/5 and 6/7/8 straddle the u32 boundary). The compiler
// resolves the byte extraction into single-cycle shift-and-mask ops.
//
// The reference kernel design for the full GEMV is in DESIGN.md; this
// file will grow with the codebook lookup, norm multiply, FMA
// accumulate, and warp-shuffle reduction in subsequent commits.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

namespace vllm {
namespace turboquant {

// ---------------------------------------------------------------------------
// Decode-only kernel. Bit-equivalent to pack.py::decode_row().
// Grid: (num_rows, 1, 1). Block: (4, 1, 1). Each block decodes one row.
// ---------------------------------------------------------------------------
__global__ void tq3_decode_only_kernel(
    const uint8_t* __restrict__ packed,   // (num_rows, 48)
    int32_t* __restrict__ out,            // (num_rows, 128)
    int num_rows)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;  // 0..3
    if (row >= num_rows || tid >= 4) return;

    // Each thread reads 12 bytes = 3 uint32 at row*48 + tid*12.
    const uint8_t* row_ptr = packed + row * 48 + tid * 12;
    uint32_t u0, u1, u2;
    // Bytes may be misaligned vs uint32 boundary (tid*12 = {0,12,24,36}) —
    // tid=0 and tid=2 are 4-byte aligned, tid=1 and tid=3 are not. Use
    // memcpy() to avoid UB from unaligned pointer casts; nvcc lowers this
    // to efficient byte-wise loads.
    memcpy(&u0, row_ptr + 0, 4);
    memcpy(&u1, row_ptr + 4, 4);
    memcpy(&u2, row_ptr + 8, 4);

    // Extract all 12 bytes as uint32 locals. The compiler resolves these
    // to shift-and-mask on the source registers.
    const uint32_t b[12] = {
        (u0 >>  0) & 0xFF, (u0 >>  8) & 0xFF, (u0 >> 16) & 0xFF, (u0 >> 24) & 0xFF,
        (u1 >>  0) & 0xFF, (u1 >>  8) & 0xFF, (u1 >> 16) & 0xFF, (u1 >> 24) & 0xFF,
        (u2 >>  0) & 0xFF, (u2 >>  8) & 0xFF, (u2 >> 16) & 0xFF, (u2 >> 24) & 0xFF,
    };

    int32_t* out_ptr = out + row * 128 + tid * 32;

    // 4 triplets × 8 indices each = 32 indices per thread.
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t b0 = b[t * 3 + 0];
        const uint32_t b1 = b[t * 3 + 1];
        const uint32_t b2 = b[t * 3 + 2];
        out_ptr[t * 8 + 0] = int32_t( b0                  & 0x7);
        out_ptr[t * 8 + 1] = int32_t((b0 >> 3)            & 0x7);
        out_ptr[t * 8 + 2] = int32_t(((b0 >> 6) | (b1 << 2)) & 0x7);
        out_ptr[t * 8 + 3] = int32_t((b1 >> 1)            & 0x7);
        out_ptr[t * 8 + 4] = int32_t((b1 >> 4)            & 0x7);
        out_ptr[t * 8 + 5] = int32_t(((b1 >> 7) | (b2 << 1)) & 0x7);
        out_ptr[t * 8 + 6] = int32_t((b2 >> 2)            & 0x7);
        out_ptr[t * 8 + 7] = int32_t((b2 >> 5)            & 0x7);
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher. Wraps the kernel for torch.utils.cpp_extension binding.
// ---------------------------------------------------------------------------
torch::Tensor tq3_decode_only(torch::Tensor packed)
{
    TORCH_CHECK(packed.device().is_cuda(), "packed must be CUDA");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 2, "packed must be 2D (num_rows, 48)");
    TORCH_CHECK(packed.size(1) == 48, "packed last dim must be 48");
    TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");

    const int num_rows = packed.size(0);
    auto out = torch::empty({num_rows, 128},
                            torch::TensorOptions().dtype(torch::kInt32).device(packed.device()));

    dim3 grid(num_rows);
    dim3 block(4);
    tq3_decode_only_kernel<<<grid, block, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(),
        out.data_ptr<int32_t>(),
        num_rows);

    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}

}  // namespace turboquant
}  // namespace vllm

// Pybind11 binding — exposed as turboquant_gemv_ext.tq3_decode_only
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tq3_decode_only", &vllm::turboquant::tq3_decode_only,
          "TurboQuant 3-bit decode-only kernel (no GEMV)",
          py::arg("packed"));
}
