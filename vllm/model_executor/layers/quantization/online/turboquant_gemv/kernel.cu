// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboQuant bs=1 GEMV kernel.
//
// Two entry points are bound on the Python side:
//   tq3_decode_only(packed)           — decode-only test harness
//   tq3_gemv_bs1(x_rot, packed,       — warp-per-output-channel GEMV
//                norms, codebook)
//
// Pack format (must stay bit-identical to
// vllm/model_executor/layers/quantization/online/turboquant.py::_pack_indices):
//   48 bytes per 128-value group, laid out as 16 × 3-byte triplets. Within
//   a triplet, the third value's bits straddle the byte boundary (positions
//   2 and 5), which is why the extract below has two OR-merges.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

namespace vllm {
namespace turboquant {

constexpr int GROUP_SIZE = 128;
constexpr int INDICES_PER_GROUP = GROUP_SIZE;
constexpr int BYTES_PER_GROUP = 48;          // = GROUP_SIZE * 3 / 8
constexpr int INDICES_PER_CHUNK = 32;        // one thread's share
constexpr int BYTES_PER_CHUNK = 12;          // = INDICES_PER_CHUNK * 3 / 8
constexpr int CHUNKS_PER_GROUP = 4;          // 4 × 12 = 48 bytes
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

// Decode 32 indices from a 12-byte chunk. Bit-equivalent to
// pack.py::decode_per_thread. Misaligned loads (tid*12 ∈ {0,12,24,36}) are
// handled by memcpy — nvcc lowers to efficient byte reads.
__device__ __forceinline__ void decode_32_indices(
    const uint8_t* __restrict__ chunk_12_bytes,
    int32_t out[INDICES_PER_CHUNK])
{
    uint32_t u0, u1, u2;
    memcpy(&u0, chunk_12_bytes + 0, 4);
    memcpy(&u1, chunk_12_bytes + 4, 4);
    memcpy(&u2, chunk_12_bytes + 8, 4);
    const uint32_t b[12] = {
        (u0 >>  0) & 0xFF, (u0 >>  8) & 0xFF, (u0 >> 16) & 0xFF, (u0 >> 24) & 0xFF,
        (u1 >>  0) & 0xFF, (u1 >>  8) & 0xFF, (u1 >> 16) & 0xFF, (u1 >> 24) & 0xFF,
        (u2 >>  0) & 0xFF, (u2 >>  8) & 0xFF, (u2 >> 16) & 0xFF, (u2 >> 24) & 0xFF,
    };
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t b0 = b[t * 3 + 0];
        const uint32_t b1 = b[t * 3 + 1];
        const uint32_t b2 = b[t * 3 + 2];
        out[t * 8 + 0] = int32_t( b0                  & 0x7);
        out[t * 8 + 1] = int32_t((b0 >> 3)            & 0x7);
        out[t * 8 + 2] = int32_t(((b0 >> 6) | (b1 << 2)) & 0x7);
        out[t * 8 + 3] = int32_t((b1 >> 1)            & 0x7);
        out[t * 8 + 4] = int32_t((b1 >> 4)            & 0x7);
        out[t * 8 + 5] = int32_t(((b1 >> 7) | (b2 << 1)) & 0x7);
        out[t * 8 + 6] = int32_t((b2 >> 2)            & 0x7);
        out[t * 8 + 7] = int32_t((b2 >> 5)            & 0x7);
    }
}

// Grid: (num_rows,). Block: (4,). Each thread decodes one 12-byte chunk.
__global__ void tq3_decode_only_kernel(
    const uint8_t* __restrict__ packed,
    int32_t* __restrict__ out,
    int num_rows)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= num_rows || tid >= CHUNKS_PER_GROUP) return;

    const uint8_t* chunk = packed + row * BYTES_PER_GROUP + tid * BYTES_PER_CHUNK;
    int32_t* out_ptr = out + row * INDICES_PER_GROUP + tid * INDICES_PER_CHUNK;

    int32_t indices[INDICES_PER_CHUNK];
    decode_32_indices(chunk, indices);
    #pragma unroll
    for (int i = 0; i < INDICES_PER_CHUNK; ++i) {
        out_ptr[i] = indices[i];
    }
}

torch::Tensor tq3_decode_only(torch::Tensor packed)
{
    TORCH_CHECK(packed.device().is_cuda(), "packed must be CUDA");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 2, "packed must be 2D (num_rows, 48)");
    TORCH_CHECK(packed.size(1) == BYTES_PER_GROUP, "packed last dim must be 48");
    TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");

    const int num_rows = packed.size(0);
    auto out = torch::empty({num_rows, INDICES_PER_GROUP},
                            torch::TensorOptions().dtype(torch::kInt32).device(packed.device()));

    dim3 grid(num_rows);
    dim3 block(CHUNKS_PER_GROUP);
    tq3_decode_only_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(),
        out.data_ptr<int32_t>(),
        num_rows);
    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}

// bs=1 GEMV. Grid: (OC,). Block: (32,) — one warp per output channel.
// Each thread walks a stride of n_groups/32 groups along K and accumulates
// in fp32. Final reduction via warp shuffle.
//
// Inputs:
//   x_rot:    (K,) bf16 — FWHT-rotated activation
//   packed:   (OC * n_groups, 48) uint8
//   norms:    (OC, n_groups) bf16
//   codebook: (8,) bf16
//   out:      (OC,) bf16
__global__ void tq3_gemv_bs1_kernel(
    const __nv_bfloat16* __restrict__ x_rot,
    const uint8_t*       __restrict__ packed,
    const __nv_bfloat16* __restrict__ norms,
    const __nv_bfloat16* __restrict__ codebook,
    __nv_bfloat16*       __restrict__ out,
    int K, int OC, int n_groups)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const int oc  = blockIdx.x;
    const int tid = threadIdx.x;
    if (oc >= OC) return;

    float cb[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        cb[i] = __bfloat162float(codebook[i]);
    }

    float psum = 0.0f;

    for (int g = tid; g < n_groups; g += 32) {
        const uint8_t* grp = packed + (oc * n_groups + g) * BYTES_PER_GROUP;
        const float norm = __bfloat162float(norms[oc * n_groups + g]);

        #pragma unroll
        for (int c = 0; c < CHUNKS_PER_GROUP; ++c) {
            int32_t idx[INDICES_PER_CHUNK];
            decode_32_indices(grp + c * BYTES_PER_CHUNK, idx);
            const __nv_bfloat16* x_chunk =
                x_rot + g * GROUP_SIZE + c * INDICES_PER_CHUNK;

            #pragma unroll
            for (int i = 0; i < INDICES_PER_CHUNK; ++i) {
                const float w = cb[idx[i]] * norm;
                const float x = __bfloat162float(x_chunk[i]);
                psum += w * x;
            }
        }
    }

    // Block is exactly one warp, so FULL_MASK is correct for the reduction.
    psum += __shfl_xor_sync(FULL_MASK, psum, 16);
    psum += __shfl_xor_sync(FULL_MASK, psum,  8);
    psum += __shfl_xor_sync(FULL_MASK, psum,  4);
    psum += __shfl_xor_sync(FULL_MASK, psum,  2);
    psum += __shfl_xor_sync(FULL_MASK, psum,  1);

    if (tid == 0) {
        out[oc] = __float2bfloat16(psum);
    }
#endif  // __CUDA_ARCH__ >= 800
}

torch::Tensor tq3_gemv_bs1(
    torch::Tensor x_rot,
    torch::Tensor packed,
    torch::Tensor norms,
    torch::Tensor codebook)
{
    TORCH_CHECK(x_rot.is_cuda() && packed.is_cuda() && norms.is_cuda() && codebook.is_cuda(),
                "all inputs must be CUDA");
    TORCH_CHECK(x_rot.dtype() == torch::kBFloat16, "x_rot must be bf16");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(norms.dtype() == torch::kBFloat16, "norms must be bf16");
    TORCH_CHECK(codebook.dtype() == torch::kBFloat16, "codebook must be bf16");
    TORCH_CHECK(packed.is_contiguous() && norms.is_contiguous(), "must be contiguous");
    TORCH_CHECK(codebook.numel() == 8, "codebook must have 8 entries");

    const int K = x_rot.numel();
    const int OC = norms.size(0);
    const int n_groups = norms.size(1);
    TORCH_CHECK(K == n_groups * GROUP_SIZE, "K must equal n_groups * 128");
    TORCH_CHECK(packed.dim() == 2 && packed.size(0) == OC * n_groups
                && packed.size(1) == BYTES_PER_GROUP,
                "packed shape must be (OC * n_groups, 48)");

    auto out = torch::empty({OC},
        torch::TensorOptions().dtype(torch::kBFloat16).device(x_rot.device()));

    dim3 grid(OC);
    dim3 block(32);
    tq3_gemv_bs1_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_rot.data_ptr()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(norms.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(codebook.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        K, OC, n_groups);
    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}

}  // namespace turboquant
}  // namespace vllm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tq3_decode_only", &vllm::turboquant::tq3_decode_only,
          "TurboQuant 3-bit decode-only kernel (no GEMV)",
          py::arg("packed"));
    m.def("tq3_gemv_bs1", &vllm::turboquant::tq3_gemv_bs1,
          "TurboQuant 3-bit GEMV at bs=1 (one warp per output channel)",
          py::arg("x_rot"), py::arg("packed"), py::arg("norms"),
          py::arg("codebook"));
}
