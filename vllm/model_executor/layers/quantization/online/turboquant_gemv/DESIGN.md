# TurboQuant bs=1 GEMV kernel — design

**Status:** design complete, implementation pending.
**Reference:** AWQ's `gemv_cuda.cu` (339 lines) — canonical skeleton we're adapting.

## Kernel signature

```cuda
template <int NPerBlock, int Batch, int BlockSize, int GroupSize>
__global__ void tq3_gemv_kernel(
    const bf16* x_rot,         // (M, IC) pre-rotated input (by _rotate_input, same as existing path)
    const uint32_t* packed,    // (OC/kInterleave, IC * kInterleave * 3 / 32) interleaved 3-bit weights
    const bf16* norms,         // (OC, n_groups) shape-gain per group per output channel
    const bf16* codebook,      // (8,) 8-entry codebook (broadcast once to all threads)
    const bf16* bias,          // (OC,) optional
    bf16* out,                 // (M, OC)
    int IC, int OC);
```

## Launch config

- `NPerBlock = 2` (output cols per block after interleave)
- `kInterleave = 4`
- `BlockSize = 256` threads per block
- `GroupSize = 128`
- `num_blocks = OC / NPerBlock / kInterleave`

For Qwen3-8B q_proj (OC=4096): 4096 / 8 = **512 blocks × 256 threads = 131,072 threads** — enough to saturate A100's 108 SMs comfortably.

## Pack format (decision: **new format, 32×3-bit per 3 u32**)

Each thread reads **96 bits (3 × uint32) = 32 three-bit indices** per step, via one 128-bit aligned `float4` load (wastes 32 bits but keeps alignment):

```
packed layout per thread-step:
  u32[0]: indices  0..10  (bits 0-32: first 10 × 3 bits + 2 bits of #11)
  u32[1]: indices 10..21  (remaining 1 bit of #10 + 10 × 3 bits + 1 bit of #21)
  u32[2]: indices 21..32  (remaining 2 bits of #21 + 10 × 3 bits + final 1 bit of #31)
  u32[3]: UNUSED (128-bit alignment padding)
```

Wait — 32 values × 3 bits = 96 bits, 3 × 32 = 96. Three u32 hold exactly 32 indices. BUT: cross-boundary bits straddle u32s. Need either (a) careful cross-u32 extraction or (b) different packing.

**Simpler alternative: 10 × 3-bit per u32, 2 bit padding per u32.** 32 indices need ceil(32/10) = 4 u32s = 128 bits exactly. Clean alignment, no cross-boundary logic, 6.67% storage overhead vs dense (4 × 32 bits stores 32 × 3.75 effective bits ≈ 3-bit + padding). Acceptable.

**Pack format chosen:**
```
Per 32 output values:
  128 bits = 4 × uint32
  Each uint32 holds 10 × 3-bit indices in low 30 bits, 2 high bits padding
  
u32[0]: indices  0..10 in bits [0..30], bits [30..31] = 0
u32[1]: indices 10..20 in bits [0..30], bits [30..31] = 0
u32[2]: indices 20..30 in bits [0..30], bits [30..31] = 0
u32[3]: indices 30..32 (2 values) in bits [0..6], bits [6..31] = 0
```

Storage overhead: 32 × 3 bits dense = 96 bits; we use 128 bits = **33% overhead**. Hmm, that's heavy.

**Revised chosen format: 32 × 3-bit per 3 u32 with cross-boundary math.** 96 bits exactly, 0% overhead. Cross-boundary logic is static + compile-time; not a runtime cost. This matches what FLUTE does.

Final decode per thread (3 × uint32 → 32 × 3-bit indices):
```cuda
uint3 packed_triplet = *reinterpret_cast<const uint3*>(packed + thread_offset);  // 96-bit load (padded to float4 for alignment)
uint32_t u0 = packed_triplet.x, u1 = packed_triplet.y, u2 = packed_triplet.z;

// Indices 0..10 live entirely in u0 (bits 0..30)
idx[0]  = (u0      ) & 0x7;
idx[1]  = (u0 >>  3) & 0x7;
...
idx[9]  = (u0 >> 27) & 0x7;
// Index 10 straddles u0/u1: bits 30..31 of u0 + bit 0 of u1
idx[10] = ((u0 >> 30) | ((u1 & 1) << 2)) & 0x7;
// Indices 11..20 live in u1 bits 1..31
idx[11] = (u1 >>  1) & 0x7;
...
// etc — fully compile-time predictable
```

All cross-boundary cases: indices 10 and 21 straddle u32 boundaries. Two special cases, handled with `lop3.b32` for the 3-input `(a | (b << 2)) & 0x7` combination in a single PTX op.

## Per-thread work

Each thread processes `kElemsPerThread = 32` K-positions per step:

1. **Load 96 bits of packed weights** → 32 × 3-bit indices (static decode, 2 lop3 ops)
2. **Lookup codebook[idx]** → 32 × bf16 values (codebook in registers; just indexed gather via `__shfl_sync` for inter-thread broadcast or shared memory)
3. **Multiply by norm** → 32 × bf16 scaled values (one `norm` per group, fetched once)
4. **Load 32 x_rot values** via `float4` cast
5. **FMA accumulate** via `__hfma2` (16 `bf16x2` ops per thread step)

Main K-loop iterates `IC / (BlockSize * 32 / kInterleave)` times.

## Warp reduce

Same tree reduction as AWQ:
```cuda
psum += __shfl_xor_sync(~0, psum, 16);
psum += __shfl_xor_sync(~0, psum,  8);
psum += __shfl_xor_sync(~0, psum,  1);
```

## Expected performance

Back-of-envelope for Qwen3-8B q_proj (IC=4096, OC=4096, bs=1) on A100:

- Weight read: 4096 × 4096 × 3 bits / 8 = 6 MB → @ 1.5 TB/s = 4 μs
- Codebook: 32 B in registers, negligible
- Norms: 4096 × 32 × 2 B = 256 KB → @ 1.5 TB/s = 0.17 μs
- x_rot: 4096 × 2 B = 8 KB → 0.005 μs
- FMA: 16.8 MFLOPs → < 1 ns
- **Total bandwidth-limited floor: ~5 μs per call**

Compared to:
- Current Triton: ~250 μs
- BF16 `F.linear`: ~23 μs
- AWQ/Marlin reported: ~15 μs

Target: **~15-25 μs per call**, which gives Qwen3-8B bs=1 throughput of ~60-80 tok/s (matching AWQ).

## Work items

- [ ] Python pack/unpack reference for the 32-per-3-u32 format
- [ ] CPU bit-equivalence test (reuse `cpu_bit_equivalence_v2.py` structure)
- [ ] Checkpoint migration: convert existing 48-byte-per-128 format to new layout at load time (keep on-disk format unchanged; re-pack in `process_weights_after_loading`)
- [ ] CUDA kernel skeleton: load + decode only, return indices tensor for verification
- [ ] Numerical verification vs existing Triton kernel (cos-sim ≥ 0.9999)
- [ ] Add codebook lookup + norm multiply
- [ ] Add GEMV accumulate + warp reduce
- [ ] Template-dispatch over M (1..7)
- [ ] Build system: integrate into vLLM's `setup.py` / CMake
- [ ] Runtime dispatch: `M < 16` → new kernel, else existing Triton
- [ ] Benchmark isolated kernel (target: ≤25 μs/call on A100 for q_proj shape)
- [ ] End-to-end Qwen3-8B bs=1 (target: ≥60 tok/s)
- [ ] Quality regression (MMLU ±0.3% vs Triton baseline)

## References (cloned locally at /Users/varjo/code/)

- `llm-awq/awq/kernels/csrc/quantization_new/gemv/gemv_cuda.cu` — canonical 4-bit GEMV reference (339 lines)
- `flute/flute/csrc/qgemm_kernel_example.cu` — 3-bit LUT example (283 lines)
- `flute/flute/csrc/qgemm_kernel_raw_generated.cu` — shape-specialized 3-bit kernel (825 lines)
- `quip-sharp/quiptools/quiptools_e8p_gemv.cu` — register-staged decode + XOR-parity sign reduction (584 lines)
