# TurboQuant bs=1 GEMV — dedicated CUDA kernel

Follow-up to PR #39970. Separate kernel targeting **batch size 1** decode.

## Why a separate kernel

The existing Triton `_polar_fused_gemm_kernel` lands ~10× slower than BF16 at
batch size 1 on Qwen3-8B (measured 8.35 tok/s TQ3 vs 90.1 tok/s BF16 on
A100 80GB, per `results/pr-bench/kernel-bench/` in the verda-model-bench
repo). Full post-mortem in `.plans/triton-ceiling-essay.md` of the bench
repo. Summary: Triton's 2D tile abstractions compose to ALU bottlenecks
at M=1 that the compiler can't remove, and the primitives needed to work
around them (`tl.gather`, `tl.join`, `tl.trans`) carry their own PTX
costs. Raw CUDA with a 1D grid + warp-shuffle reductions is the
established way past this ceiling — Marlin, AWQ, FLUTE, QuIP# all went
this route.

## Target performance

- bs=1 latency within 1.3-1.5× BF16 = ~60-90 tok/s on A100 for Qwen3-8B
- Matches AWQ/Marlin-class numbers
- Bs≥16 continues to use the existing Triton kernel (already reasonable)
- Dispatch in `process_weights_after_loading`: bs<16 → new GEMV kernel,
  bs≥16 → existing Triton

## Architecture (per FLUTE / QuIP# / AWQ reference kernels)

1. **1-D grid over N output columns.** `grid = (cdiv(N, BLOCK_N),)`. One
   CTA owns one BLOCK_N-wide slice of output, loops over full K.
2. **Block = 1 warp × BLOCK_N threads** (BLOCK_N = 64 or 128). Each
   thread owns one output channel.
3. **No tensor cores.** At M=1 tensor cores waste 15/16 throughput. Use
   scalar FMA + warp-shuffle reductions (`__shfl_xor_sync`).
4. **x vector broadcast to SMEM once at kernel entry.** K × 4 B = 32 KB
   for K=8192, fits comfortably.
5. **Norms in registers** (one fp16 per group per thread).
6. **Pack format change**: 10 values per int32 (30 bits used, 2 bits
   padding). One `ldg.u32` load per thread per 10 weights. Decode via
   static shifts. **~8% storage overhead vs 3-bit dense, buys clean
   single-instruction decode.**
7. **`lop3.b32` PTX** for any 3-input bitwise combination. One cycle per op.
8. **`cp.async` + circular double-buffered SMEM** for next group's weights
   while current group computes. Marlin pattern.
9. **Split-K** for narrow-N layers (q_proj with N=4096) to saturate SMs.

## Reference implementations to port from (ranked)

1. `HanGuo97/flute` → `csrc/qgemm_kernel_generated.cu` — 3-bit LUT, A100-optimized, closest match
2. `Cornell-RelaxML/quip-sharp` → `quiptools_cuda/quiptools_e8p_gemv.cu` — register-staged decode, XOR-parity sign reduction (cleaner than Marlin)
3. `IST-DASLab/marlin` → `marlin_cuda_kernel.cu` — L1-bypass + async-pipeline reference
4. `mit-han-lab/llm-awq` → `awq/kernels/csrc/quantization/gemv_cuda.cu` — canonical 4-bit GEMV (~100 lines)

## Work items

- [ ] Clone FLUTE + QuIP# + AWQ GEMV repos locally, read end-to-end
- [ ] Decide pack format: stay at 48-byte-per-128-value OR move to 10-per-int32
  - Stay: zero checkpoint-format break, can't use single-u32-decode trick
  - Change: clean decode, ~8% storage overhead, checkpoint migration needed
- [ ] Implement pack/unpack reference in Python (bit-equivalence test)
- [ ] Implement CUDA kernel skeleton — just load+decode, no GEMV yet
- [ ] Numerical verification against existing Triton kernel output (cos-sim ≥ 0.9999)
- [ ] Add GEMV accumulation with `__shfl_xor_sync` reduction
- [ ] Benchmark isolated kernel per-call (target: ~30-50 μs on A100 for q_proj)
- [ ] Wire into `process_weights_after_loading` dispatch
- [ ] End-to-end bs=1 benchmark on Qwen3-8B (target: ~60+ tok/s on A100)
- [ ] Quality regression check (MMLU/GSM8K parity with Triton kernel)

## Files planned

```
vllm/model_executor/layers/quantization/online/turboquant_gemv/
├── README.md               this file
├── kernel.cu               the CUDA kernel itself (1D grid, no tensor cores)
├── launcher.py             host-side launcher + dispatcher
├── pack.py                 Python-side pack/unpack reference for the chosen format
└── tests/
    └── test_gemv.py        correctness vs Triton + isolated perf bench
```

## Decision log

- **2026-04-17**: Phase 0 coalescing fix (5× win) shipped as PR #39970
  commit. Triton tuning plateau confirmed empirically. This branch
  begins. Next session starts with FLUTE clone + read.
