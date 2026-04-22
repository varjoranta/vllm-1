# TurboQuant (Online 3-4 bit Weight Compression)

`--quantization turboquant` is an online weight-only quantization scheme that compresses BF16 weights to 3 or 4 bits at model load time and requires **no calibration data**.

It combines three techniques:

1. **Walsh–Hadamard randomized rotation** of weight groups. After rotation each coordinate is approximately `N(0, 1/d)`, which lets a small shared codebook work well across all weights.
2. **Lloyd–Max optimal scalar quantization** at 2/3/4 bits into a 4/8/16-entry codebook.
3. **Per-group shape-gain decomposition** — we store `original_norm / reconstruction_norm` (the classical shape-gain factor, Gray 1984) rather than raw L2 norms. This halves the 3-bit reconstruction error in practice by accounting for the norm shrinkage introduced by the quantization step itself.

## Background and naming

The algorithm implemented here is the **scalar case of HIGGS** (Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh — *Pushing the Limits of Large Language Model Quantization via the Linearity Theorem*, [NAACL 2025](https://aclanthology.org/2025.naacl-long.543/); preprint [arXiv:2411.17525](https://arxiv.org/abs/2411.17525)). HIGGS describes exactly this combination of Random Hadamard Transform pre-processing, MSE-optimal Lloyd–Max grid, and per-group L2 normalization. A reference implementation also exists in [HuggingFace transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/higgs.md).

The implementation was originally based on **TurboQuant** ([Zandieh et al., ICLR 2026, arXiv:2504.19874](https://arxiv.org/abs/2504.19874)), which is actually an **online vector quantizer for KV-cache and approximate nearest-neighbour search**, not for static weight compression. Engineering simplifications during development — choosing scalar over vector quantization, WHT over general random rotations, Lloyd–Max over learned grids — converged the weight path onto the HIGGS scalar algorithm. The KV-cache application of TurboQuant is implemented separately in [#38479](https://github.com/vllm-project/vllm/pull/38479) by @vibhavagarwal5.

The `turboquant` name is kept for API and plugin-package compatibility (`--quantization turboquant`, `OnlineQuantScheme.TURBOQUANT`), but **HIGGS is the correct primary citation** for the algorithm in this weight-compression path.

## When to use TurboQuant

- You have a standard BF16 checkpoint and want to serve it at ~4× smaller GPU memory with **zero calibration work**.
- You want a serving-time drop-in: no offline pre-processing of the checkpoint, no additional tools.
- You accept a small quality cost in exchange for fitting larger models on smaller GPUs.

## Quick start

```bash
vllm serve <model> --quantization turboquant
```

Or via the Python API:

```python
from vllm import LLM

llm = LLM(model="...", quantization="turboquant")
```

Defaults to 3-bit weights with group size 128. To pick a different bit width (2, 3, or 4):

```bash
vllm serve <model> --quantization turboquant --quantization-config '{"bits": 4}'
```

Or via the Python API:

```python
llm = LLM(model="...", quantization="turboquant", quantization_config={"bits": 4})
```

## What gets quantized

- **Linear layers** — compressed to 3-bit (default) using `TurboQuantOnlineLinearMethod`.
- **MoE experts** — compressed to 3-bit (default) using `TurboQuantOnlineFusedMoEMethod`. Each expert's `w13` and `w2` are packed via the same `_compress_2d` pipeline the Linear path uses; a shared scratch pool decompresses one layer at a time, then delegates to vLLM's unquantized MoE kernel. Validated end-to-end on Qwen3-30B-A3B-Instruct-2507 (E=128, top-k=8, hidden=2048, 48 layers): 13.69 GiB weight memory vs ~60 GiB BF16 (**4.4× compression**), **GSM8K-200 accuracy 91.5%** on H100 80GB.
- **Attention projections on partial-rotary models** (`partial_rotary_factor < 1.0`, e.g. MiniMax M2.5/M2.7, Qwen3.6-A3B) use a **block-diagonal Walsh–Hadamard** rotation so the RoPE-rotated prefix and the content-only suffix stay under independent rotations inside a group. Detection is automatic from `VllmConfig.model_config.hf_text_config.partial_rotary_factor`; non-partial-rotary models run full-width WHT as before.
- **Hardware-native FP4 (MXFP4/NVFP4) is out of scope.** The Walsh–Hadamard rotation is applied globally across each weight group, which is fine for per-row scaling but conflicts with per-block scaled formats — a global rotation spreads outlier mass across block boundaries and pollutes the per-block scales. Block-aligned rotation for those formats is a separate PR.

### MoE-specific invocation notes

The MoE path inherits from `OnlineMoEMethodBase` and requires a backend that does **not** permute expert weight storage during setup (FlashInfer-CUTLASS and AITER both do). On Hopper, vLLM auto-selects FlashInfer-CUTLASS for unquantized MoE, which our scratch-pool invariant can't tolerate — so force Triton explicitly:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    quantization="turboquant",
    kernel_config={"moe_backend": "triton"},
)
```

or via CLI: `vllm serve ... --quantization turboquant -cc.moe_backend=triton`. A clear `ValueError` is raised at `process_weights_after_loading` time if an incompatible backend is detected.

Layers can be excluded via the `ignore` list in the `OnlineQuantizationConfigArgs`:

```python
llm = LLM(
    model="...",
    quantization="turboquant",
    quantization_config={
        "ignore": ["re:.*lm_head.*", "model.layers.0.self_attn.o_proj"],
    },
)
```

## Implementation notes

- **Zero GPU memory at init**. Weights are allocated on the meta device; only the compressed form materializes on GPU, per layer, during `process_weights_after_loading`.
- **Two Triton kernels for Linear** are registered as `torch.library.custom_op` with `register_fake`, so they're fullgraph- and `torch.compile`-safe. The runtime picks between them based on output dimension — a fused dequant-GEMM for smaller outputs, and an FWHT-on-input GEMM for larger outputs (single rotation of the activation instead of N inverse rotations of rows).
- **MoE path uses a Python-level dequant** into a module-level bf16 scratch pool (one `w13` + one `w2` buffer, shared across all MoE layers since only one layer runs at a time). The unquantized Triton MoE kernel then runs on the freshly-decompressed pool. A fused 3D CUDA dequant kernel is planned as a follow-up — the Python path is correct but ~5× slower than the theoretical fused path.
- **Block-diagonal WHT (partial-rotary path)** runs through the PyTorch fallback — the current Triton kernels assume a full-width WHT and would produce incorrect output if applied to block-diag-compressed weights. A block-diag Triton variant is a natural follow-up.
- **BF16 tensor cores** are used for the main Linear GEMM. The accumulator is kept in FP32.
- A PyTorch fallback path runs when Triton is unavailable (useful for CPU debugging).

## Known quirks

- **`VLLM_USE_DEEP_GEMM=0`** should be set when running TurboQuant on a box that doesn't have the `deep_gemm` package installed. vLLM's `kernel_warmup` probes every Linear for possible DeepGEMM use and crashes if the package isn't importable, even though TurboQuant never hits that path. This is a vLLM infrastructure behavior unrelated to the algorithm.

## Minimum hardware

| Implementation | Volta | Turing | Ampere | Ada | Hopper |
| -------------- | ----- | ------ | ------ | --- | ------ |
| TurboQuant     | ❌    | ❌     | ✅︎     | ✅︎  | ✅︎     |

Minimum compute capability is 8.0 (Ampere). The GEMM kernels cast to the model's activation dtype (BF16 or FP16) before `tl.dot` to use Tensor Cores. BF16 Tensor Cores were introduced in Ampere; Turing's 2nd-gen Tensor Cores only support FP16 and would fail on BF16 weights.
