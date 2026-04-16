# TurboQuant (Online 3-4 bit Weight Compression)

TurboQuant is an online weight-only quantization scheme based on [Zandieh et al., ICLR 2026](https://arxiv.org/abs/2503.19878). It compresses BF16 weights to 3 or 4 bits at model load time and requires **no calibration data**.

TurboQuant combines three techniques:

1. **Walsh–Hadamard randomized rotation** of weight groups. After rotation each coordinate is approximately `N(0, 1/d)`, which lets a small shared codebook work well across all weights.
2. **Lloyd–Max optimal scalar quantization** at 2/3/4 bits into a 4/8/16-entry codebook.
3. **Per-group norm correction** — we store `original_norm / reconstruction_norm` (not raw L2 norms), which halves the quantization error at 3 bits in practice.

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

Defaults to 3-bit weights with group size 128.

## What gets quantized

- **Linear layers** — compressed to 3-bit (default) using `TurboQuantOnlineLinearMethod`.
- **MoE experts** — kept at BF16 in this release. Dispatch falls back to the default unquantized MoE method. MoE support is planned in a follow-up.

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
- **Two Triton kernels** are registered as `torch.library.custom_op` with `register_fake`, so they're fullgraph- and `torch.compile`-safe. The runtime picks between them based on output dimension — a fused dequant-GEMM for smaller outputs, and an FWHT-on-input GEMM for larger outputs (single rotation of the activation instead of N inverse rotations of rows).
- **BF16 tensor cores** are used for the main GEMM. The accumulator is kept in FP32.
- A PyTorch fallback path runs when Triton is unavailable (useful for CPU debugging).

## Minimum hardware

| Implementation | Volta | Turing | Ampere | Ada | Hopper |
| -------------- | ----- | ------ | ------ | --- | ------ |
| TurboQuant     | ❌    | ❌     | ✅︎     | ✅︎  | ✅︎     |

Minimum compute capability is 8.0 (Ampere). The GEMM kernels cast to the model's activation dtype (BF16 or FP16) before `tl.dot` to use Tensor Cores. BF16 Tensor Cores were introduced in Ampere; Turing's 2nd-gen Tensor Cores only support FP16 and would fail on BF16 weights.
