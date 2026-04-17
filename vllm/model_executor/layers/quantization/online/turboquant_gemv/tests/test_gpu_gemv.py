# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU correctness + latency test for kernel.cu::tq3_gemv_bs1.

Reference: W[oc, k] = codebook[decode(packed)[oc, g(k), k%128]] * norms[oc, g(k)]
           out[oc]  = sum_k W[oc, k] * x[k]

Tolerance: bf16 accumulation at fp32 with up to K=12288 terms → ~1% error
is normal; use atol=5e-2 rtol=5e-2.

Run standalone:
    python tests/test_gpu_gemv.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pack import pack_reference  # noqa: E402


QWEN3_8B_SHAPES = [
    ("q/k/v/o_proj", 4096, 4096),
    ("gate/up_proj", 4096, 12288),
    ("down_proj",    12288, 4096),
]


def _round_to_bf16(arr: np.ndarray) -> np.ndarray:
    """Quantize a fp32 numpy array through bf16 and back to fp32."""
    import torch
    return torch.from_numpy(arr).to(torch.bfloat16).float().numpy()


def build_reference(rng, K: int, OC: int):
    """Random inputs + fp32 reference that matches kernel's bf16 precision."""
    n_groups = K // 128
    indices_3d = rng.integers(0, 8, size=(OC, n_groups, 128), dtype=np.int64)
    packed_np = pack_reference(indices_3d.reshape(OC * n_groups, 128))
    codebook_np = np.sort(rng.standard_normal(8).astype(np.float32))
    norms_np = rng.standard_normal((OC, n_groups)).astype(np.float32) * 0.1
    x_np = rng.standard_normal(K).astype(np.float32) * 0.5

    # Round inputs through bf16 so the reference sees the same values the
    # kernel reads from HBM.
    codebook_bf = _round_to_bf16(codebook_np)
    norms_bf = _round_to_bf16(norms_np)
    x_bf = _round_to_bf16(x_np)

    W_ref = codebook_bf[indices_3d] * norms_bf[:, :, None]
    out_ref = W_ref.reshape(OC, K) @ x_bf
    return packed_np, codebook_np, norms_np, x_np, out_ref


def run_case(ext, torch, name: str, K: int, OC: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    packed_np, codebook_np, norms_np, x_np, out_ref_np = build_reference(rng, K, OC)

    device = "cuda"
    packed = torch.from_numpy(packed_np).to(device).contiguous()
    codebook = torch.from_numpy(codebook_np).to(torch.bfloat16).to(device).contiguous()
    norms = torch.from_numpy(norms_np).to(torch.bfloat16).to(device).contiguous()
    x_rot = torch.from_numpy(x_np).to(torch.bfloat16).to(device).contiguous()

    # Warm + correctness
    out = ext.tq3_gemv_bs1(x_rot, packed, norms, codebook)
    out_np = out.float().cpu().numpy()
    abs_err = np.abs(out_np - out_ref_np)
    # Only compute relative error where reference is meaningfully non-zero;
    # near-zero outputs (cancellation) produce uninformative huge rel values.
    ref_scale = float(np.abs(out_ref_np).max())
    large = np.abs(out_ref_np) > 0.05 * ref_scale
    max_abs = float(abs_err.max())
    max_rel = float((abs_err[large] / np.abs(out_ref_np[large])).max()) if large.any() else 0.0

    # Benchmark: 50 warmup, 200 timed
    for _ in range(50):
        ext.tq3_gemv_bs1(x_rot, packed, norms, codebook)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 200
    for _ in range(iters):
        ext.tq3_gemv_bs1(x_rot, packed, norms, codebook)
    torch.cuda.synchronize()
    us_per_call = (time.perf_counter() - t0) / iters * 1e6

    return {
        "name": name, "K": K, "OC": OC,
        "max_abs": max_abs, "max_rel": max_rel,
        "us": us_per_call,
    }


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from build import compile as build_ext  # noqa: E402
    ext = build_ext()
    print(f"Device: {torch.cuda.get_device_name(0)}")

    print("Correctness + timing:")
    cases = [("tiny", 128, 32), *QWEN3_8B_SHAPES]
    for name, K, OC in cases:
        r = run_case(ext, torch, name, K, OC)
        print(f"  {r['name']:20s} K={r['K']:5d} OC={r['OC']:5d}  "
              f"max_abs={r['max_abs']:.4f} max_rel={r['max_rel']:.4f}  "
              f"{r['us']:.2f} µs")
        assert r["max_rel"] < 5e-2, (
            f"{name}: max_rel={r['max_rel']} > 5% on non-cancellation entries"
        )

    print("\nPASS: CUDA GEMV matches reference.")


if __name__ == "__main__":
    main()
