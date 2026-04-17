# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU bit-equivalence test: kernel.cu::tq3_decode_only must match
pack.py::decode_row() on the same inputs.

Requires CUDA; JIT-compiles kernel.cu on first run.

Run standalone:
    python tests/test_gpu_decode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pack import decode_row, pack_reference  # noqa: E402


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    # Compile the extension (from ../build.py)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from build import compile as build_ext  # noqa: E402
    ext = build_ext()
    print(f"Loaded extension: {ext}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    rng = np.random.default_rng(0)

    # Test 1: random rows
    for trial, rows in enumerate([1, 4, 17, 128, 513, 4096]):
        indices_np = rng.integers(0, 8, size=(rows, 128), dtype=np.int64)
        packed_np = pack_reference(indices_np)
        packed_cuda = torch.from_numpy(packed_np).cuda().contiguous()
        out_cuda = ext.tq3_decode_only(packed_cuda)
        out_np = out_cuda.cpu().numpy()

        # Compare against Python reference
        for r in range(rows):
            ref = decode_row(packed_np[r])
            assert np.array_equal(out_np[r], ref), (
                f"trial {trial} rows={rows} row {r} mismatch.\n"
                f"  ref[:32] = {ref[:32]}\n"
                f"  cuda[:32] = {out_np[r][:32]}"
            )
        # Bonus: output must equal the original indices (roundtrip)
        assert np.array_equal(out_np, indices_np), (
            f"trial {trial} roundtrip mismatch for rows={rows}"
        )
        print(f"  rows={rows}: PASS ({rows * 128} indices)")

    # Test 2: edge cases
    for label, fill in [("zeros", 0), ("sevens", 7)]:
        indices_np = np.full((16, 128), fill, dtype=np.int64)
        packed_np = pack_reference(indices_np)
        packed_cuda = torch.from_numpy(packed_np).cuda().contiguous()
        out_cuda = ext.tq3_decode_only(packed_cuda)
        assert np.array_equal(out_cuda.cpu().numpy(), indices_np), f"{label} failed"
        print(f"  edge {label}: PASS")

    print("ALL PASS: CUDA kernel is bit-equivalent to Python reference.")


if __name__ == "__main__":
    main()
