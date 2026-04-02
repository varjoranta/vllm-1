# SPDX-License-Identifier: Apache-2.0
"""Tests for TurboQuant KV cache compression.

Tests are split into CPU-only (config, centroids) and GPU-required
(quantizer round-trip, cache layout, attention backend shapes).
"""

import math

import pytest
import torch

from vllm.turboquant.config import TurboQuantConfig
from vllm.turboquant.centroids import solve_lloyd_max, get_centroids


# ── Config tests (CPU only) ──────────────────────────────────────────


class TestTurboQuantConfig:
    def test_tq3_defaults(self):
        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
        assert c.total_bits == 3
        assert c.mse_bits == 2
        assert c.value_quant_bits == 8  # FP8 default
        assert c.value_fp8 is True

    def test_tq4_defaults(self):
        c = TurboQuantConfig.from_cache_dtype("tq4", head_dim=128)
        assert c.total_bits == 4
        assert c.mse_bits == 3

    def test_tq_k4v3_asymmetric(self):
        c = TurboQuantConfig.from_cache_dtype("tq_k4v3", head_dim=128)
        assert c.asymmetric is True
        assert c.total_bits == 4
        assert c.mse_bits == 3
        assert c.v_total_bits == 3
        assert c.v_mse_bits == 2

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            TurboQuantConfig.from_cache_dtype("tq99", head_dim=128)

    def test_key_packed_size_tq3_128(self):
        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
        # MSE: ceil(128 * 2 / 8) = 32 bytes
        # QJL: ceil(128 / 8) = 16 bytes
        # Norms: 4 bytes
        assert c.key_packed_size == 32 + 16 + 4  # = 52

    def test_key_packed_size_tq4_128(self):
        c = TurboQuantConfig.from_cache_dtype("tq4", head_dim=128)
        # MSE: ceil(128 * 3 / 8) = 48 bytes
        # QJL: 16 bytes, Norms: 4 bytes
        assert c.key_packed_size == 48 + 16 + 4  # = 68

    def test_value_packed_size_fp8(self):
        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
        # FP8: 1 byte per element, no scale/zero
        assert c.value_packed_size == 128

    def test_value_packed_size_4bit(self):
        c = TurboQuantConfig(head_dim=128, total_bits=3, value_quant_bits=4)
        # 4-bit: ceil(128 * 4 / 8) + 4 = 64 + 4 = 68
        assert c.value_packed_size == 68

    def test_slot_size_tq3_fp8(self):
        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
        # key: 52, value: 128 (FP8)
        assert c.slot_size == 52 + 128  # = 180

    def test_padded_slot_power_of_2(self):
        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
        padded = c.padded_slot_size
        # Must be power of 2
        assert padded & (padded - 1) == 0
        # Must be >= slot_size
        assert padded >= c.slot_size

    def test_head_dim_256(self):
        """Test with head_dim=256 (Qwen3.5 hybrid attention layers)."""
        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=256)
        assert c.key_packed_size == math.ceil(256 * 2 / 8) + math.ceil(256 / 8) + 4
        assert c.value_packed_size == 256  # FP8
        assert c.padded_slot_size >= c.slot_size

    def test_env_override(self, monkeypatch):
        """TQ_VALUE_BITS env var overrides default."""
        monkeypatch.setenv("TQ_VALUE_BITS", "4")
        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
        assert c.value_quant_bits == 4
        assert c.value_fp8 is False

    def test_backward_compat_alias(self):
        c = TurboQuantConfig.from_kv_cache_dtype("tq4", head_dim=128)
        assert c.total_bits == 4
        assert c.mse_bits == 3

    def test_compression_ratio_tq3_fp8(self):
        """tq3 with FP8 values should give ~2x compression over FP16."""
        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
        fp16_per_head = 128 * 2 * 2  # K + V, 2 bytes each = 512 bytes
        tq_per_head = c.padded_slot_size
        ratio = fp16_per_head / tq_per_head
        assert ratio >= 1.5, f"Expected >= 1.5x compression, got {ratio:.2f}x"


# ── Centroids tests (CPU only) ───────────────────────────────────────


class TestCentroids:
    def test_solve_lloyd_max_2bit(self):
        centroids, boundaries = solve_lloyd_max(128, bits=2)
        assert centroids.shape == (4,)
        assert boundaries.shape == (3,)
        # Centroids should be sorted
        assert (centroids[1:] >= centroids[:-1]).all()
        # Boundaries between centroids
        for i in range(3):
            assert boundaries[i] > centroids[i]
            assert boundaries[i] < centroids[i + 1]

    def test_solve_lloyd_max_3bit(self):
        centroids, boundaries = solve_lloyd_max(128, bits=3)
        assert centroids.shape == (8,)
        assert boundaries.shape == (7,)

    def test_get_centroids_cached(self):
        c1 = get_centroids(128, 2)
        c2 = get_centroids(128, 2)
        assert c1 is c2  # Same object from cache

    def test_centroids_symmetric(self):
        """Centroids for Gaussian should be approximately symmetric."""
        centroids, _ = solve_lloyd_max(128, bits=2)
        # Check approximate symmetry: c[0] ≈ -c[3], c[1] ≈ -c[2]
        assert abs(centroids[0].item() + centroids[3].item()) < 1e-6
        assert abs(centroids[1].item() + centroids[2].item()) < 1e-6


# ── Quantizer tests (GPU required) ───────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuantizer:
    def test_rotation_matrix_orthogonal(self):
        from vllm.turboquant.quantizer import generate_rotation_matrix
        Pi = generate_rotation_matrix(128, seed=42, device="cuda")
        # Pi @ Pi^T should be identity
        identity = Pi @ Pi.T
        eye = torch.eye(128, device="cuda")
        assert torch.allclose(identity, eye, atol=1e-5)

    def test_rotation_matrix_deterministic(self):
        from vllm.turboquant.quantizer import generate_rotation_matrix
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=42)
        assert torch.equal(Pi1, Pi2)

    def test_qjl_matrix_deterministic(self):
        from vllm.turboquant.quantizer import generate_qjl_matrix
        S1 = generate_qjl_matrix(128, seed=42)
        S2 = generate_qjl_matrix(128, seed=42)
        assert torch.equal(S1, S2)

    def test_quantize_dequantize_roundtrip(self):
        from vllm.turboquant.quantizer import TurboQuantizer
        config = TurboQuantConfig(head_dim=128, total_bits=4)
        tq = TurboQuantizer(config, layer_idx=0).cuda()

        x = torch.randn(32, 128, device="cuda")
        compressed = tq.quantize(x)
        x_hat = tq.dequantize(compressed)

        # Check shapes
        assert x_hat.shape == x.shape
        # Check reconstruction quality (cosine similarity > 0.9 for tq4)
        cos_sim = torch.nn.functional.cosine_similarity(
            x.reshape(-1, 128), x_hat.reshape(-1, 128), dim=1
        ).mean()
        assert cos_sim > 0.9, f"Cosine similarity {cos_sim:.4f} too low"

    def test_quantize_output_shapes(self):
        from vllm.turboquant.quantizer import TurboQuantizer
        config = TurboQuantConfig(head_dim=128, total_bits=3)
        tq = TurboQuantizer(config, layer_idx=0).cuda()

        x = torch.randn(16, 128, device="cuda")
        compressed = tq.quantize(x)

        assert compressed["mse_indices"].shape == (16, 128)
        assert compressed["mse_indices"].dtype == torch.uint8
        assert compressed["qjl_signs"].shape == (16, 128)
        assert compressed["qjl_signs"].dtype == torch.int8
        assert compressed["vec_norm"].shape == (16,)
        assert compressed["vec_norm"].dtype == torch.float16
        assert compressed["res_norm"].shape == (16,)

    def test_pack_unpack_roundtrip(self):
        from vllm.turboquant.quantizer import TurboQuantizer
        config = TurboQuantConfig(head_dim=128, total_bits=3)
        tq = TurboQuantizer(config, layer_idx=0).cuda()

        x = torch.randn(8, 128, device="cuda")
        compressed = tq.quantize(x)
        packed = tq.pack_cache(compressed)
        unpacked = tq.unpack_cache(packed)

        assert torch.equal(compressed["mse_indices"], unpacked["mse_indices"])
        # Signs may lose precision in pack/unpack but should be {-1, +1}
        assert (unpacked["qjl_signs"].abs() == 1).all()
        # Norms should survive fp16 roundtrip
        assert torch.allclose(
            compressed["vec_norm"].float(),
            unpacked["vec_norm"].float(),
            atol=1e-3,
        )

    def test_different_seeds_different_matrices(self):
        from vllm.turboquant.quantizer import TurboQuantizer
        config = TurboQuantConfig(head_dim=128, total_bits=3)
        tq0 = TurboQuantizer(config, layer_idx=0).cuda()
        tq1 = TurboQuantizer(config, layer_idx=1).cuda()
        # Different layers should have different rotation matrices
        assert not torch.equal(tq0.Pi, tq1.Pi)


# ── Cache shape tests (CPU only) ─────────────────────────────────────


class TestCacheShape:
    def test_kv_cache_shape_no_leading_2(self):
        """TQ cache has no leading 2 dimension (combined K+V)."""
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        shape = TurboQuantAttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=128,  # effective_head_size = padded_slot // 2
        )
        assert len(shape) == 4
        assert shape[0] == 100  # num_blocks
        assert shape[1] == 16   # block_size
        assert shape[2] == 8    # num_kv_heads
        assert shape[3] == 256  # head_size * 2 = padded_slot

    def test_supported_dtypes(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        assert TurboQuantAttentionBackend.supports_kv_cache_dtype("tq3")
        assert TurboQuantAttentionBackend.supports_kv_cache_dtype("tq4")
        assert TurboQuantAttentionBackend.supports_kv_cache_dtype("tq_k4v3")
        assert not TurboQuantAttentionBackend.supports_kv_cache_dtype("fp8")
        assert not TurboQuantAttentionBackend.supports_kv_cache_dtype(None)

    def test_backend_name(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        assert TurboQuantAttentionBackend.get_name() == "TURBOQUANT"

    def test_supports_decoder_only(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        from vllm.v1.attention.backend import AttentionType
        assert TurboQuantAttentionBackend.supports_attn_type(
            AttentionType.DECODER
        )


# ── Dtype mapping tests (CPU only) ───────────────────────────────────


class TestDtypeMapping:
    def test_tq_dtypes_map_to_uint8(self):
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        assert STR_DTYPE_TO_TORCH_DTYPE["tq3"] == torch.uint8
        assert STR_DTYPE_TO_TORCH_DTYPE["tq4"] == torch.uint8
        assert STR_DTYPE_TO_TORCH_DTYPE["tq_k4v3"] == torch.uint8


# ── Integration test (GPU + model download) ──────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQualityIntegration:
    """End-to-end quality test: serve a small model with TQ and verify
    output is coherent. Uses Qwen3-0.6B (pure transformer, head_dim=128).

    Marked slow — run with: pytest -m "not slow" to skip, or explicitly:
    pytest tests/kernels/test_turboquant.py::TestQualityIntegration -v
    """

    MODEL = "Qwen/Qwen3-0.6B"

    @pytest.mark.slow
    def test_tq3_fp8_generates_coherent_output(self):
        """TQ3 with FP8 values should produce coherent text."""
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=self.MODEL,
            kv_cache_dtype="tq3",
            max_model_len=512,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )
        params = SamplingParams(
            temperature=0.0, max_tokens=30,
        )
        outputs = llm.generate(
            ["The capital of France is"],
            params,
        )
        text = outputs[0].outputs[0].text.lower()
        assert "paris" in text, (
            f"Expected 'paris' in output, got: {text!r}"
        )

    @pytest.mark.slow
    def test_tq4_fp8_generates_coherent_output(self):
        """TQ4 with FP8 values should also produce coherent text."""
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=self.MODEL,
            kv_cache_dtype="tq4",
            max_model_len=512,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )
        params = SamplingParams(
            temperature=0.0, max_tokens=30,
        )
        outputs = llm.generate(
            ["1 + 1 = 2, 2 + 2 = 4, 3 + 3 ="],
            params,
        )
        text = outputs[0].outputs[0].text
        assert "6" in text, (
            f"Expected '6' in output, got: {text!r}"
        )

    @pytest.mark.slow
    def test_fp16_baseline_matches(self):
        """Baseline FP16 should definitely work (sanity check)."""
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=self.MODEL,
            max_model_len=512,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )
        params = SamplingParams(
            temperature=0.0, max_tokens=30,
        )
        outputs = llm.generate(
            ["The capital of France is"],
            params,
        )
        text = outputs[0].outputs[0].text.lower()
        assert "paris" in text, (
            f"Baseline failed — expected 'paris', got: {text!r}"
        )
