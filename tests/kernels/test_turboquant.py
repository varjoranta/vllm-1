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
        # MSE: ceil(128 * 2 / 8) = 32, QJL: 16, Norms: 4
        assert c.key_packed_size == 52

    def test_key_packed_size_tq4_128(self):
        c = TurboQuantConfig.from_cache_dtype("tq4", head_dim=128)
        # MSE: ceil(128 * 3 / 8) = 48, QJL: 16, Norms: 4
        assert c.key_packed_size == 68

    def test_value_packed_size_fp8(self):
        c = TurboQuantConfig(head_dim=128, total_bits=3, value_quant_bits=8)
        assert c.value_fp8 is True
        assert c.value_packed_size == 128

    def test_value_packed_size_4bit(self):
        c = TurboQuantConfig(head_dim=128, total_bits=3, value_quant_bits=4)
        # 4-bit: ceil(128 * 4 / 8) + 4 = 68
        assert c.value_packed_size == 68

    def test_padded_slot_power_of_2(self):
        for dtype in ("tq3", "tq4", "tq_k4v3"):
            c = TurboQuantConfig.from_cache_dtype(dtype, head_dim=128)
            padded = c.padded_slot_size
            assert padded & (padded - 1) == 0, f"{dtype}: not power of 2"
            assert padded >= c.slot_size

    def test_head_dim_256(self):
        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=256)
        assert c.key_packed_size == math.ceil(256 * 2 / 8) + math.ceil(256 / 8) + 4

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TQ_VALUE_BITS", "8")
        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
        assert c.value_quant_bits == 8
        assert c.value_fp8 is True

    def test_hybrid_forces_fp8(self):
        c = TurboQuantConfig(head_dim=128, total_bits=3,
                             value_quant_bits=4, hybrid=True)
        assert c.effective_value_quant_bits == 8
        assert c.value_fp8 is True

    def test_asymmetric_v_mse_bits(self):
        c = TurboQuantConfig(head_dim=128, total_bits=4,
                             asymmetric=True, v_total_bits=3)
        assert c.mse_bits == 3      # K: 4 total - 1 QJL = 3
        assert c.v_mse_bits == 2    # V: 3 total - 1 QJL = 2

    def test_symmetric_v_mse_bits(self):
        c = TurboQuantConfig(head_dim=128, total_bits=3)
        assert c.v_mse_bits == c.mse_bits  # same when not asymmetric


# ── Centroids tests (CPU only) ───────────────────────────────────────


class TestCentroids:
    def test_solve_lloyd_max_2bit(self):
        centroids, boundaries = solve_lloyd_max(128, bits=2)
        assert centroids.shape == (4,)
        assert boundaries.shape == (3,)
        assert (centroids[1:] >= centroids[:-1]).all()

    def test_solve_lloyd_max_3bit(self):
        centroids, boundaries = solve_lloyd_max(128, bits=3)
        assert centroids.shape == (8,)
        assert boundaries.shape == (7,)

    def test_get_centroids_cached(self):
        c1 = get_centroids(128, 2)
        c2 = get_centroids(128, 2)
        assert c1 is c2

    def test_centroids_symmetric(self):
        centroids, _ = solve_lloyd_max(128, bits=2)
        assert abs(centroids[0].item() + centroids[3].item()) < 1e-6
        assert abs(centroids[1].item() + centroids[2].item()) < 1e-6


# ── Quantizer tests (GPU required) ───────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuantizer:
    def test_rotation_matrix_orthogonal(self):
        from vllm.turboquant.quantizer import generate_rotation_matrix
        Pi = generate_rotation_matrix(128, seed=42, device="cuda")
        identity = Pi @ Pi.T
        eye = torch.eye(128, device="cuda")
        assert torch.allclose(identity, eye, atol=1e-5)

    def test_rotation_matrix_deterministic(self):
        from vllm.turboquant.quantizer import generate_rotation_matrix
        assert torch.equal(
            generate_rotation_matrix(128, seed=42),
            generate_rotation_matrix(128, seed=42),
        )

    def test_quantize_dequantize_roundtrip(self):
        from vllm.turboquant.quantizer import TurboQuantizer
        config = TurboQuantConfig(head_dim=128, total_bits=4)
        tq = TurboQuantizer(config, layer_idx=0).cuda()
        x = torch.randn(32, 128, device="cuda")
        compressed = tq.quantize(x)
        x_hat = tq.dequantize(compressed)
        cos_sim = torch.nn.functional.cosine_similarity(
            x.reshape(-1, 128), x_hat.reshape(-1, 128), dim=1
        ).mean()
        assert cos_sim > 0.9, f"Cosine similarity {cos_sim:.4f} too low"

    def test_pack_unpack_roundtrip(self):
        from vllm.turboquant.quantizer import TurboQuantizer
        config = TurboQuantConfig(head_dim=128, total_bits=3)
        tq = TurboQuantizer(config, layer_idx=0).cuda()
        x = torch.randn(8, 128, device="cuda")
        compressed = tq.quantize(x)
        packed = tq.pack_cache(compressed)
        unpacked = tq.unpack_cache(packed)
        assert torch.equal(compressed["mse_indices"], unpacked["mse_indices"])
        assert (unpacked["qjl_signs"].abs() == 1).all()

    def test_different_layers_different_matrices(self):
        from vllm.turboquant.quantizer import TurboQuantizer
        config = TurboQuantConfig(head_dim=128, total_bits=3)
        tq0 = TurboQuantizer(config, layer_idx=0).cuda()
        tq1 = TurboQuantizer(config, layer_idx=1).cuda()
        assert not torch.equal(tq0.Pi, tq1.Pi)


# ── Cache shape tests (CPU only) ─────────────────────────────────────


class TestCacheShape:
    def test_kv_cache_shape_no_leading_2(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        shape = TurboQuantAttentionBackend.get_kv_cache_shape(
            num_blocks=100, block_size=16, num_kv_heads=8, head_size=128,
        )
        assert len(shape) == 4
        assert shape == (100, 16, 8, 256)  # head_size * 2

    def test_supported_dtypes_include_k4v3(self):
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
