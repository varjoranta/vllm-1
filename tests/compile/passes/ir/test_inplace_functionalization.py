# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for IR inplace functionalization pass integration.

This test suite verifies that the inplace raising pass, lowering pass,
and clone cleanup pass work together correctly with donated buffer tracking.
"""

import pytest
import torch
from torch import nn
from vllm.compilation.passes.ir.inplace_raising import (
    VllmIRInplaceFunctionalizationPass,
)

import vllm.kernels  # noqa: F401 to register kernels
from vllm import ir
from vllm.compilation.passes.inductor_pass import get_pass_context, pass_context
from vllm.compilation.passes.ir.lowering_pass import (
    CloneCleanupPass,
    VllmIRLoweringPass,
)
from vllm.config import get_current_vllm_config
from vllm.config.utils import Range
from vllm.ir import ops
from vllm.platforms import current_platform

from ...backend import TestBackend


class MaybeInplaceModel(nn.Module):
    """Model using only maybe_inplace variants."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.weight1 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.weight2 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor, residual1: torch.Tensor, residual2: torch.Tensor
    ):
        # First maybe_inplace - x is donated
        x_normed1, residual_out1 = ops.fused_add_rms_norm.maybe_inplace(
            x, residual1, self.weight1, 1e-5
        )
        # Second maybe_inplace - x_normed1 is donated
        x_normed2, residual_out2 = ops.fused_add_rms_norm.maybe_inplace(
            x_normed1, residual2, self.weight2, 1e-5
        )
        return x_normed2, residual_out1, residual_out2


class FunctionalModel(nn.Module):
    """Model using only functional (default) variants."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.weight1 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.weight2 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor, residual1: torch.Tensor, residual2: torch.Tensor
    ):
        # First functional - no donation
        x_normed1, residual_out1 = ops.fused_add_rms_norm(
            x, residual1, self.weight1, 1e-5
        )
        # Second functional - no donation
        x_normed2, residual_out2 = ops.fused_add_rms_norm(
            x_normed1, residual2, self.weight2, 1e-5
        )
        return x_normed2, residual_out1, residual_out2


class MixedModel(nn.Module):
    """Model mixing maybe_inplace and functional variants."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.weight1 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.weight2 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor, residual1: torch.Tensor, residual2: torch.Tensor
    ):
        # First maybe_inplace - x is donated
        x_normed1, residual_out1 = ops.fused_add_rms_norm.maybe_inplace(
            x, residual1, self.weight1, 1e-5
        )
        # Second functional - no donation, but x_normed1 is used
        x_normed2, residual_out2 = ops.fused_add_rms_norm(
            x_normed1, residual2, self.weight2, 1e-5
        )
        # Return both to prevent x_normed1 from being optimized away
        return x_normed1, x_normed2, residual_out1, residual_out2


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Only test on cuda and rocm platform",
)
@pytest.mark.parametrize(
    "model_class,expected_clones,expected_raised",
    [
        (MaybeInplaceModel, 0, 2),  # Both activations donated, no clones needed
        (
            FunctionalModel,
            0,
            0,
        ),  # No donation, no clones (functional ops don't need clones)
        (MixedModel, 0, 1),  # One donated, one not
    ],
)
def test_inplace_functionalization(
    default_vllm_config, model_class, expected_clones, expected_raised
):
    """Test inplace raising, lowering, and clone cleanup with various patterns."""
    torch.set_default_device(current_platform.device_type)

    vllm_config = get_current_vllm_config()

    # Create passes in order they run during compilation
    raising_pass = VllmIRInplaceFunctionalizationPass(vllm_config)
    lowering_pass = VllmIRLoweringPass(vllm_config)
    cleanup_pass = CloneCleanupPass(vllm_config)

    # Set up backend with pre-grad pass
    backend = TestBackend(lowering_pass, cleanup_pass)
    backend.inductor_config["pre_grad_custom_pass"] = raising_pass

    model = model_class()
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    residual1 = torch.randn(8, 16, dtype=torch.bfloat16)
    residual2 = torch.randn(8, 16, dtype=torch.bfloat16)

    # Reference output without optimization
    with ir.direct_dispatch(False):
        ref_output = model(x.clone(), residual1.clone(), residual2.clone())

    # Compile with inplace optimization
    with ir.direct_dispatch(False), pass_context(compile_range=Range(1, 8192)):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        output = compiled_model(x.clone(), residual1.clone(), residual2.clone())

    # Verify correctness (relaxed tolerance for bfloat16)
    for i in range(len(ref_output)):
        torch.testing.assert_close(output[i], ref_output[i], rtol=1e-2, atol=1e-2)

    # Verify expected number of ops were raised
    if expected_raised > 0:
        assert hasattr(raising_pass, "raised_ops")
        assert "fused_add_rms_norm" in raising_pass.functionalized_ops
        assert raising_pass.functionalized_ops["fused_add_rms_norm"] == expected_raised
    else:
        # No maybe_inplace ops, so nothing should be raised
        assert (
            not hasattr(raising_pass, "raised_ops")
            or "fused_add_rms_norm" not in raising_pass.functionalized_ops
        )

    # Verify lowering happened (2 ops in all cases)
    assert "fused_add_rms_norm" in lowering_pass.selected_impls
    assert len(lowering_pass.selected_impls["fused_add_rms_norm"]) == 2

    # Verify expected number of clones after cleanup
    actual_clones = backend.op_count(torch.ops.aten.clone.default, before=False)
    assert actual_clones == expected_clones, (
        f"Expected {expected_clones} clones, got {actual_clones}"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Only test on cuda and rocm platform",
)
def test_donated_buffer_context_propagation(default_vllm_config):
    """Test that donated_input_ids propagates correctly through pass_context."""
    torch.set_default_device(current_platform.device_type)

    vllm_config = get_current_vllm_config()

    # Create a custom backend that inspects pass_context in cleanup pass
    raising_pass = VllmIRInplaceFunctionalizationPass(vllm_config)
    lowering_pass = VllmIRLoweringPass(vllm_config)

    # Track donated_input_ids as seen by cleanup pass
    donated_ids_seen = []

    class InspectingCleanupPass(CloneCleanupPass):
        def __call__(self, graph):
            # Capture donated_input_ids from pass_context
            ctx = get_pass_context()
            if hasattr(ctx, "donated_input_ids"):
                donated_ids_seen.append(set(ctx.donated_input_ids))
            super().__call__(graph)

    cleanup_pass = InspectingCleanupPass(vllm_config)

    backend = TestBackend(lowering_pass, cleanup_pass)
    backend.inductor_config["pre_grad_custom_pass"] = raising_pass

    model = MaybeInplaceModel()
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    residual1 = torch.randn(8, 16, dtype=torch.bfloat16)
    residual2 = torch.randn(8, 16, dtype=torch.bfloat16)

    with ir.direct_dispatch(False), pass_context(compile_range=Range(1, 8192)):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_model(x.clone(), residual1.clone(), residual2.clone())

    # Verify donated_input_ids was set and propagated
    assert len(donated_ids_seen) > 0, "CleanupPass should have seen donated_input_ids"
    # Should have donated inputs (exact indices depend on AOTAutograd)
    assert len(donated_ids_seen[0]) > 0, "Should have at least one donated input"
    # All donated ids should be valid non-negative integers
    for idx in donated_ids_seen[0]:
        assert isinstance(idx, int) and idx >= 0, f"Invalid donated index: {idx}"
