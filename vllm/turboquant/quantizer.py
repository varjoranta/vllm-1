# SPDX-License-Identifier: Apache-2.0
"""TurboQuant quantizer: WHT rotation + Lloyd-Max codebook.

Core quantization logic shared between KV cache compression and
weight compression. Implements PolarQuant (Algorithm 1) from the
TurboQuant paper.
"""

import torch

from vllm.turboquant.centroids import optimal_centroids
from vllm.turboquant.config import TurboQuantConfig


def _fast_wht_batch(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform for batched vectors.

    O(d log d) via butterfly pattern. x shape: (batch, d) where d is power of 2.
    """
    d = x.shape[1]
    h = 1
    while h < d:
        x = x.reshape(-1, d // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.reshape(-1, d)
        h <<= 1
    return x * (d ** -0.5)


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class TurboQuantQuantizer:
    """PolarQuant quantizer with WHT rotation and optimal codebook.

    Supports asymmetric K/V via separate instances with different bit widths.
    """

    def __init__(self, dim: int, bits: int, seed: int = 42,
                 device: str = "cuda"):
        self.dim = dim
        self.bits = bits
        self.device = device
        self.padded_dim = _next_power_of_2(dim)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.signs1 = (torch.randint(0, 2, (self.padded_dim,),
                       generator=gen) * 2 - 1).float().to(device)
        self.signs2 = (torch.randint(0, 2, (self.padded_dim,),
                       generator=gen) * 2 - 1).float().to(device)

        centroids_list = optimal_centroids(bits, dim)
        self.centroids = torch.tensor(centroids_list, dtype=torch.float32,
                                      device=device)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.padded_dim > self.dim:
            padded = torch.zeros(batch, self.padded_dim, device=x.device,
                                dtype=x.dtype)
            padded[:, :self.dim] = x
        else:
            padded = x.clone()
        padded *= self.signs1.unsqueeze(0)
        padded = _fast_wht_batch(padded)
        padded *= self.signs2.unsqueeze(0)
        return padded[:, :self.dim]

    def _rotate_inverse(self, y: torch.Tensor) -> torch.Tensor:
        batch = y.shape[0]
        if self.padded_dim > self.dim:
            padded = torch.zeros(batch, self.padded_dim, device=y.device,
                                dtype=y.dtype)
            padded[:, :self.dim] = y
        else:
            padded = y.clone()
        padded *= self.signs2.unsqueeze(0)
        padded = _fast_wht_batch(padded)
        padded *= self.signs1.unsqueeze(0)
        return padded[:, :self.dim]

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize vectors. x: (batch, dim). Returns (indices, norms)."""
        x = x.float()
        norms = torch.linalg.norm(x, dim=1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        x_unit = x / safe_norms.unsqueeze(1)
        y = self._rotate(x_unit)
        indices = torch.searchsorted(self.boundaries, y.contiguous())
        return indices, norms

    def dequantize(self, indices: torch.Tensor,
                   norms: torch.Tensor) -> torch.Tensor:
        """Dequantize. Returns (batch, dim) float32."""
        y_hat = self.centroids[indices]
        x_hat_unit = self._rotate_inverse(y_hat)
        return x_hat_unit * norms.unsqueeze(1)

    @classmethod
    def from_config(cls, config: TurboQuantConfig, dim: int,
                    is_key: bool = True, device: str = "cuda"):
        """Create quantizer from config. Uses separate seeds for K and V."""
        bits = config.k_bits if is_key else config.v_bits
        seed = config.seed if is_key else config.seed + 500
        return cls(dim=dim, bits=bits, seed=seed, device=device)
