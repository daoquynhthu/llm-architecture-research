"""Minimal Mamba-like block for architecture study.

This is a clean-room educational implementation. It mirrors the public
architecture pattern of selective state-space models, but it intentionally
avoids copying optimized upstream CUDA/Triton kernels.

It is suitable for reading, unit tests, and small CPU/GPU experiments. It is
not a performance implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MambaBlockConfig:
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None
    dt_min: float = 1e-3
    dt_max: float = 1e-1


class MinimalMambaBlock(nn.Module):
    """Small, readable Mamba-like sequence mixer.

    Input shape:  (batch, seq, d_model)
    Output shape: (batch, seq, d_model)
    """

    def __init__(self, cfg: MambaBlockConfig):
        super().__init__()
        self.cfg = cfg
        self.d_inner = cfg.d_model * cfg.expand
        self.dt_rank = cfg.dt_rank or math.ceil(cfg.d_model / 16)

        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=False)
        self.depthwise_conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=cfg.d_conv,
            groups=self.d_inner,
            padding=cfg.d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * cfg.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

        # Continuous-time diagonal state matrix A is parameterized in log-space.
        base = torch.arange(1, cfg.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(base).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self._init_dt_bias(cfg.dt_min, cfg.dt_max)

    def _init_dt_bias(self, dt_min: float, dt_max: float) -> None:
        dt = torch.exp(torch.empty(self.d_inner).uniform_(math.log(dt_min), math.log(dt_max)))
        inv_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = x.shape

        xz = self.in_proj(x)
        x_stream, gate = xz.chunk(2, dim=-1)

        # Local causal mixing. Conv1d operates on (B, C, L).
        x_conv = self.depthwise_conv(x_stream.transpose(1, 2))[..., :seqlen]
        x_conv = F.silu(x_conv).transpose(1, 2)

        params = self.x_proj(x_conv)
        dt_raw, b_raw, c_raw = torch.split(
            params,
            [self.dt_rank, self.cfg.d_state, self.cfg.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log.float())

        # Naive sequential selective scan. This is intentionally simple.
        state = torch.zeros(batch, self.d_inner, self.cfg.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seqlen):
            xt = x_conv[:, t]
            dtt = dt[:, t]
            Bt = b_raw[:, t]
            Ct = c_raw[:, t]

            dA = torch.exp(dtt.unsqueeze(-1) * A.unsqueeze(0)).to(x.dtype)
            dB = dtt.unsqueeze(-1) * Bt.unsqueeze(1)
            state = state * dA + xt.unsqueeze(-1) * dB
            yt = torch.einsum("bds,bs->bd", state, Ct) + self.D * xt
            outputs.append(yt)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(gate)
        return self.out_proj(y)


if __name__ == "__main__":
    cfg = MambaBlockConfig(d_model=32)
    block = MinimalMambaBlock(cfg)
    x = torch.randn(2, 8, 32)
    y = block(x)
    assert y.shape == x.shape
    print("ok", tuple(y.shape))
