"""
ttconv.py
=========
Tensor-Train compressed replacement for nn.Conv2d.

Key idea: a convolution can always be computed as
    im2col (unfold) -> dense matmul with weight reshaped to
    (C_out, C_in*kH*kW) -> reshape back to spatial output.
That dense matmul is exactly what TTLinear already replaces with a
TT-core contraction. So TTConv2d is a thin wrapper: it unfolds the input
into patches, feeds them through a TTLinear whose in_features =
C_in*kH*kW and out_features = C_out, then folds the result back into
(B, C_out, H_out, W_out).

This reuses the already-verified TT-SVD algorithm and TTLinear forward
pass unchanged -- no new tensor-network math to get wrong, only new
plumbing around unfold/fold.

Tensorization used: out_channels is factorized into m_1..m_d, in_channels
into n_1..n_d, and the spatial receptive field (kH*kW) is appended as one
extra TT core paired with a trivial output mode of size 1 -- i.e. the
full TT-matrix format has d+1 core pairs:
    (m_1,n_1), (m_2,n_2), ..., (m_d,n_d), (1, kH*kW)
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tt_core import TTLinear, factorize_balanced


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else tuple(x)


class TTConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        num_modes: int = 3,
        ranks: Union[int, Sequence[int]] = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        kH, kW = self.kernel_size
        in_modes = (*factorize_balanced(in_channels, num_modes), kH * kW)
        out_modes = (*factorize_balanced(out_channels, num_modes), 1)
        d = len(in_modes)
        ranks = [ranks] * (d - 1) if isinstance(ranks, int) else list(ranks)
        assert len(ranks) == d - 1

        self.in_modes = in_modes
        self.out_modes = out_modes

        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride
        )
        self.tt = TTLinear(
            in_features=in_channels * kH * kW,
            out_features=out_channels,
            in_modes=in_modes,
            out_modes=out_modes,
            ranks=ranks,
            bias=bias,
        )

    def _output_hw(self, H: int, W: int) -> Tuple[int, int]:
        kH, kW = self.kernel_size
        pH, pW = self.padding
        dH, dW = self.dilation
        sH, sW = self.stride
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        return H_out, W_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        H_out, W_out = self._output_hw(H, W)
        patches = self.unfold(x)  # (B, C*kH*kW, L)
        L = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(B * L, -1)  # (B*L, in_features)
        out = self.tt(patches)  # (B*L, out_channels)
        out = out.reshape(B, L, self.out_channels).transpose(1, 2)  # (B, out_channels, L)
        out = out.reshape(B, self.out_channels, H_out, W_out)
        return out

    def num_parameters(self) -> int:
        return self.tt.num_parameters()

    def compression_ratio(self) -> float:
        kH, kW = self.kernel_size
        dense = self.in_channels * kH * kW * self.out_channels
        if self.tt.bias is not None:
            dense += self.out_channels
        return dense / self.num_parameters()

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, num_modes: int = 3, ranks: Union[int, Sequence[int]] = 16) -> "TTConv2d":
        assert conv.groups == 1, "TTConv2d currently supports groups=1 convolutions only"
        kH, kW = conv.kernel_size
        in_channels, out_channels = conv.in_channels, conv.out_channels

        # Build an equivalent dense nn.Linear (this is exactly what im2col
        # convolution reduces to) so we can reuse TTLinear.from_linear's
        # already-verified TT-SVD decomposition unchanged.
        equivalent_linear = nn.Linear(in_channels * kH * kW, out_channels, bias=conv.bias is not None)
        with torch.no_grad():
            equivalent_linear.weight.copy_(conv.weight.detach().reshape(out_channels, -1))
            if conv.bias is not None:
                equivalent_linear.bias.copy_(conv.bias.detach())

        in_modes = (*factorize_balanced(in_channels, num_modes), kH * kW)
        out_modes = (*factorize_balanced(out_channels, num_modes), 1)
        d = len(in_modes)
        ranks = [ranks] * (d - 1) if isinstance(ranks, int) else list(ranks)

        layer = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias is not None,
            num_modes=num_modes,
            ranks=ranks,
        )
        layer.tt = TTLinear.from_linear(
            equivalent_linear, in_modes=in_modes, out_modes=out_modes, ranks=ranks
        )
        return layer
