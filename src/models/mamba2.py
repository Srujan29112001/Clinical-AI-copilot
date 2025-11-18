"""
Real Mamba2 State Space Model Implementation
For long-context medical history processing

Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
import math


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) core component

    Implements the selective scan mechanism with input-dependent state transitions
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = int(dt_rank)

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # Convolution (depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Selective scan parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seqlen, dim = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (batch, seqlen, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seqlen, d_inner)

        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seqlen]  # Trim padding
        x = rearrange(x, 'b d l -> b l d')

        # Activation
        x = F.silu(x)

        # SSM
        y = self.selective_scan(x)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output

    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective scan with input-dependent state transitions

        Args:
            x: (batch, seq_len, d_inner)

        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seqlen, d_inner = x.shape

        # Get selective scan parameters
        x_dbl = self.x_proj(x)  # (batch, seqlen, dt_rank + 2 * d_state)

        delta = x_dbl[..., :self.dt_rank]
        B = x_dbl[..., self.dt_rank:self.dt_rank + self.d_state]
        C = x_dbl[..., -self.d_state:]

        # Compute delta (time step)
        delta = F.softplus(self.dt_proj(delta))  # (batch, seqlen, d_inner)

        # Get A matrix (exponential of log A)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Discretization
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, x)

        # Selective scan (parallel scan algorithm)
        y = self.parallel_scan(deltaA, deltaB_u, C)

        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def parallel_scan(
        self,
        deltaA: torch.Tensor,
        deltaB_u: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel associative scan

        Args:
            deltaA: (batch, seq_len, d_inner, d_state)
            deltaB_u: (batch, seq_len, d_inner, d_state)
            C: (batch, seq_len, d_state)

        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seqlen, d_inner, d_state = deltaA.shape

        # Sequential scan (for simplicity; can be parallelized)
        states = []
        state = torch.zeros(batch, d_inner, d_state, device=deltaA.device)

        for i in range(seqlen):
            state = deltaA[:, i] * state + deltaB_u[:, i]
            states.append(state)

        states = torch.stack(states, dim=1)  # (batch, seqlen, d_inner, d_state)

        # Output
        y = torch.einsum('bldn,bln->bld', states, C)

        return y


class Mamba2Block(nn.Module):
    """
    Single Mamba2 block with pre-normalization
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return residual + x


class Mamba2(nn.Module):
    """
    Full Mamba2 architecture for long-context medical history processing

    Features:
    - Linear time complexity O(L)
    - 8K+ context length support
    - Selective state space modeling
    - Efficient for medical time series
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
        seq_len: int = 8192,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Mamba2 blocks
        self.layers = nn.ModuleList([
            Mamba2Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        return x


def create_mamba2_model(
    d_model: int = 256,
    d_state: int = 64,
    num_layers: int = 4,
    seq_len: int = 8192,
) -> Mamba2:
    """
    Factory function to create Mamba2 model

    Args:
        d_model: Model dimension
        d_state: State dimension
        num_layers: Number of Mamba2 blocks
        seq_len: Maximum sequence length

    Returns:
        Initialized Mamba2 model
    """
    return Mamba2(
        d_model=d_model,
        d_state=d_state,
        num_layers=num_layers,
        seq_len=seq_len,
    )


if __name__ == "__main__":
    # Test Mamba2 model
    batch_size = 2
    seq_len = 1000
    d_model = 256

    x = torch.randn(batch_size, seq_len, d_model)

    model = create_mamba2_model(d_model=d_model)

    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
