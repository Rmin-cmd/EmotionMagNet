import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableModReLU(nn.Module):
    """
    Complex modReLU with a learnable bias parameter.

    G(z) = ReLU(|z| + b) * (z / (|z| + eps))
    We initialize b > 0 so that most units are active at the start.
    """

    def __init__(self, init_bias: float = 0.1, eps: float = 1e-6):
        super().__init__()
        # b is now a parameter, initialized positive
        self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
        self.b0 = init_bias
        # self.theta = nn.Parameter(torch.tensor(-1.0))  # unconstrained
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: complex‐valued tensor, dtype=torch.cfloat or torch.cdouble
        # bias = self.b0 - F.softplus(self.theta)
        mag = input.abs()  # |z|
        gated = torch.relu(mag + self.bias)  # ReLU(|z| + b), most channels non‐zero initially
        phase = input / (mag + self.eps)  # safe unit‐phase
        return gated * phase
