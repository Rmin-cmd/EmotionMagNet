import torch
import torch.nn as nn


class modReLU(nn.Module):
    r"""
    Helper class to compute :math:`\text{ReLU}(x + b)` on real-valued magnitude torch.Tensor.

    Implements the operation:

    .. math::

        G(x) = \text{ReLU}(x + b)
    """

    def __init__(self, bias: float = 0.0) -> None:
        super(modReLU, self).__init__()

        assert bias < 0, "bias must be smaller than 0 to have a non-linearity effect"

        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes the :math:`\text{ReLU}(x + b)` functionality.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\text{ReLU}(x + b)`
        """
        # z: complex‐valued tensor, dtype=torch.cfloat or torch.cdouble
        mag = input.abs()
        # apply bias and ReLU to magnitude
        gated_mag = torch.relu(mag + self.bias)
        # recover unit‐phase
        phase = input / (mag + 1e-8)
        return gated_mag * phase