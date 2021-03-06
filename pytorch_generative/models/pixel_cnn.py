"""Implementation of PixelCNN [1].

TODO(eugenhotaj): Explain.

NOTE: Our implementation does *not* use autoregressive channel masking. This
means that each output depends on whole pixels and not sub-pixels. For outputs
with multiple channels, other methods can be used, e.g. [2].

[1]: https://arxiv.org/abs/1606.05328
[2]: https://arxiv.org/abs/1701.05517
"""

import torch
from torch import distributions
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base

 
class MaskedResidualBlock(nn.Module):
  """A residual block masked to respect the autoregressive property."""
  
  def __init__(self, n_channels):
    """Initializes a new MaskedResidualBlock instance.

    Args:
      n_channels: The number of input (and output) channels.
    """
    super().__init__()
    self._net = nn.Sequential(
        # NOTE(eugenhotaj): The PixelCNN paper users ReLU->Conv2d since they do
        # not use a ReLU in the first layer. 
        nn.Conv2d(in_channels=n_channels, 
                  out_channels=n_channels//2, 
                  kernel_size=1),
        nn.ReLU(),
        pg_nn.MaskedConv2d(is_causal=False,
                           in_channels=n_channels//2,
                           out_channels=n_channels//2,
                           kernel_size=3,
                           padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=n_channels//2, 
                  out_channels=n_channels,
                  kernel_size=1),
        nn.ReLU())

  def forward(self, x):
    return x + self._net(x)


class PixelCNN(base.AutoregressiveModel):
  """The PixelCNN model."""

  def __init__(self, 
               in_channels=1, 
               out_dim=1,
               probs_fn=torch.sigmoid,
               sample_fn=lambda x: distributions.Bernoulli(probs=x).sample(),
               n_residual=15,
               residual_channels=128, 
               head_channels=32):
    """Initializes a new PixelCNN instance.
    
    Args:
      in_channels: The number of channels in the input image (typically either 
        1 or 3 for black and white or color images respectively).
      out_dim: The dimension of the output. Given input of the form NCHW, the 
        output from the model will be N out_dim CHW.
      probs_fn: See the base class.
      sample_fn: See the base class.
      n_residual: The number of residual blocks.
      residual_channels: The number of channels to use in the residual layers.
      head_channels: The number of channels to use in the two 1x1 convolutional
        layers at the head of the network.
    """
    super().__init__(probs_fn, sample_fn)
    self._out_dim = out_dim

    self._input = pg_nn.MaskedConv2d(is_causal=True,
                                     in_channels=in_channels,
                                     out_channels=2*residual_channels, 
                                     kernel_size=7, 
                                     padding=3)
    self._masked_layers = nn.ModuleList([
        MaskedResidualBlock(n_channels=2*residual_channels) 
        for _ in range(n_residual) 
    ])
    self._head = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(in_channels=2*residual_channels, 
                  out_channels=head_channels, 
                  kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=head_channels, 
                  out_channels=self._out_dim * in_channels, 
                  kernel_size=1))

  def forward(self, x):
    n, c, h, w = x.shape
    x = self._input(x)
    skip = torch.zeros_like(x) + x
    for layer in self._masked_layers:
      x = layer(x)
      skip += x
    out = self._head(skip).view((n, self._out_dim, c, h, w))
    return self._probs_fn(out)
