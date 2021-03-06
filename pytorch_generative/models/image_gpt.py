"""Implementation of (convolutional) ImageGPT.

ImageGPT is an autoregressive generative model which uses a (decoder only)
Transformer architecture for image generation.

N.B.: Our implementation operates over images instead of embedding tokens like 
[1]. This defeats the purpose slightly as the main motivation of the original 
paper is to demonstrate that the same architecture can be effective for both 
images and text. Our intention, instead, is to demonstrate the capabilities of
the pytorch-generative library.

References (used throughout the file):
  [1]: https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf
"""

import torch
from torch import distributions
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base 


class TransformerBlock(nn.Module):
  """An ImageGPT Transformer block."""

  def __init__(self, 
               n_channels, 
               n_attention_heads):
    """Initializes a new TransformerBlock instance.

    Args:
      n_channels: The number of input and output channels.
      n_attention_heads: The number of attention heads to use.
    """
    super().__init__()
    self._attn = pg_nn.MaskedAttention(
        in_channels=n_channels,
        embed_channels=n_channels,
        out_channels=n_channels,
        n_heads=n_attention_heads,
        is_causal=False)
    self._out = nn.Sequential(
        nn.Conv2d(
            in_channels=n_channels, 
            out_channels=4*n_channels, 
            kernel_size=1),
        nn.GELU(),
        nn.Conv2d (
            in_channels=4*n_channels, 
            out_channels=n_channels, 
            kernel_size=1))

  def forward(self, x):
    x = x + self._attn(x)
    return x + self._out(x)


class ImageGPT(base.AutoregressiveModel):
  """The ImageGPT Model.
  
  Unlike [1], our implementation operates over image inputs, instead of 
  embeddings. Because of this, we don't use normalization (such as LayerNorm) 
  because it would break the model's autoregressive property. Furthermore, we
  implement skip connections from each block to the output. We find that this
  makes training a lot more stable and allows for much faster convergence.
  """
  def __init__(self,       
               in_channels,
               in_size,
               out_dim=1,
               probs_fn=torch.sigmoid,
               sample_fn=lambda x: distributions.Bernoulli(probs=x).sample(),
               n_transformer_blocks=8,
               n_attention_heads=4,
               n_embedding_channels=16):
    """Initializes a new ImageGPT instance.
    
    Args:
      in_channels: The number of input channels.
      in_size: Size of the input images. Used to create positional encodings.
      out_dim: The dimension of the output. Given input of the form NCHW, the 
        output from the model will be N out_dim CHW.
      probs_fn: See the base class.
      sample_fn: See the base class.
      n_transformer_blocks: Number of TransformerBlocks to use.
      n_attention_heads: Number of attention heads to use.
      n_embedding_channels: Number of attention embedding channels to use.
    """
    super().__init__(probs_fn, sample_fn)
    self._out_dim = out_dim
    self._pos = nn.Parameter(torch.zeros(1, in_channels, in_size, in_size))
    self._input = pg_nn.MaskedConv2d(
        is_causal=True, 
        in_channels=in_channels,
        out_channels=n_embedding_channels,
        kernel_size=3,
        padding=1)
    self._transformer = nn.ModuleList(
        TransformerBlock(n_channels=n_embedding_channels,
                         n_attention_heads=n_attention_heads)
        for _ in range(n_transformer_blocks))
    self._out = nn.Conv2d(in_channels=n_embedding_channels,
                          out_channels=self._out_dim * in_channels,
                          kernel_size=1)

  def forward(self, x):
    n, c, h, w = x.shape

    x = self._input(x + self._pos)
    skip = x
    for block in self._transformer:
      x = block(x)
      skip = x + skip
    out = self._out(skip).view(n, self._out_dim, c, h, w)
    return self._probs_fn(out)
