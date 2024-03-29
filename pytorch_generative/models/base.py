"""Base classes for models."""

import torch
from torch import distributions
from torch import nn


class AutoregressiveModel(nn.Module):
  """The base class for Autoregressive generative models. """

  def __init__(self, probs_fn, sample_fn):
    """Initializes a new AutoregressiveModel instance.

    Args:
      probs_fn: A function which takes as input the model's logits and returns
        probabilities.
      sample_fn: A function which takes as input sufficient statistics of some
        distribution and returns a sample from that distribution.
    """
    super().__init__()
    self._probs_fn = probs_fn
    self._sample_fn = sample_fn

  def _get_conditioned_on(self, out_shape, conditioned_on):
    assert out_shape is None or conditioned_on is None, \
      'Must provided one, and only one of "out_shape" or "conditioned_on"'
    if conditioned_on is None:
      device = next(self.parameters()).device
      conditioned_on = (torch.ones(out_shape) * - 1).to(device)
    else:
      conditioned_on = conditioned_on.clone()
    return conditioned_on

  # TODO(eugenhotaj): This function does not handle subpixel sampling correctly.
  def sample(self, out_shape=None, conditioned_on=None):
    """Generates new samples from the model.

    The model output is assumed to be the parameters of either a Bernoulli or 
    multinoulli (Categorical) distribution depending on its dimension.

    Args:
      out_shape: The expected shape of the sampled output in NCHW format. 
        Should only be provided when 'conditioned_on=None'.
      conditioned_on: A batch of partial samples to condition the generation on.
          out = distribution(probs=out).sample().view(n, c)
        Only dimensions with values < 0 will be sampled while dimensions with 
        values >= 0 will be left unchanged. If 'None', an unconditional sample
        will be generated.
    """
    with torch.no_grad():
      conditioned_on = self._get_conditioned_on(out_shape, conditioned_on)
      n, c, h, w = conditioned_on.shape
      for row in range(h):
        for col in range(w):
          out = self.forward(conditioned_on)[:, :, :, row, col]
          out = self._sample_fn(out).view(n, c)
          conditioned_on[:, :, row, col] = torch.where(
              conditioned_on[:, :, row, col] < 0,
              out, 
              conditioned_on[:, :, row, col])
      return conditioned_on
