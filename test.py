import functools

import matplotlib.pyplot as plt
import numpy as np
import pytorch_generative as pg
import torch
from torch import distributions
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils import data
from torchvision import transforms
from torchvision import datasets

#####################################################################################################################################

BATCH_SIZE = 128

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    lambda x: distributions.Bernoulli(probs=x).sample()])

train_loader = data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=TRANSFORM),
    batch_size=BATCH_SIZE, 
    shuffle=True,
    pin_memory=True,
    num_workers=10)
test_loader = data.DataLoader(
    datasets.MNIST('../data', train=False, download=True,transform=TRANSFORM),
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers=10)
    
    #####################################################################################################################################
    
class MaskedConv2d(nn.Conv2d):
  """A Conv2d layer masked to respect the autoregressive property.

  Autoregressive masking means that the computation of the current pixel only
  depends on itself, pixels to the left, and pixels above. When the convolution
  is causally masked (i.e. 'is_causal=True'), the computation of the current 
  pixel does not depend on itself.

  E.g. for a 3x3 kernel, the following masks are generated for each channel:
                      [[1 1 1],                   [[1 1 1]
      is_causal=False  [1 1 0],    is_causal=True  [1 0 0]
                       [0 0 0]]                    [0 0 0]
  In [1], they refer to the left masks as 'type A' and right as 'type B'. 

  N.B.: This layer does *not* implement autoregressive channel masking.
  """

  def __init__(self, is_causal, *args, **kwargs):
    """Initializes a new MaskedConv2d instance.
    
    Args:
      is_causal: Whether the convolution should be causally masked.
    """
    super().__init__(*args, **kwargs)
    i, o, h, w = self.weight.shape
    mask = torch.zeros((i, o, h, w))
    mask.data[:, :, :h//2, :] = 1
    mask.data[:, :, h//2, :w//2 + int(not is_causal)] = 1
    self.register_buffer('mask', mask)

  def forward(self, x):
    self.weight.data *= self.mask
    return super().forward(x)


@functools.lru_cache(maxsize=32)
def _get_causal_mask(size):
  mask = np.tril(np.ones((size, size), dtype=np.float32), k=-1)
  mask = torch.tensor(mask).unsqueeze(0)
  return mask


@functools.lru_cache(maxsize=32)
def _get_positional_encoding(shape):
  n, c, h, w = shape
  zeros = torch.zeros(n, c, h, w) 
  return torch.cat((
    (torch.arange(-.5, .5, 1 / h)[None, None, :, None] + zeros),
    (torch.arange(-.5, .5, 1 / w)[None, None, None, :] + zeros)),
    dim=1)

# TODO(eugenhotaj): Should the MaskedAttention block handle positional encodings?
class MaskedAttention(nn.Module):
  """TODO."""

  def __init__(self, 
               query_channels, 
               key_channels,
               value_channels,
               kv_extra_channels=0):
    super().__init__()
    self._key_channels = key_channels
    self._value_channels = value_channels

    kv_in_channels = query_channels + kv_extra_channels
    self._query = nn.Conv2d(
        in_channels=query_channels, 
        out_channels=self._key_channels, 
        kernel_size=1)
    self._kv = nn.Conv2d(
        in_channels=query_channels + kv_extra_channels, 
        out_channels=self._key_channels + self._value_channels, 
        kernel_size=1)
 
  def forward(self, x, kv_extra_channels=None):
    n, _, h, w = x.shape 

    # Compute the query, key, and value.
    query = self._query(x).view(n, self._key_channels, -1)
    if kv_extra_channels is not None:
      x = torch.cat((x, kv_extra_channels), dim=1)
    kv = self._kv(x)
    key = kv[:, :self._key_channels, :, :].view(n, self._key_channels, -1)
    value = kv[:, self._key_channels:, :, :].view(n, self._value_channels, -1)

    # Compute the causual attention weights using stable softmax.
    mask = _get_causal_mask(h * w).to(next(self.parameters()).device)
    probs = (query.permute(0, 2, 1) @ key) - (1. - mask) * 1e10
    probs = probs - probs.max(dim=-1, keepdim=True)[0]
    probs = torch.exp(probs / np.sqrt(self._key_channels)) * mask
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-6)
    
    return (value @ probs.permute(0, 2, 1)).view(n, -1, h, w) 

   
class ResidualBlock(nn.Module):
  
  def __init__(self, n_channels):
    super().__init__()
    self._input_conv = nn.Conv2d(
        in_channels=n_channels, out_channels=n_channels, kernel_size=2, 
        padding=1)
    self._output_conv = nn.Conv2d(
        in_channels=n_channels, out_channels=2*n_channels, kernel_size=2, 
        padding=1)

  def forward(self, x):    
    _, c, h, w = x.shape
    out = F.elu(self._input_conv(F.elu(x))[:, :, :h, :w])
    out = self._output_conv(out)[:, :, :h, :w]
    out, gate = out[:, :c, :, :], out[:, c:, :, :]
    out = out * torch.sigmoid(gate)
    return x + out


class PixelSNAILBlock(nn.Module):
  
  def __init__(self, 
               n_channels,
               n_residual_blocks,
               attention_key_channels,
               attention_value_channels):
    super().__init__()

    def conv(in_channels):
      return nn.Conv2d(in_channels, out_channels=n_channels, kernel_size=1)
       
    self._residual = nn.Sequential(
        *[ResidualBlock(n_channels) for _ in range(n_residual_blocks)])
    self._attention = MaskedAttention(
        query_channels=n_channels,
        key_channels=attention_key_channels,
        value_channels=attention_value_channels,
        kv_extra_channels=1)
    self._residual_out = conv(n_channels)
    self._attention_out = conv(attention_value_channels)
    self._out = conv(n_channels)

  def _elu_conv_elu(self, conv, x):
    return F.elu(conv(F.elu(x)))

  def forward(self, x, input_img):
    res = self._residual(x)
    attn = self._attention(res, input_img)
    res, attn = (self._elu_conv_elu(self._residual_out, res), 
                 self._elu_conv_elu(self._attention_out, attn))
    return self._elu_conv_elu(self._out, res + attn)


class PixelSNAIL(nn.Module):

  def __init__(self, 
               in_channels=1, 
               n_channels=16,
               n_pixel_snail_blocks=1,
               n_residual_blocks=1,
               attention_key_channels=2,
               attention_value_channels=8):
    super().__init__()
    self._input = MaskedConv2d(is_causal=True, 
                               in_channels=in_channels, 
                               out_channels=n_channels,
                               kernel_size=3,
                               padding=1)
    self._pixel_snail_blocks = nn.ModuleList([
        PixelSNAILBlock(n_channels=n_channels,
                        n_residual_blocks=n_residual_blocks, 
                        attention_key_channels=attention_key_channels,
                        attention_value_channels=attention_value_channels)
        for _ in range(n_pixel_snail_blocks)
    ])
    self._output = nn.Sequential(
        nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=1),
        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1))

  def forward(self, x):
    input_img = x
    x = self._input(x)
    skip = x
    for block in self._pixel_snail_blocks:
      x = block(skip, input_img)
      skip += x
    return torch.sigmoid(self._output(skip))
    
#####################################################################################################################################

N_EPOCHS = 427
IN_CHANNELS = 1
N_CHANNELS = 64
N_PIXEL_SNAIL_BLOCKS = 8
N_RESIDUAL_BLOCKS = 2
ATTENTION_VALUE_CHANNELS = N_CHANNELS // 2
ATTENTION_KEY_CHANNELS = ATTENTION_VALUE_CHANNELS // 8

torch.cuda.empty_cache()
model = PixelSNAIL(IN_CHANNELS,
                   N_CHANNELS,
                   N_PIXEL_SNAIL_BLOCKS,
                   N_RESIDUAL_BLOCKS,
                   ATTENTION_KEY_CHANNELS,
                   ATTENTION_VALUE_CHANNELS).to(torch.device("cuda"))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
                                                                         
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: .999977)
bce_loss_fn = nn.BCELoss(reduction='none')

def loss_fn(x, _, preds):
  batch_size = x.shape[0]
  x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
  return bce_loss_fn(preds, x).sum(dim=1).mean()

trainer = pg.trainer.Trainer(model=model, 
                             loss_fn=loss_fn, 
                             optimizer=optimizer, 
                             train_loader=train_loader, 
                             eval_loader=test_loader,
                             lr_scheduler=scheduler,
                            device=torch.device("cuda"))
trainer.interleaved_train_and_eval(N_EPOCHS)

#####################################################################################################################################

def _get_conditioned_on(out_shape, conditioned_on, device):
    assert out_shape is None or conditioned_on is None, \
      'Must provided one, and only one of "out_shape" or "conditioned_on"'
    if conditioned_on is None:
      conditioned_on = (torch.ones(out_shape) * - 1).to(device)
    else:
      conditioned_on = conditioned_on.clone()
    return conditioned_on

def sample(model, out_shape=None, conditioned_on=None):
  """Generates new samples from the model.
  The model output is assumed to be the parameters of either a Bernoulli or 
  multinoulli (Categorical) distribution depending on its dimension.
  Args:
    out_shape: The expected shape of the sampled output in NCHW format. 
      Should only be provided when 'conditioned_on=None'.
    conditioned_on: A batch of partial samples to condition the generation on.
      Only dimensions with values < 0 will be sampled while dimensions with 
      values >= 0 will be left unchanged. If 'None', an unconditional sample
      will be generated.
  """
  device = next(model.parameters()).device
  with torch.no_grad():
    conditioned_on = _get_conditioned_on(out_shape, conditioned_on, device)
    n, c, h, w = conditioned_on.shape
    for row in range(h):
      for column in range(w):
        out = model.forward(conditioned_on)[:, :, row, column]
        distribution = (distributions.Categorical if out.shape[1] > 1 
                        else distributions.Bernoulli)
        out = distribution(probs=out).sample()
        conditioned_on[:, :, row, column] = torch.where(
            conditioned_on[:, :, row, column] < 0,
            out, 
            conditioned_on[:, :, row, column])
    return conditioned_on
    
a = sample(model, (10, 1, 28, 28))
    
    
