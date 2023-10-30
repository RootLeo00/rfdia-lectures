import pickle
import math

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.utils as vutils
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

class SelfAttention(nn.Module):
  """Implements dot-product self-attention/computes self-attention scores

    Args:
        embed_dim (int): The input feature dimension for Q, K, and V.

    Attributes:
        embed_dim (int): The input feature dimension for Q, K, and V.
        W_q (nn.Linear): Linear projection for the query vector.
        W_k (nn.Linear): Linear projection for the key vector.
        W_v (nn.Linear): Linear projection for the value vector.

    """
  def __init__(self, embed_dim):
    super().__init__()

    # TODO
    self.embed_dim = embed_dim
    self.W_q = nn.Linear(embed_dim, embed_dim)     # linear projections for Q, K, and V
    self.W_k = nn.Linear(embed_dim, embed_dim)
    self.W_v = nn.Linear(embed_dim, embed_dim)

  def forward(self, x):
    """Forward pass of the self-attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C), where B is batch size, N is sequence length,
                             and C is the input feature dimension.
        Returns:
            torch.Tensor: Output tensor after applying self-attention of the same shape as input x.
        """
    B, N, C = x.shape

    # TODO: compute the Q, K, V
    q = self.W_q(x) # computes query vector
    k = self.W_k(x) # computes key vector
    v = self.W_v(x) # computes value vector

    # TODO: compute the attention matrix using Q and K
    attention = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (C ** 0.5), dim=-1) # ATTENTION MATRIX + normalization with sqrt(d)

    # TODO: compute the final version using the attention,
    # V, and the final projection
    x = torch.matmul(attention, v)

    return x #, attention
