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

class MultiHeadsSelfAttention(nn.Module):
    """Implements multi-head self-attention mechanism.

      This module implements the multi-head self-attention mechanism. The input
      tensor is divided into multiple heads, each with its own set of query, key,
      and value projections, and the attention mechanism is applied to each head.
      The outputs of all heads are concatenated and projected to produce
      the final output.

      Args:
          embed_dim (int): The input feature dimension for Q, K, and V.
          num_heads (int): The number of attention heads.

      Attributes:
          head_dim (int): The dimension of each attention head.
          scale (float): The scaling factor for attention scores.
          num_heads (int): The number of attention heads.
          W_q (nn.Linear): Linear projection for the query vector.
          W_k (nn.Linear): Linear projection for the key vector.
          W_v (nn.Linear): Linear projection for the value vector.
          projection (nn.Linear): Linear projection for the final output.

      """
    def __init__(self, embed_dim, num_heads): #input_dim, embed_dim, num_heads
        super().__init__()

        # Calculate the dimension of each attention head
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_heads = num_heads

        # Linear projections for Q, K, and V for each head
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Linear projection for the final output
        self.projection = nn.Linear(embed_dim, embed_dim) #final projection with weight matrix W^0


    def forward(self, x):
        B, N, C = x.shape # batch size, sequence length, channels

        #d_k = q.size()[-1]
        #values = SelfAttention(embed_dim=d_k)(q,k,v) --> rewrite to use self-attention module

        # Linear projections for Q, K, and V for each head
        q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores using Q and K
        attention = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale
        attention = torch.softmax(attention, dim=-1)

        # Apply attention to V
        x = torch.matmul(attention, v)

        # Reshape and concatenate attention outputs
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        # contiguous makes a copy of the tensor such that the order of its elements in memory
        # is the same as if it had been created from scratch with the same data. Some pytorch tensor operations
        # leave the memory arrangement unchanged

        # Apply the final projection
        x = self.projection(x)

        return x
