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

from multiheads_selfattention import MultiHeadsSelfAttention
from mlp import MLP

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # layer norm
        self.norm1 = nn.LayerNorm(embed_dim)

        # multi-head self-attention
        self.self_attention = MultiHeadsSelfAttention(embed_dim, num_heads)

        # layer norm
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP embed_dim --> 4*embed_dim --> embed_dim
        self.feed_forward = MLP(embed_dim, mlp_ratio*embed_dim)

    def forward(self, x):
        # input layer norm
        x1 = self.norm1(x)

        # self-attention
        attn_out = self.self_attention(x1)

        # add layer
        x2 = attn_out + x

        # second layer normalization
        x3 = self.norm2(x2)

        # MLP layer
        x4 = self.feed_forward(x3)

        # add layer
        x_final = x2 + x4

        return x_final
