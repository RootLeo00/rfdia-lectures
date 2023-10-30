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

from block import Block
from positional_encoding import PositionalEncoding
from patch_embed import PatchEmbed


class ViT(nn.Module):
  def __init__(self, embed_dim, nb_blocks, patch_size, nb_classes=10):
    super().__init__()

    num_patches = (28 // patch_size) ** 2

    self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #None
    self.pos_embed = PositionalEncoding(embed_dim)
    self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)

    blocks = []
    for _ in range(nb_blocks):
      blocks.append(
          Block(embed_dim)
      )
    self.blocks = nn.Sequential(*blocks)

    self.norm = nn.LayerNorm(embed_dim)
    self.head = nn.Linear(embed_dim, nb_classes)#None


  def forward(self, x):
    ## image to patches
    x = self.patch_embed(x)

    ## concatenating class token
    B, N, _ = x.shape
    class_tokens = self.class_token.expand(B, -1, -1)
    x = torch.cat([class_tokens, x], dim=1)

    ## adding positional embedding
    x = x + self.pos_embed(x)

    ## forward in the transformer
    x = self.blocks(x)

    ## Normalize the output
    x = x = self.norm(x)

    output = self.head(x[:, 0]) ## classification output

    return output

