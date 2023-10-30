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

class PatchEmbed(nn.Module):
  def __init__(self, in_chan=1, patch_size=7, embed_dim=128):
    super().__init__()
    H = 28 #MNIST

    self.embed_dim = embed_dim
    self.num_patches = H // patch_size #get the number of patches, the stride will be the same as the size of the patch to avoid overlap

    self.projection = nn.Conv2d(in_channels = in_chan, out_channels = embed_dim, kernel_size = patch_size, stride = patch_size)

  def forward(self, x):

    x = self.projection(x)
    B, C, H, W = x.shape

    nb_tokens = self.num_patches **2 # number of tokens is number of overall patches
    x = x.reshape(B, C, nb_tokens , -1) #keep batch size and channel size
    x = x.permute(0, 2, 1, 3) # --> B, nb_tokens, C, -1

    return x.view(B, -1, C) #channels from convolution have become features
