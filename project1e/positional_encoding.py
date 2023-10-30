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

class PositionalEncoding(nn.Module):
  """
  Taken from pytorch documentation: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
  """
  def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        #use torch.exp instead of 10000**2 --> convert with log
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe) # non-learnable parameters that are saved in state_dict
        #register_parameter

  def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return x