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

class MLP(nn.Module):
  def __init__(self, in_features, hid_features):
    super().__init__()

    self.in_features = in_features
    self.hid_features = hid_features
    #modules = []

    self.lin1 = nn.Linear(in_features, hid_features)
    self.gelu1 = nn.GELU()

    self.lin2 = nn.Linear(hid_features, in_features)
    self.gelu2 = nn.GELU()

  def forward(self, x):
    x = self.gelu2((self.lin2(self.gelu1(self.lin1(x)))))

    return x