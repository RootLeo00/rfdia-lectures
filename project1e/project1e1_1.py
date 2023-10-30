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

from vit import ViT

def get_dataset():
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  train_dataset = torchvision.datasets.MNIST('.', train=True, download=True, transform=transform)
  test_dataset = torchvision.datasets.MNIST('.', train=False, download=True, transform=transform)


  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=128)

  return train_loader, test_loader


@torch.no_grad()
def eval_model(model, loader):
  model.eval()
  acc = 0.
  c = 0

  for x, y in loader:
    x, y = x.cuda(), y.cuda()
    yhat = model(x)

    acc += torch.sum(yhat.argmax(dim=1) == y).cpu().item()
    c += len(x)

  model.train()
  return round(100 * acc / c, 2)


def main():
  epochs = 25
  embed_dim = 32
  patch_size = 7
  nb_blocks = 2

  model = ViT(embed_dim, nb_blocks, patch_size).cuda()
  opt = torch.optim.Adam(model.parameters())

  train_loss_stat = []
  eval_loss_stat = []

  train_loader, test_loader = get_dataset()

  for epoch in range(epochs):
    train_loss = 0.
    for x, y in train_loader:
      x, y = x.cuda(), y.cuda()

      opt.zero_grad()
      yhat = model(x)
      loss = F.cross_entropy(yhat, y)
      loss.backward()

      opt.step()

      train_loss += loss.item()

    train_loss_stat.append(train_loss)
    print(f"--- Epoch {epoch} ---")
    print(f"Train loss: {train_loss / len(train_loader)}")

  acc = eval_model(model, test_loader)
  print(f"Test accuracy: {acc}%")
  plt.title("Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.plot(train_loss_stat)
  plt.savefig("./results/train_loss_stats.png")

main()