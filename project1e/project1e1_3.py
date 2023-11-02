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

import timm
from tqdm import tqdm

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
def eval_model(model, loader, num_channels=1):
  model.eval()
  acc = 0.
  c = 0

  for x, y in loader:
    x, y = x.cuda(), y.cuda()
    B, C, W, H = x.shape

    #if C != num_channels:
      #x = torchvision.transforms.Grayscale(x, num_output_channels = num_channels)
      #x = torch.cat([x,x,x], dim = 1)  # adapt number of channels to fit in ViT

    yhat = model(x)

    acc += torch.sum(yhat.argmax(dim=1) == y).cpu().item()
    c += len(x)

  model.train()
  return round(100 * acc / c, 2)



def main(): 
  #list all timm models
  #timm.list_models()
  model = timm.create_model('vit_small_patch16_224', pretrained = True, in_chans = 1, img_size=28)

  model.cuda()
  model.train()

  epochs = 25

  opt = torch.optim.Adam(model.parameters())

  train_loss_stat = []
  eval_loss_stat = []

  train_loader, test_loader = get_dataset()

  for epoch in tqdm(range(epochs)):
    train_loss = 0.

    for x, y in train_loader:
      x, y = x.cuda(), y.cuda()
      #print(x.shape)

      opt.zero_grad()
      #x = None # adapt number of channels to fit in ViT
      yhat = model(x)
      loss = F.cross_entropy(yhat, y)
      loss.backward()

      opt.step()

      train_loss += loss.item()
      
    train_loss_stat.append(train_loss)
    print(f"\n--- Epoch {epoch} ---")
    print(f"Train loss: {train_loss / len(train_loader)}")

  acc = eval_model(model, test_loader, num_channels=3)
  print(f"Test accuracy: {acc}")
  plt.title("Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.plot(train_loss_stat)
  plt.savefig("./results/train_loss_stats_timm_on_mnist_pretrained.png")


main()