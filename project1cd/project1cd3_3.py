import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from project1cd1 import ConvNet2
from epoch import epoch

from utils import *
import pandas as pd

PRINT_INTERVAL = 200
PATH="datasets"

# styling and helper packages

import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl

#sns.set_theme()
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['lines.markersize'] = 4

#plt.style.use('ggplot')
plt.style.use('seaborn-white')
#mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['lines.linestyle'] = '--'



"""## 3.3 Variants of the optimization algorithm"""


def get_dataset_CIFAR(batch_size, cuda=False):
    """
    This function loads the dataset and performs transformations on each
    image (listed in `transform = ...`).
    """

    # image mean and std for CIFAR10
    mean_CIFAR = (0.491, 0.482, 0.447)
    sig_CIFAR = (0.202, 0.199, 0.201)


    train_dataset = datasets.CIFAR10(PATH, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_CIFAR, std=sig_CIFAR),
            transforms.RandomCrop(size = 28),
            transforms.RandomHorizontalFlip(p=0.5)
        ]))


    val_dataset = datasets.CIFAR10(PATH, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_CIFAR, std=sig_CIFAR)
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, pin_memory=cuda, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=cuda, num_workers=2)

    return train_loader, val_loader


# Import the package
from torch.optim import lr_scheduler

def main_CIFAR_3_3(batch_size=128, lr=0.1, epochs=5, cuda=False):

    #   {"batch_size": 128, "epochs": 5, "lr": 0.1}

    # define model, loss, optim
    model = ConvNet2()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    lr_sched = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    if cuda: # only with GPU, and not with CPU
        cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()

    # Get the data
    train, test = get_dataset_CIFAR(batch_size, cuda)

    # init plots
    plot = AccLossPlot()
    global loss_plot
    loss_plot = TrainLossPlot()

    # We iterate on the epochs
    for i in tqdm(range(epochs)):
        print("=================\n=== EPOCH "+str(i+1)+" =====\n=================\n")
        # Train phase
        top1_acc, avg_top5_acc, loss = epoch(train, model, criterion, optimizer, cuda)
        # Test phase
        top1_acc_test, top5_acc_test, loss_test = epoch(test, model, criterion, cuda=cuda)
        # plot
        plot.update(loss.avg, loss_test.avg, top1_acc.avg, top1_acc_test.avg)

        lr_sched.step()

main_CIFAR_3_3(cuda = True)
