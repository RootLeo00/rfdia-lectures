import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from convnet2 import ConvNet2
from epoch import epoch

from utils import *
import pandas as pd

PRINT_INTERVAL = 200
PATH="datasets"

# styling and helper packages

from tqdm import tqdm
import matplotlib as mpl

#sns.set_theme()
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['lines.markersize'] = 4

#plt.style.use('ggplot')
plt.style.use('seaborn-white')
#mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['lines.linestyle'] = '--'
"""# Part 3 : Results improvements

## 3.1 Standardization
"""

# adapting the dataset function to CIFAR10

#--> psydo code: for conv layer in self.features : do this

def get_dataset_CIFAR_improved(batch_size, cuda=False):
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
            transforms.Normalize(mean=mean_CIFAR, std=sig_CIFAR)
        ]))


    val_dataset = datasets.CIFAR10(PATH, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_CIFAR, std=sig_CIFAR) #should we noramlize the validation set as well?
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, pin_memory=cuda, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=cuda, num_workers=2)

    return train_loader, val_loader

# make modifications to main function to accomodate CIFAR10

def main_CIFAR_standardized(batch_size=128, lr=0.1, epochs=5, cuda=False):

    # ex :
    #   {"batch_size": 128, "epochs": 5, "lr": 0.1}

    # define model, loss, optim
    model = ConvNet2()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    if cuda: # only with GPU, and not with CPU
        cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()

    # Get the data
    train, test = get_dataset_CIFAR_improved(batch_size, cuda)

    # init plots
    plot = AccLossPlot()
    global loss_plot
    loss_plot = TrainLossPlot()

    #initialize dataframe df
    df_train = pd.DataFrame(columns=['loss', 'top1_acc', 'top5_acc'])
    df_test = pd.DataFrame(columns=['loss', 'top1_acc', 'top5_acc'])


    # We iterate on the epochs
    for i in tqdm(range(epochs)):
        print("=================\n=== EPOCH "+str(i+1)+" =====\n=================\n")
        # Train phase
        top1_acc, avg_top5_acc, loss = epoch(train, model, criterion, optimizer, cuda)
        # Test phase
        top1_acc_test, top5_acc_test, loss_test = epoch(test, model, criterion, cuda=cuda)
        # plot
        # plot.update(loss.avg, loss_test.avg, top1_acc.avg, top1_acc_test.avg)

        #save data into a dataframe
        newdf_train = pd.DataFrame([[loss.val, top1_acc.val, avg_top5_acc.val]], columns=['loss', 'top1_acc', 'top5_acc'])
        newdf_test = pd.DataFrame([[loss_test.val, top1_acc_test.val, top5_acc_test.val]], columns=['loss', 'top1_acc', 'top5_acc'])
        
        # concat newdf to df
        df_train = pd.concat([df_train, newdf_train], axis=0) # merge across rows (it means that the number of columns should be the same, and we are increasing vertically)
        df_test = pd.concat([df_test, newdf_test], axis=0) # merge across rows (it means that the number of columns should be the same, and we are increasing vertically)
    
    # save df to a file
    df_train.to_csv('results/results_train_1cd3.csv')
    df_test.to_csv('results/results_val_1cd3.csv')
    

#start training on CIFAR

main_CIFAR_standardized(128, 0.1, cuda=True, epochs = 10)
