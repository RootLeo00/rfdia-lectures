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
import matplotlib as plt

from utils import *
import pandas as pd
from epoch import epoch

PRINT_INTERVAL = 200
PATH="datasets"

# styling and helper packages

# import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl

#sns.set_theme()
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['lines.markersize'] = 4

# plt.style.use('ggplot')
# plt.style.use('seaborn-white')
#mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['lines.linestyle'] = '--'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------------2.1 - Network architecture -----------------------------------------------

# set global paramters for Part 2:

class ConvNet2(nn.Module):
    def __init__(self, channels = 3):
        super(ConvNet2, self).__init__()

        self.channels = channels # to handle color images
        self.conv1 = 32
        self.conv2 = 64
        self.conv3 = 64
        self.fc4 = 1000
        self.fc5 = 10

        # conv net as feature extractor
        self.features = nn.Sequential(
            #--> input: 4x3x28x28 (color image)
            nn.Conv2d(self.channels, 32, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0), # --> output: [4, 32, 16, 16]

            nn.Conv2d(32, 64, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0), #--> ouput: [4, 64, 8, 8])

            nn.Conv2d(64, 64, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0), #--> output: [4, 64, 4, 4])
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, self.fc4),
            nn.ReLU(),
            nn.Linear(self.fc4, self.fc5),

            # Reminder: The softmax is included in the loss, do not put it here
            # nn.Softmax()
        )


    def forward(self, input):
        bsize = input.size(0) # batch size
        output = self.features(input) # output of the conv layers

        output = output.view(bsize, -1) # we flatten the 2D feature maps into one 1D vector for each input
        output = self.classifier(output) # we compute the output of the fc layers

        return output
    
    def count_parameters(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params
    

# adapting the dataset function to CIFAR10

def get_dataset_CIFAR(batch_size, cuda=True):
    """
    This function loads the dataset and performs transformations on each
    image (listed in `transform = ...`).
    """
    train_dataset = datasets.CIFAR10(PATH, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    val_dataset = datasets.CIFAR10(PATH, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, pin_memory=cuda, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=cuda, num_workers=2)

    return train_loader, val_loader

# --------------------------2.2 - Network learning -----------------------------------------------

def main_CIFAR(batch_size, lr, epochs, cuda=True):

    # ex :{"batch_size": 128, "epochs": 5, "lr": 0.1}

    # define model, loss, optim
    model = ConvNet2()
    # total_weights = model.count_parameters()
    # print("Total number of weights to learn:", total_weights)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)

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
    
    #initialize dataframe df
    df_train = pd.DataFrame(columns=['loss', 'top1_acc', 'top5_acc'])
    df_test = pd.DataFrame(columns=['loss', 'top1_acc', 'top5_acc'])

    # We iterate on the epochs
    for i in range(epochs):
        print("=================\n=== EPOCH "+str(i+1)+" =====\n=================\n")
        # Train phase
        top1_acc, avg_top5_acc, loss = epoch(train, model, criterion, optimizer, cuda)
        # Test phase
        top1_acc_test, top5_acc_test, loss_test = epoch(test, model, criterion, cuda=cuda)

        #save data into a dataframe
        newdf_train = pd.DataFrame([[loss.val, top1_acc.val, avg_top5_acc.val]], columns=['loss', 'top1_acc', 'top5_acc'])
        newdf_test = pd.DataFrame([[loss_test.val, top1_acc_test.val, top5_acc_test.val]], columns=['loss', 'top1_acc', 'top5_acc'])
        
        # concat newdf to df
        df_train = pd.concat([df_train, newdf_train], axis=0) # merge across rows (it means that the number of columns should be the same, and we are increasing vertically)
        df_test = pd.concat([df_test, newdf_test], axis=0) # merge across rows (it means that the number of columns should be the same, and we are increasing vertically)
        
        # plot
        plot.update(loss.avg, loss_test.avg, top1_acc.avg, top1_acc_test.avg)

    # save df to a file
    df_train.to_csv('results/results_train_1cd2.csv')
    df_test.to_csv('results/results_val_1cd2.csv')

    
if __name__ == '__main__':
    #start training on CIFAR
    print('Starting training on CIFAR10')
    print('using device: ', device)
    main_CIFAR(batch_size=128, lr=0.1, epochs=50, cuda=True)


# # --------------------------2.3 - Network visualization -----------------------------------------------
# # Visualize convolutional filter
# kernels = np.array(CNN_CIFAR.features[0].weight.detach().permute(0, 2, 3, 1))
# print(kernels.shape)

# kernels[0].shape
# plt.plot(kernels[0] - np.min(kernels[0])) / (np.max(kernels[0]) - np.min(kernels[0]))
# plt.show()

# #fig, axarr = plt.subplots(kernels.size(0))
# #for idx in range(1):
# #    axarr[idx].imshow(kernels[idx].permute(1, 2, 0).squeeze())

# # specify test_model
# test_model = model #CNN_CIFAR

# # load sample image
# train, test = get_dataset_CIFAR(batch_size = 4, cuda = True)
# input_img, target_img = next(iter(test))

# # enforce gradient calculation for the input_img
# input_img.requires_grad = True

# # forward/inference
# out = test_model(input_img)
# print(out.shape)

# #probabilities = torch.nn.functional.softmax(output[0], dim=0)
# #best_id = decode_output(out)

# # backprop
# #out[0, best_id].backward()
# #grads = input_img.grad
# #plot_maps(norm_flat_image(grads),norm_flat_image(input_img) )