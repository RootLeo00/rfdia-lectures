import math
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import MNISTData
from tqdm import tqdm

import torch
import torch.nn as nn

class simpleNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(simpleNN, self).__init__()

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.Tanh() #hidden layer

        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax() #output layer

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        #x = self.softmax(x) softmax is already included in pytorch loss function for cross entropy loss
        return x

def init_model(nx, nh, ny, eta):
    #####################
    ## Your code here  ##
    #####################

    model = simpleNN(input_size = nx, hidden_size=nh, output_size=ny)
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=eta)

    ####################
    ##      END        #
    ####################

    return model, loss, optim



def loss_accuracy(loss, Yhat, Y):
    """
    Calculate the loss and accuracy of the model's predictions.

    Parameters:
    - Yhat: Predicted values (torch.Tensor) with shape (batch_size, num_classes)
    - Y: Ground truth labels (torch.Tensor) with shape (batch_size, num_classes)

    Returns:
    - L: Loss (scalar, torch.Tensor)
    - acc: Accuracy (percentage, float)
    """

    assert Yhat.size() == Y.size() # catch dimension errors early on

    epsilon = 1e-10

    L = loss(Yhat, Y)

    # Get the predicted class labels by taking the argmax along axis 1 (columns)
    _, indsY = torch.max(Yhat, 1)

    # Convert one-hot encoded true labels back to class indices
    _, inds_true = torch.max(Y, 1)

    # Compare the predicted labels with the true labels and calculate accuracy
    correct = torch.sum(indsY == inds_true)
    total = Y.size(0)  # Number of samples in the batch
    acc = correct.item() / total

    return L, acc



def main():
    # Write main
    plt.style.use('ggplot')

    data = MNISTData()

    N = data.Xtrain.shape[0]
    Nbatch = 32

    nx = data.Xtrain.shape[1]

    nh = 8

    ny = data.Ytrain.shape[1]
    eta = 0.05

    model, loss, optim = init_model(nx, nh, ny, eta)

    curves = [[],[], [], []]

    # epoch
    for iteration in tqdm(range(100)):

        # permute
        perm = np.random.permutation(N)
        Xtrain = data.Xtrain[perm, :]
        Ytrain = data.Ytrain[perm, :]

        #####################
        ## Your code  here ##
        #####################
        # batches
        for j in range(N // Nbatch):

            indsBatch = range(j * Nbatch, (j+1) * Nbatch)
            X = Xtrain[indsBatch, :]
            Y = Ytrain[indsBatch, :]

            # Zero your gradients for every batch!
            optim.zero_grad()

            # the forward with the predict method from the model
            outputs = model(X)

            # using the functions: loss_accuracy
            batch_loss, _ = loss_accuracy(nn.CrossEntropyLoss(), outputs, Y)

            # the backward function with autograd
            batch_loss.backward()

            # and then an optimization step
            optim.step()


        ####################
        ##      FIN        #
        ####################

        Yhat_train = model(data.Xtrain)
        Yhat_test = model(data.Xtest)
        Ltrain, acctrain = loss_accuracy(nn.CrossEntropyLoss(), Yhat_train, data.Ytrain)
        Ltrain = Ltrain.item()
        Ltest, acctest = loss_accuracy(nn.CrossEntropyLoss(), Yhat_test, data.Ytest)
        Ltest = Ltest.item()

        title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(iteration, acctrain, Ltrain, acctest, Ltest)
        print(title)

        curves[0].append(acctrain)
        curves[1].append(acctest)
        curves[2].append(Ltrain)
        curves[3].append(Ltest)

    fig = plt.figure()
    plt.plot(curves[0], label="acc. train")
    plt.plot(curves[1], label="acc. test")
    plt.plot(curves[2], label="loss train")
    plt.plot(curves[3], label="loss test")
    plt.legend()
    plt.show()

main()