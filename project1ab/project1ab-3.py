# !wget https://github.com/rdfia/rdfia.github.io/raw/master/data/2-ab.zip
# !unzip -j 2-ab.zip
# !wget https://github.com/rdfia/rdfia.github.io/raw/master/code/2-ab/utils-data.py
import math
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import CirclesData
# %run 'utils-data.py'

device = torch.device("cpu")
dtype = torch.float

def init_model(nx, nh, ny):
    model = torch.nn.Sequential(
    torch.nn.Linear(nx, nh),
    torch.nn.Tanh(),
    torch.nn.Linear(nh, ny),
    # torch.nn.Sigmoid() # deleted because I am using CrossEntropyLoss that includes softmax
    )
    loss = torch.nn.CrossEntropyLoss()
    return model, loss


def loss_accuracy(Yhat, Y):
    L = 0
    acc = 0

    # Compute the cross-entropy loss
    L = -torch.mean(torch.sum(Y * torch.log(Yhat), dim=1))

    # Compute the precision (accuracy)
    _, indsY = torch.max(Y, 1)
    _, indsYhat = torch.max(Yhat, 1)
    correct_predictions = torch.sum(indsY == indsYhat)
    acc = torch.mean(correct_predictions.float())

    return L, acc



def sgd(model, eta):
    with torch.no_grad():
        for param in model.parameters():
            param -= eta * param.grad
        model.zero_grad()

# init
data = CirclesData()
data.plot_data()
N = data.Xtrain.shape[0]
Nbatch = 10
nx = data.Xtrain.shape[1]
nh = 10
ny = data.Ytrain.shape[1]
eta = 0.03

model , loss= init_model(nx, nh, ny)
print(model)

curves = [[], [], [], []]

# epoch
for iteration in range(150):

    # permute
    perm = np.random.permutation(N)
    Xtrain = data.Xtrain[perm, :]
    Ytrain = data.Ytrain[perm, :]

    for j in range(N // Nbatch):

        indsBatch = range(j * Nbatch, (j+1) * Nbatch)
        X = Xtrain[indsBatch, :]
        Y = Ytrain[indsBatch, :]

        # Batch forward pass
        Yhat = model(X)

        # computing the loss on the batch
        L = loss(Yhat, Y)

        # batch backward pass
        # grads = backward(params, outputs, Y) # deleted because I am using autograd
        L.backward() # compute the gradients

        # gradient descent
        model = sgd(model, eta)

    # computing and showing the loss and the accuracy on train and test
    Yhat_train, _ = forward(params, data.Xtrain)
    Yhat_test, _ = forward(params, data.Xtest)
    Ltrain, acctrain = loss_accuracy(Yhat_train, data.Ytrain)
    Ltest, acctest = loss_accuracy(Yhat_test, data.Ytest)
    Ygrid, _ = forward(params, data.Xgrid)

    # showing the decision boundary with plot_data_with_grid
    title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(
        iteration, acctrain, Ltrain, acctest, Ltest)
    print(title)
    data.plot_data_with_grid(Ygrid, title)

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
