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

def init_params(nx, nh, ny):
    params = {}

    params["Wh"] = torch.randn(nh, nx, device=device, dtype=dtype, requires_grad=True) * 0.3
    params["Wy"] = torch.randn(ny, nh, requires_grad=True) * 0.3
    params["bh"] = torch.zeros(nh, 1)
    params["by"] = torch.zeros(ny, 1)

    return params

def linear_affine_transformation(X,W, b):
    print("torch.mm(X, W.T)",torch.mm(X, W.T).shape, "\n", torch.mm(X, W.T))
    return torch.mm(X, W.T) + b


def activation_function_tanh(x):
    return torch.tanh(x)


def activation_function_softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=0)


def activation_function_sigmoid(x):
    return 1/(1+torch.exp(-x))


def activation_function_relu(x):
    return torch.max(torch.zeros(x.size()), x)



def forward(params, X):
    bsize = X.size(0)  # size nbatch * nx
    nh = params['Wh'].size(0)  # matrix of weights Wh of size nh × nx
    ny = params['Wy'].size(0)  # bias vector bh of size nh
    outputs = {}

    outputs["X"] = X
    outputs["htilde"] = linear_affine_transformation(X,
        params["Wh"], params["bh"])
    outputs["h"] = activation_function_tanh(outputs["htilde"])
    print("Wy", params["Wy"].shape, "\n", params["Wy"])
    print("by\n",params["by"].shape, "\n",  params["by"])
    outputs["ytilde"] = linear_affine_transformation(
        outputs["h"], params["Wy"], params["by"])
    outputs["yhat"] = activation_function_softmax(outputs["ytilde"])

    return outputs['yhat'], outputs



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



def sgd(params, eta):
    with torch.no_grad():

        params["Wh"] -= eta * params["Wh"].grad
        params["Wy"] -= eta * params["Wy"].grad
        params["bh"] -= eta * params["bh"].grad
        params["by"] -= eta * params["by"].grad
        #reset gradient to zero
        params["Wh"].grad.zero_()
        params["Wy"].grad.zero_()
        params["bh"].grad.zero_()   
        params["by"].grad.zero_()

    return params





# init
data = CirclesData()
data.plot_data()
N = data.Xtrain.shape[0]
Nbatch = 10
nx = data.Xtrain.shape[1]
nh = 10
ny = data.Ytrain.shape[1]
eta = 0.03

params = init_params(nx, nh, ny)
print(params)

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
        Yhat, outputs = forward(params, X)

        # computing the loss on the batch
        L, acc = loss_accuracy(Yhat, Y)

        # batch backward pass
        # grads = backward(params, outputs, Y) # deleted because I am using autograd
        L.backward() # compute the gradients

        # gradient descent
        params = sgd(params, grads, eta)

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
