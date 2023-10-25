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

    # params["Wh"] = torch.randn(nh, nx, device=device, dtype=dtype) * 0.3
    # params["Wy"] = torch.randn(ny, nh) * 0.3
    # params["bh"] = torch.zeros(1,nh) #broadcast
    # params["by"] = torch.zeros(1,ny) #broadcast
    #Initialize randomly
    params["Wh"] = torch.randn(nx, nh) * 0.3
    params["Wy"] = torch.randn(nh, ny) * 0.3
    #Initialize as zero vectors
    params["bh"] = torch.randn(nh) #* std
    params["by"] = torch.randn(ny) #* std #

    return params

def linear_affine_transformation(X,W, b):
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

   #####################
    ## Your code here  ##
    #####################
    # fill values for X, htilde, h, ytilde, yhat

    Wh = params['Wh']
    bh = params['bh']
    Wy = params['Wy']
    by = params['by']

    # Calculate htilde before activation of hidden layer --> make sure it works with batches
    # htilde = torch.matmul(X, Wh) + bh #for numpy: params['Wh'] @ X + params["bh"]
    htilde = torch.mm(X, Wh) + bh

    # Apply the activation function tanh to get h (hidden layer activation)
    h = torch.tanh(htilde)

    #h = torch.relu(htilde)

    # Calculate ytilde
    ytilde = torch.matmul(h, Wy) + by

    # Apply activation function for the output layer = softmax for multi-class classification
    yhat = torch.softmax(ytilde, dim=1)

    # yhat = ytilde
    # Store intermediate values in outputs dictionary = cache
    outputs["X"] = X #handover new X
    outputs["htilde"] = htilde
    outputs["h"] = h
    outputs["ytilde"] = ytilde
    outputs["yhat"] = yhat

    ##      END        #
    ####################

    return outputs['yhat'], outputs



def loss_accuracy(Yhat, Y):
    assert Yhat.size() == Y.size() # catch dimension errors early on
    epsilon = 1e-10

    #L = F.cross_entropy(Yhat, Y)
    L = -torch.sum(Y * torch.log(Yhat + epsilon)) / Y.size(0) # normalize by batch size, add epsilon to avoir log errors
    #L = - torch.mean(Y * torch.log(Yhat + epsilon))
    #L.requires_grad = True

    # Get the predicted class labels by taking the argmax along axis 1 (columns)
    _, indsY = torch.max(Yhat, 1)

    # Convert one-hot encoded true labels back to class indices
    _, inds_true = torch.max(Y, 1)

    # Compare the predicted labels with the true labels and calculate accuracy
    correct = torch.sum(indsY == inds_true)
    total = Y.size(0)  # Number of samples in the batch
    acc = correct.item() / total

    return L, acc



def backward(params, outputs, Y):
    bsize = Y.shape[0]
    grads = {}
    #####################
    ## Your code here  ##
    #####################
    # fill values for Wy, Wh, by, bh

    # Retrieve intermediate values from the forward pass
    X = outputs["X"]
    htilde = outputs["htilde"]
    h = outputs["h"]
    ytilde = outputs["ytilde"]
    yhat = outputs["yhat"]

    dWh = torch.zeros_like(params['Wh'])
    dbh = torch.zeros_like(params['bh'])
    dWy = torch.zeros_like(params['Wy'])
    dby = torch.zeros_like(params['by'])

    # Compute the gradient of the loss with respect to yhat (cross-entropy loss derivative)
    # dyhat = (yhat - Y) / bsize

    dyhat = yhat - Y

    # Compute the gradients for the output layer
    dWy += torch.matmul(h.t(), dyhat)  # dWy = h^T * dyhat
    dby += torch.sum(dyhat, dim=0)     # dby = sum(dyhat)

    # Compute the gradient for Wh and bh
    dh = torch.matmul(dyhat, params["Wy"].t())
    dhtilde = dh * (htilde > 0).float()  # Gradient for ReLU activation

    # Backpropagate the gradients to the hidden layer
    dh = torch.matmul(dyhat, params['Wy'].t())  # dh = dyhat * Wy^T
    dh *= (1 - h * h)  # Gradient of tanh activation function

    # Compute the gradients for the hidden layer
    dWh += torch.matmul(X.t(), dh)  # dWh = X^T * dh
    dbh += torch.sum(dh, dim=0)      # dbh = sum(dh)

    # Store the gradients in a dictionary
    grads = {
        'Wh': dWh,
        'bh': dbh,
        'Wy': dWy,
        'by': dby
    }

    ####################
    ##      END        #
    ####################
    return grads




def sgd(params, grads, eta):

    params["Wh"] = params["Wh"] - eta * grads["Wh"]
    params["Wy"] = params["Wy"] - eta * grads["Wy"]
    params["bh"] = params["bh"] - eta * grads["bh"]
    params["by"] = params["by"] - eta * grads["by"]

    return params





# init
data = CirclesData()
# data.plot_data()
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
        grads = backward(params, outputs, Y)

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
