import math
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import CirclesData

def init_params(nx, nh, ny):
    """
    nx, nh, ny: integers
    out params: dictionnary
    """
    params = {}

    #####################
    ## Your code here  ##
    #####################
    # fill values for Wh, Wy, bh, by
    # activaye autograd on the network weights

    params["Wh"] = torch.randn(nx, nh, requires_grad=True)
    params["Wy"] = torch.randn(nh, ny, requires_grad=True)
    params["bh"] = torch.randn(nh, requires_grad=True)
    params["by"] = torch.randn(ny, requires_grad=True)

    ####################
    ##      END        #
    ####################
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
    """
    params: dictionnary
    X: (n_batch, dimension)
    """
    bsize = X.shape[0]

    #Use shape instead of size
    nh = params['Wh'].shape[0]  #nh = params['Wh'].size(0)
    ny = params['Wy'].shape[0]  #ny = params['Wy'].size(0)
    nx = X.shape[-1]

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
    htilde = torch.mm(X, Wh) + bh

    # Apply the activation function tanh to get h (hidden layer activation)
    h = torch.tanh(htilde)

    # Calculate ytilde
    ytilde = torch.matmul(h, Wy) + by

    # Apply activation function for the output layer = softmax for multi-class classification
    yhat = torch.softmax(ytilde, dim=1)

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

    L = -torch.sum(Y * torch.log(Yhat + epsilon)) / Y.size(0) # normalize by batch size, add epsilon to avoir log errors

    # Get the predicted class labels by taking the argmax along axis 1 (columns)
    _, indsY = torch.max(Yhat, 1)

    # Convert one-hot encoded true labels back to class indices
    _, inds_true = torch.max(Y, 1)

    # Compare the predicted labels with the true labels and calculate accuracy
    correct = torch.sum(indsY == inds_true)
    total = Y.size(0)  # Number of samples in the batch
    acc = correct.item() / total

    return L, acc


def sgd(params, eta):
    #####################
    ## Your code here  ##
    #####################
    # update the network weights
    # warning: use torch.no_grad()
    # and reset to zero the gradient accumulators
    # Update the network weights using gradient descent

    with torch.no_grad():
        for param in params.values():
            param -= eta * param.grad
            param.grad.zero_()

    ####################
    ##      END        #
    ####################
    return params




def main():
    # init
    data = CirclesData()
    data.plot_data()
    N = data.Xtrain.shape[0]
    Nbatch = 16
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.03

    params = init_params(nx, nh, ny)

    curves = [[],[], [], []]

    # epoch
    for iteration in range(100):

        # permute
        perm = np.random.permutation(N)
        Xtrain = data.Xtrain[perm, :]
        Ytrain = data.Ytrain[perm, :]

        #####################
        ## Your code here  ##
        #####################

        # Initialize accumulators for training loss and accuracy
        train_loss = 0.0
        train_accuracy = 0.0

        for j in range(N // Nbatch):

            indsBatch = range(j * Nbatch, (j+1) * Nbatch)
            X = torch.tensor(Xtrain[indsBatch, :])  # Convert to PyTorch tensor
            Y = torch.tensor(Ytrain[indsBatch, :])  # Convert to PyTorch tensor

            # Forward pass
            Yhat, cache = forward(params, X)

            # Calculate the loss and accuracy
            loss, accuracy = loss_accuracy(Yhat, Y)

            loss.backward()

            # Update the network weights using SGD
            params = sgd(params, eta)

            # Accumulate training loss and accuracy
            train_loss += loss.item()
            train_accuracy += accuracy

            # Reset gradients to zero
            for param in params.values():
                param.grad.zero_()

        # Calculate average training loss and accuracy
        train_loss /= (N // Nbatch)
        train_accuracy /= (N // Nbatch)


        ####################
        ##      END        #
        ####################

        # Evaluate on the test set
        Yhat_train, _ = forward(params, data.Xtrain)
        Yhat_test, _ = forward(params, data.Xtest)
        Ltrain, acctrain = loss_accuracy(Yhat_train, data.Ytrain)
        Ltrain = Ltrain.item()
        Ltest, acctest = loss_accuracy(Yhat_test, data.Ytest)
        Ltest = Ltest.item()
        Ygrid, _ = forward(params, data.Xgrid)

        title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(iteration, acctrain, Ltrain, acctest, Ltest)
        print(title)
        # detach() is used to remove the predictions from the computational graph in autograd
        data.plot_data_with_grid(Ygrid.detach(), title)

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