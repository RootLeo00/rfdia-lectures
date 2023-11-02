import math
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import CirclesData
from tqdm import tqdm

def init_params(nx, nh, ny, std = 0.3, mean = 0):
    """
    nx, nh, ny: integers
    out params: dictionnary
    """
    params = {}

    #####################
    ## Your code here  ##
    #####################
    # fill values for Wh, Wy, bh, by

    # requires_grad = False because of manual backprob. This is the default for new tensors.
    # all weights will be initialized according to a normal distribution of mean 0 and standard deviation 0.3.

    #Initialize randomly
    params["Wh"] = torch.randn(nx, nh) * std
    params["Wy"] = torch.randn(nh, ny) * std

    #Initialize as zero vectors
    params["bh"] = torch.randn(nh) #* std
    params["by"] = torch.randn(ny) #* std #

    ####################
    ##      END        #
    ####################

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

    #####################
    ## Your code here  ##
    #####################
    # update the params values

    # batch version of sgd

    params["Wh"] -= eta * grads["Wh"]
    params["Wy"] -= eta * grads["Wy"]
    params["bh"] -= eta * grads["bh"]
    params["by"] -= eta * grads["by"]

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
    print('nx = ', nx)
    nh = 10
    ny = data.Ytrain.shape[1]
    print('ny = ', ny)
    eta = 0.02

    params = init_params(nx, nh, ny)
    print(params)
    curves = [[],[], [], []]

    iterations = 100
    # epoch
    for iteration in tqdm(range(iterations)):

        # permute
        perm = np.random.permutation(N)
        Xtrain = data.Xtrain[perm, :]
        Ytrain = data.Ytrain[perm, :]

        print(Ytrain.shape)

        #####################
        ## Your code here  ##
        #####################
        # batches

        for j in range(N // Nbatch):

            indsBatch = range(j * Nbatch, (j+1) * Nbatch)

            X = torch.tensor(Xtrain[indsBatch, :], dtype=torch.float32)
            Y = torch.tensor(Ytrain[indsBatch, :], dtype=torch.float32)

            # Forward pass
            Yhat, cache = forward(params, X)

            # Compute loss and accuracy
            loss, accuracy = loss_accuracy(Yhat, Y)

            # Backward pass
            grads = backward(params, cache, Y)

            # Update parameters using SGD
            params = sgd(params, grads, eta)

        ####################
        ##      END        #
        ####################

        Yhat_train, _ = forward(params, data.Xtrain)
        Yhat_test, _ = forward(params, data.Xtest)
        Ltrain, acctrain = loss_accuracy(Yhat_train, data.Ytrain)
        Ltest, acctest = loss_accuracy(Yhat_test, data.Ytest)
        Ygrid, _ = forward(params, data.Xgrid)

        title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(iteration, acctrain, Ltrain, acctest, Ltest)
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
    plt.grid(True)
    plt.legend()
    plt.show()

main()