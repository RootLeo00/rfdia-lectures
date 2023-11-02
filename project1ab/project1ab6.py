import math
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import CirclesData

import sklearn.svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# data
def load_dataset_circles():
    data = CirclesData()
    Xtrain = data.Xtrain.numpy()
    Ytrain = data.Ytrain[:, 0].numpy()

    Xgrid = data.Xgrid.numpy()

    Xtest = data.Xtest.numpy()
    Ytest = data.Ytest[:, 0].numpy()
    return Xtrain, Ytrain, Xgrid, Xtest, Ytest

def plot_svm_predictions(data, predictions):
      plt.figure(2)
      plt.clf()
      plt.imshow(np.reshape(predictions, (40,40)))
      plt.plot(data._Xtrain[data._Ytrain[:,0] == 1,0]*10+20, data._Xtrain[data._Ytrain[:,0] == 1,1]*10+20, 'bo', label="Train")
      plt.plot(data._Xtrain[data._Ytrain[:,1] == 1,0]*10+20, data._Xtrain[data._Ytrain[:,1] == 1,1]*10+20, 'ro')
      plt.plot(data._Xtest[data._Ytest[:,0] == 1,0]*10+20, data._Xtest[data._Ytest[:,0] == 1,1]*10+20, 'b+', label="Test")
      plt.plot(data._Xtest[data._Ytest[:,1] == 1,0]*10+20, data._Xtest[data._Ytest[:,1] == 1,1]*10+20, 'r+')
      plt.xlim(0,39)
      plt.ylim(0,39)
      plt.clim(0.3,0.7)
      plt.draw()
      plt.pause(1e-3)



def linearSVM():

    ############################
    ### Your code here   #######
    ### Train the SVM    #######
    ## See https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    ## and https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    ############################

    Xtrain, Ytrain, Xgrid, Xtest, Ytest = load_dataset_circles()

    clf = LinearSVC(random_state=0, tol=1e-5)
    svm = clf.fit(Xtrain, Ytrain)

    ###########################

    ## Print results
    Ytest_pred = svm.predict(Xtest)
    accuracy = np.sum(Ytest == Ytest_pred) / len(Ytest)
    print(f"Accuracy : {100 * accuracy:.2f}%")
    Ygrid_pred = svm.predict(Xgrid)

    plot_svm_predictions(data, Ygrid_pred)



def nonLinearSVM():

    Xtrain, Ytrain, Xgrid, Xtest, Ytest = load_dataset_circles()
    
    clf = SVC(gamma='auto')
    svm_nonlinear = clf.fit(Xtrain, Ytrain)

    ## Print results for non-linear SVM
    Ytest_pred = svm_nonlinear.predict(Xtest)
    accuracy = np.sum(Ytest == Ytest_pred) / len(Ytest)
    print(f"Accuracy : {100 * accuracy:.2f}%")
    Ygrid_pred = svm_nonlinear.predict(Xgrid)

    plot_svm_predictions(data, Ygrid_pred)


def main():
    linearSVM()
    nonLinearSVM()