# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:15:08 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split



def sigmoid(Z):
    """
    Compute the sigmoid of Z

    Arguments:
    Z - A scalar or numpy array of any size.

    Return:
    A - sigmoid(Z)
    """
   
    A = 1/(1+np.exp(-Z))
    
    return A

def sig_tag(A):
    Z = sigmoid(A)* (1 - sigmoid(A))
    
    return Z
    
def reLU(Z):
    """
    Compute the reLU of Z

    Arguments:
    Z - A scalar or numpy array of any size.

    Return:
    A - reLU(Z)
    """
    A = np.maximum(0,Z)
    
    return A

def reLU_deriv(A):
    Z = (A > 0) * 1
    return Z
    
def tanh(Z):
    """
    Compute the Hyperbolic Tagent of Z

    Arguments:
    Z - A scalar or numpy array of any size.

    Return:
    A - tanh(Z)
    """
    A = np.tanh(Z)

    return A

def tanh_tag(A):
    Z =1 - tanh(A)**2
    return Z


def init_parameters(Lin, Lout):
    """
    Init_parameters randomly initialize the parameters of a layer with Lin
    incoming inputs and Lout outputs 
    Input arguments: 
    Lin - the number of incoming inputs to the layer (not including the bias)
    Lout - the number of output connections 
    Output arguments:
    Theta - the initial weight matrix, whose size is Lout x Lin+1 (the +1 is for the bias).    
    Usage: Theta = init_parameters(Lin, Lout)
    
    """
    
    factor = np.sqrt(6/(Lin+Lout))
    Theta = np.zeros((Lout, Lin+1))
    Theta = 2*factor*(np.random.rand(Lout, Lin+1) - 0.5)
    
    
    return Theta
    

def ff_predict(Thetas, X, y, activation = reLU):
    """
    ff_predict employs forward propagation on a 3 layer networks and
    determines the labels of  the inputs 
    Input arguments
    Theta1 - matrix of parameters (weights)  between the input and the first hidden layer
    Theta2 - matrix of parameters (weights)  between the hidden layer and the output layer (or
          another hidden layer)
    X - input matrix
    y - input labels
    Output arguments:
    p - the predicted labels of the inputs
    Usage: p = ff_predict(Theta1, Theta2, X) 
    """
    theta_size = len(Thetas)
    m = X.shape[0] # num of samples
    num_outputs = Thetas[-1].shape[0]
    p = np.zeros((m,1)) #predictions for each sample.. 
  

    
    
    X_0 = np.ones( (X.shape[0], 1) ) 
    X1 = np.concatenate((X_0, X), axis = 1) # add bias col, each row an input 
    
    an = np.copy(X1)
    #for theta in Thetas:
    for i in range(theta_size - 1):
        #helper_func(an,theta)
        theta = Thetas[i]
        
        zn = np.dot(an, theta.T)
        an = activation(zn)
        
        ones = np.ones( (an.shape[0],1) )
        an = np.concatenate((ones,an),axis = 1)
   
    zL = np.dot( an, Thetas[-1].T)
    aL = sigmoid(zL)
    
    p = np.argmax(aL.T, axis=0) 
    p = p.reshape(p.shape[0],1) 
    detectp = np.sum( p==y ) / m * 100
    
    return p, detectp


def backprop(Thetas, X, y, activation = reLU, a_deriv = reLU_deriv, max_iter = 1000, alpha = 0.1, Lambda = 0):
    """
    backprop - BackPropagation for training a neural network
    Input arguments
    Thetas - list of thetas, each a matrix of parameters (weights)  between the
        theta_l and theta_l+1
    X - input matrix
    y - labels of the input examples
    max_iter - maximum number of iterations (epochs).
    alpha - learning coefficient.
    Lambda - regularization coefficient.
    
    Output arguments
    J - the cost function
    Thetas - list of updated weight matrix between the input and the first 
        hidden layer

    Usage:
    [J,Theta1,Theta2] = backprop(Theta1, Theta2, X,y,max_iter, alpha,Lambda)
    """

    m = X.shape[0] # num of samples
    num_outputs = Thetas[-1].shape[0]
    theta_size = len(Thetas)
    deltaL = np.zeros((num_outputs, 1)) 
    ybin = np.zeros(deltaL.shape)

    p = np.zeros((m, 1))
    J = 0
    acc_list = []
    J_list = []

    for q in range(max_iter):
       
        # ME'A'PES the cost and theta grads for each ITERATION      
        J = 0
                
        dTheta_l = []
        Theta_grad_l = []

        for i in range(theta_size):
            dTheta_l.append(np.zeros(Thetas[i].shape))
            Theta_grad_l.append(np.zeros(Thetas[i].shape))
            
        r = np.random.permutation(m)
 
        for k in range(m): # for each sample
            X1 = X[r[k], :] #pick one sample
            X1 = X1.reshape( 1, X1.shape[0] ) #make row
            
            # forward propogation BUT save zn, an
            X1_0 = np.ones( (X1.shape[0], 1) )
            X1 = np.concatenate((X1_0, X1), axis=1)
            X1 = X1.T 
            
            # ME'A'PES the zn, an for each SAMPLE

            an = np.copy(X1)
    #########################
            an_list = [np.copy(an)] 
            zn_l = [] 

            for j in range(theta_size - 1): 
                theta = Thetas[j]
                zn = np.dot(theta, an)
                an = activation(zn)
                # add bias to an
                ones = np.ones( (an.shape[1],1) )
                #print(an_list[0])
                an = np.concatenate((ones,an),axis = 0)
                #print(an_list[0]) 
                # add zn and an to lists
                zn_l.append(zn) 
                an_list.append(an)
            # for last col activation is always sigmoid 
            zL = np.dot( Thetas[-1], an) #maybe an_list[-1]
            zn_l.append(zL)
            aL = sigmoid(zL)
            an_list.append(aL)
            
            # backpropogation
            ybin = np.zeros(aL.shape)
            ybin[ y[r[k]], : ] = 1 #check y for samples labeled value
            
            
         
            J += (-1) * (np.dot(ybin.T, np.log(aL))) + (np.dot((1 - ybin).T, np.log(1 - aL))) 
            deltaL = (aL - ybin)
   
            delta_n_l = [deltaL] # going to go from deltaL ..to.. delta2
            
            for itera in range(theta_size-1):
                g_d_zn = a_deriv(zn_l[-(itera+2)])
                delta_n = np.dot(Thetas[-(itera+1)].T[1:,:], delta_n_l[-1])
                delta_n = delta_n * g_d_zn
                delta_n_l.append(delta_n)
                                 
            
            for iterat in range(theta_size):
                dTheta_l[iterat] = dTheta_l[iterat] + np.dot(delta_n_l[-(iterat+1)], an_list[iterat].T )
        theta_sum = 0

        for iterati in range(theta_size):
            theta_sum += np.sum(np.square(Thetas[iterati]))
        J = J/m + (Lambda/2)* theta_sum
        J_list.append(J[0,0])
        
        for iteratio in range(theta_size):
            Theta_grad_l[iteratio] = 1/m * dTheta_l[iteratio] + (Lambda * np.sum(Thetas[iteratio]))
 
        for iteration in range(theta_size):
            Thetas[iteration] = Thetas[iteration] - alpha * Theta_grad_l[iteration]
        p, acc = ff_predict(Thetas, X, y)
        acc_list.append(acc)
        if np.mod(q, max_iter/10) == 0: #do i need (int)(maxiter/10)?
            print('Cost function J = ', J, 'in iteration', q, 'acc training set = ', acc)

    plt.figure(1)
    plt.title("accuracy") 
    plt.plot(acc_list)
    plt.figure(2)
    plt.title("J cost") 
    plt.plot(J_list)

    plt.show()
    return J, Thetas

 

def try_digits():
    digits = datasets.load_digits()
    
    # flatten the images
    n_samples = len(digits.images) 
    data = digits.images.reshape((n_samples, -1)) 
    
    
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.3, shuffle=False
    )
    
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))
    
    L1 = X_train.shape[1] # number of tchunot
    num_outputs = np.unique(y_train).size
    num_hidden = 32
    Theta1 = init_parameters(L1, num_hidden)
    Theta2 = init_parameters(num_hidden, num_outputs)
    Thetas = [Theta1, Theta2]
    alpha = 0.1 # when num_hidden is 16 this can be bigger
    activation = reLU
    [J,Thetas] = backprop(Thetas, X_train, y_train, activation, reLU_deriv, 1000, alpha, 0.1)
    p, acc = ff_predict(Thetas, X_test, y_test)
    print('for ', len(Thetas) - 1 , 'hidden layers with ', num_hidden ,'nodes',
          ' with activation' , activation , 'alpha ' , alpha, 'acc for X_test = ' , acc)

def gimmel():
    
    from tensorflow.keras.datasets import mnist
    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    train_images_n = train_images[:7200,:]
    train_images_n = train_images_n.reshape((7200, 28*28))
    train_labels_n = train_labels[:7200,]
    train_labels_n = train_labels_n.reshape((train_labels_n.shape[0]),1)
    
    test_images_n = test_images[:3000,:]
    test_images_n = test_images_n.reshape((3000,28*28))
    test_labels_n = test_labels[:3000,]
    test_labels_n = test_labels_n.reshape((test_labels_n.shape[0]),1)
    
    #moved to 784 inputs, on 3000 images
    L1 = train_images_n.shape[1] # number of tchunot
    num_outputs = np.unique(train_labels_n).size
    num_hidden = 64
    
    Theta1 = init_parameters(L1, num_hidden)
    Theta2 = init_parameters(num_hidden, num_outputs)
    Thetas = [Theta1, Theta2]
    
    Theta1_h = init_parameters(L1, num_hidden)
    Theta2_h = init_parameters(num_hidden, num_hidden)
    Theta3_h = init_parameters(num_hidden, num_outputs)
    Thetas_hey = [Theta1_h, Theta2_h, Theta3_h]

    alpha = 0.01
    activation = reLU # sigmoid #  
    a_deriv = reLU_deriv # sig_tag # 
    Lambda = 0 #.00001
    max_iter = 100
    [J,Thetas_hey] = backprop(Thetas_hey, train_images_n, train_labels_n, activation, a_deriv, 
                          max_iter, alpha, Lambda)
    p, acc = ff_predict(Thetas_hey, train_images_n, train_labels_n, activation)
    print('for ', len(Thetas_hey) - 1 , 'hidden layers with ', num_hidden ,'nodes',
          ' with activation function' , activation.__name__ , 'alpha ' , alpha, 
          'Lambda ' , Lambda , ", max iter", max_iter, 'acc for X_test = ' , acc)
    

#for  1 hidden layers with  32 nodes  with activation function sigmoid 
#    alpha  0.1 Lambda  0 max iter 100 acc for X_test =  81.30555555555556

#for  1 hidden layers with  32 nodes  with activation function sigmoid 
#   alpha  0.1 Lambda  0 , max iter 1000 acc for X_test =  95.76388888888889

#for  1 hidden layers with  32 nodes  with activation function sigmoid 
#   alpha  0.1 Lambda  1e-06 , max iter 1000 acc for X_test =  92.94444444444444

#changed  lambda to 1  (and alpha to 0.01)
#    .py:235: RuntimeWarning: divide by zero encountered in log

#for  1 hidden layers with  64 nodes  with activation function sigmoid 
#   alpha  0.01 Lambda  0 , max iter 1000 acc for X_test =  89.61111111111111
#       could use more iterations

#for  1 hidden layers with  64 nodes  with activation function sigmoid 
#   alpha  0.1 Lambda  0 , max iter 1000 acc for X_test =  96.84722222222221

#for  1 hidden layers with  32 nodes  with activation function sigmoid 
#   alpha  0.001 Lambda  0.0001 , max iter 1000 acc for X_test =  45.33333333333333

#for activation function sigmoid alpha  0.1 Lambda  0.01 acc for X_test =  9.902777777777779
if __name__ == "__main__":
    gimmel()

    



