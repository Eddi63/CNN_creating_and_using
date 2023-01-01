# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:59:52 2022

@author: Eden Akiva
"""

import numpy as np
import matplotlib.pyplot as plt

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

# assume mini batches of 32 or 64

def zero_pad(X, p):
    """
    zero_pad -zero padding all feature maps in a dataset X.
    The padding is carried out on the hight and width dimensions.
    Input Arguments:
    X - np array of shape (m, hight, width, n_c)
    p - integer, number of zeros to add on each side
    Returns:
    Xp - X after zero padding of size (m, hight + 2 * p, width + 2 * p, n_c)
    """
    Xp = np.pad(X, ((0,0),(p,p),(p,p),(0,0)), mode = 'constant', constant_values = (0,0))
    return Xp

Z = np.random.randint(0,10,(5,10,10, 3))
p = 1


Zp = zero_pad(Z, p)

im = (Zp[0,:,:,0])   
plt.imshow(im)         

for k in range(Zp.shape[0]): # in this case 5
    plt.imshow(Zp[k,:,:,2])
    plt.pause(0.1)

def conv(fmap_patch, filtMat, b):
    """
    conv - apply a dot product of one patch of a previous layer feature map
    Input Arguments:
    fmap_patch - patch of the input data of shape (f, f, n_c)
    filtMat - Weight parameters of shape (f, f, n_c)
    b - Bias parameters of shape (1, 1, 1)
    Returns:
    y - a scalar value, the result of convolving the sliding window (W, b) on a slice of the input data
    """
    z = np.sum(fmap_patch * filtMat)
    y = z + float(b)
    return y
#fmap_patch -(f, f, n_c)
#filtMat -(f, f, n_c)
#b -(1, 1, 1)
    
Z1 = np.ones((1, 3,3,3))
Z1[0, :, 1:,:] = 0     
plt.imshow(Z1[0, :, :, 0])
filtmap = np.ones((3,3,3))
filtmap[:,1,:] = 0
filtmap[:,1,:] = -1
b = 2*np.ones(1)
y = conv(Z1, filtmap, b )
print(y) 

def conv_forward(Fmap_input, filt_weights, b, p = 0, s = 1):
    """
    Forward propagation - convnet
    Input Arguments:
    Fmap_input - input feature maps (or output of previous layer),
    np array (m, n_H, n_W, n_C)
    m - number of input samples, n_H - hight, n_W - 'width', n_C - number of channels
    filt_weights - Weights, numpy array of shape (f, f, n_C, n_filt)
    b - bias, numpy array of shape (1, 1, 1, n_filt)
    p - padding parameter (default: p = 0), s - stride (default: s = 1)
    Returns:
    Fmap_output - output, numpy array of shape (m, n_H, n_W, n_filt) 
    """
    
    map_padded = zero_pad(Fmap_input, p)
    
    padded_h = map_padded.shape[1]
    padded_w = map_padded.shape[2]
    c = map_padded.shape[3]
    m = map_padded.shape[0]
    
    f = filt_weights.shape[0] #assume square
    n_f = filt_weights.shape[3]
    
    new_h = int(np.floor( (padded_h - f) / s) + 1)
    new_w = int(np.floor( (padded_w - f) / s) + 1)
    
    Fmap_output = np.zeros((m, new_h, new_w, n_f))
    for i_eg in range(m):
        for i_h in range(new_h):
            for i_w in range(new_w):
                for i_num_f in range(n_f):
                    Fmap_output[i_eg , i_h , i_w , i_num_f]  = \
                        conv(map_padded[i_eg , i_h:i_h+f, i_w:i_w+f,:], filt_weights[:,:,:,i_num_f], b[:,:,:,i_num_f]) 
                        # input= fmap_patch-(f, f, n_c)
                              #  filtMat -(f, f, n_c)
                              #  b -(1, 1, 1)
    #+f+1??
    return Fmap_output
       
#Fmap_input - (m, n_H, n_W, n_C)
#filt_weights -(f, f, n_C, n_filt)
#b -(1, 1, 1, n_filt)

#Fmap_output -(m, n_H, n_W, n_filt)
# do you mean NEW_H  and NEW_W ????

A_prev = np.random.randn(2, 5, 5, 3) # shape (m, n_H, n_W, n_C)
filterr = np.random.randn(3, 3, 3, 12) #filter a saved word??  #shape (f, f, n_C, n_filt)
b = np.random.randn(1, 1, 1, 12) #shape (1, 1, 1, n_filt)
A_next = conv_forward(A_prev, filterr, b)
print(A_next.shape)



