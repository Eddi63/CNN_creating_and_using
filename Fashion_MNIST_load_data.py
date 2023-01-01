# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:15:07 2022

@author: 
"""

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras 
from tensorflow.keras import layers

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000] #x_train shape (55000, 28, 28)
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:] #x_valid shape (5000, 28, 28)
print("X_train shape =", X_train.shape)
print("type of X_train = ", X_train.dtype)
# scale the pixel intensities down to 0-1 range by dividing them by 255.0
# this also converts them to floats.
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

# We will use a list of class names for Fashion MNIST to know the classes    
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# for example the first 5 images represents:
class_names[y_train[0]]

# presenting some data
for k in range(10):
    plt.figure(1, figsize = [5,5])
    plt.imshow(X_train[k,:,:], cmap = 'gray')
    plt.suptitle(class_names[y_train[k]])
    plt.pause(0.5)
    
    

model = keras.Sequential([
            #layers.Dense(784, activation="relu"), #input column always 784 entries
        layers.Dense(256, input_shape=(784, ), activation="relu") ,# kernel_regularizer=keras.regularizers.l2()), #X_train.shape[0]
        layers.Dense(256, activation="relu") ,# kernel_regularizer=keras.regularizers.l2()),
        layers.Dense(10, activation="softmax") #1 if face, 0 otherwise
    ])

# Compilation step

#opt = keras.optimizers.RMSprop(learning_rate=0.01) #default 0.001 
model.compile(optimizer= "rmsprop", # opt, #
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

X_train = X_train.reshape((55000, 28 * 28))
model.fit(X_train, y_train, epochs=5, batch_size=128) #128

################# VALIDATION
# Computing average accuracy over the VALIDATION set
X_valid = X_valid.reshape((5000, 28 * 28))
valid_loss, valid_acc = model.evaluate(X_valid, y_valid)

# elu and tanh ok but relu seems to work best
# as for nodes, the more the better it looks like

################ TEST
# Computing average accuracy over the TEST set
X_test = X_test.reshape((10000, 28 * 28))
test_loss, test_acc = model.evaluate(X_test, y_test)
                                     
                                     
                                     
                                     
                                     
                                    