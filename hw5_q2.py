# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 18:35:14 2022

@author: Eden Akiva
"""

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

#input layer:
inputs = keras.Input(shape=(28, 28, 1))  #size of input in mnist
#now work with functional api (as opposed to sequential):
x = layers.Conv2D(filters=32, kernel_size=3, padding ='same', activation="relu")(inputs)
#filters kovim how many feature maps will be in yetziah
x = layers.MaxPooling2D(pool_size=2)(x)
#layer 2:
x = layers.Conv2D(filters=64, kernel_size=3, padding ='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
#layer 3, no pooling, so dim doesnt get too small:
x = layers.Conv2D(filters=64, kernel_size=3, padding ='same', activation="relu")(x)

# FULLY CONNECTED LAYER,  BUT WHY
x = layers.Dense(20, activation='relu')(x)

#flatten layer:
x = layers.Flatten()(x)
#output layer:, activation softmax
outputs = layers.Dense(10, activation="softmax")(x) #10 for digits

#somehow below gets all in btwn layers too: bc of x?
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

#cnn gets tensors- matrix that has depth dim or more(at least 3)
#depth = num of channels
#input shape doesnt include batch dimention - how many inputs we train on netork in each stage
#filters increase
#poolsize stays same - masnen


# ALEPH ################################
    
fashion_mnist = keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
print("X_train shape =", X_train.shape)
print("type of X_train = ", X_train.dtype)
# scale the pixel intensities down to 0-1 range by dividing them by 255.0
# this also converts them to floats.
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

# We will use a list of class names for Fashion MNIST to know the classes    
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model.compile(optimizer="rmsprop",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"])
#model compilation:

#model run:
model.fit(X_train, y_train, epochs=5, batch_size=64)


# BET ########################################################

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.3f}")

#Convolution_NN
#Epoch 5/5 loss: 0.2607 - accuracy: 0.9056
#Test accuracy: 0.888 - loss: 0.3111 - accuracy: 0.8885

#Fully_CN
#compared with fcn attempt at
#Epoch 5/5 - loss: 0.2910 - accuracy: 0.8919
# Test accuracy: 0.888 - loss: 0.3583 - accuracy: 0.8750

# fully_cn is much faster, probably bc we only have two hidden 
#   layers there, as opposed to 4 layers in conv_nn

# GIMMEL ###################################################

# padding ='valid' to each conv2

# Epoch 5/5 - loss: 0.2631 - accuracy: 0.9037
# Test accuracy: 0.892  - loss: 0.2977 - accuracy: 0.8916

# VS:  ##########
 
# padding ='same' to each conv2

#Epoch 5/5 - loss: 0.2200 - accuracy: 0.9203
#Test accuracy: 0.899  - loss: 0.2739 - accuracy: 0.8992


#CONCLUSION:
    
# with padding the cnn gets to a higher accuracy faster, but then plateaus
# at pretty much the same place... so with padding is better, but not by
# much in this case