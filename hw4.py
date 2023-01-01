# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:20:41 2022

@author: Eden Akiva
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras 
from tensorflow.keras import layers

def showvec(X): #,Y):
    fig, axs = plt.subplots(5, 5)
    plt.subplots_adjust(wspace=0, hspace=0.1)
    r = c = 0
    for i in range(X.shape[1]):
        Z = X[:,i]
        Z = Z.reshape((28,28))
        axs[r,c].imshow(Z.T, cmap='gray')
        axs[r,c].axis('off')
        c+=1
        c = c%5
        if (c == 0):
            r+=1
        
    
df = pd.read_excel('face_data.xlsx') #10,000 pics, 3,000 faces, 7,000 not
M = df.to_numpy()
MT = M.T #just to change things up?
X = MT[:-1,:]
Y = MT[784,:]
X_train = MT[:-1,1000:5000] #2,000 faces and 2,000 others
Y_train = MT[784,1000:5000]
Y_train[Y_train==-1]=0
Y_train = Y_train.reshape(1,Y_train.shape[0])

X_test = np.concatenate((MT[:-1, :1000] , MT[:-1, 5000:]), axis=1)

Y_test_a = MT[784, :1000]
Y_test_b = MT[784, 5000:]
Y_test = np.concatenate((Y_test_a , Y_test_b), axis=None)
Y_test[Y_test==-1]=0
Y_test = Y_test.reshape(1, Y_test.shape[0])

showvec(X_train[:,:25])
showvec(X_train[:, -25:])


####################### DALED ################################

model = keras.Sequential([
            #layers.Dense(X_train.shape[0], activation="relu"), #input column always 784 entries
        layers.Dense(784, input_shape=(X_train.shape[0],), activation="relu") ,# kernel_regularizer=keras.regularizers.l2()), #X_train.shape[0]
        layers.Dense(1, activation="sigmoid") #1 if face, 0 otherwise
    ])

# Compilation step

opt = keras.optimizers.RMSprop(learning_rate=0.01) #default 0.001 
model.compile(optimizer= opt, # "rmsprop", #
              loss= "binary_crossentropy",
              metrics=["accuracy"])


model.fit(X_train.T, Y_train.T, epochs=5, batch_size=12) #128

# Computing average accuracy over the entire test set
test_loss, test_acc = model.evaluate(X_test.T, Y_test.T)#.flatten())
print('test_acc = ', test_acc)
# test_acc =  0.9156526327133179

############################### VAV ##################################
# alpha = 0.01 and regulizer l2  seem to have best results sooo:
    #on real test
    
from scipy.io import loadmat

data_set = loadmat('face_test.mat')
#print(data_set)
#print(data_set.keys())

X_final = data_set['Xtest'] #5000,784
Y_final = data_set['ytest'] #5000,1
Y_final[Y_final==-1]=0
finalt_loss, finalt_acc = model.evaluate(X_final, Y_final)

#- loss: 0.2070 - accuracy: 0.9408

################################ ZAIN ******************************


model_2 = keras.Sequential([
        layers.Dense(X_train.shape[0], input_shape=(X_train.shape[0],), activation="relu"), #input column always 784 entries
        layers.Dense(784, activation="relu" ),#  kernel_regularizer=keras.regularizers.l1()),
            #layers.Dense(784, activation="relu" ),# kernel_regularizer=keras.regularizers.l1()), 
        layers.Dense(1, activation="sigmoid") #1 if face, 0 otherwise
    ])

# Compilation step

#opt_2 = keras.optimizers.RMSprop(learning_rate=0.01) #default 0.001 
model_2.compile(optimizer= "rmsprop", #opt_2, # 
              loss= "binary_crossentropy",
              metrics=["accuracy"])


model_2.fit(X_train.T, Y_train.T, epochs=5, batch_size=12) #128

# Computing average accuracy over the entire test set
test_loss_2, test_acc_2 = model_2.evaluate(X_test.T, Y_test.T)

finalt_loss_2, finalt_acc_2 = model_2.evaluate(X_final, Y_final)

#cross -loss: 0.2449 - accuracy: 0.9195
#final - loss: 0.2922 - accuracy: 0.8918

####################### CHET ################################


model_3 = keras.Sequential([
        #   layers.Dense(X_train.shape[0], activation="relu"), #input column always 784 entries
        layers.Dense(64, input_shape=(X_train.shape[0],), activation="relu" ),#  kernel_regularizer=keras.regularizers.l1()),
        layers.Dense(1, activation="sigmoid") #1 if face, 0 otherwise
    ])

# Compilation step

#opt_3 = keras.optimizers.RMSprop(learning_rate=0.01) #default 0.001 
model_3.compile(optimizer= "rmsprop", #opt_3, # 
              loss= "binary_crossentropy",
              metrics=["accuracy"])


model_3.fit(X_train.T, Y_train.T, epochs=5, batch_size=12) #138

# Computing average accuracy over the entire test set
test_loss_3, test_acc_3 = model_3.evaluate(X_test.T, Y_test.T)

finalt_loss_3, finalt_acc_3 = model_3.evaluate(X_final, Y_final)

#batch 128
# cross accuracy: 0.8996
# final accuracy: 0.1654

#batch 12:
# cross - loss: 1.5268 - accuracy: 0.4254
# final - loss: 1.2947 - accuracy: 0.4934

############ VS


model_4 = keras.Sequential([
            #layers.Dense(X_train.shape[0], activation="relu"), #input column always 784 entries
        layers.Dense(16, input_shape=(X_train.shape[0],), activation="relu" ),#  kernel_regularizer=keras.regularizers.l1()),
        layers.Dense(16, activation="relu" ),
        layers.Dense(16, activation="relu" ),
        layers.Dense(8, activation="relu" ),
        layers.Dense(8, activation="relu" ),
        layers.Dense(1, activation="sigmoid") #1 if face, 0 otherwise
    ])

# Compilation step

#opt_4 = keras.optimizers.RMSprop(learning_rate=0.01) #default 0.001 
model_4.compile(optimizer= "rmsprop", #opt_4, # 
              loss= "binary_crossentropy",
              metrics=["accuracy"])

model_4.fit(X_train.T, Y_train.T, epochs=5, batch_size=12) #148

# Computing average accuracy over the entire test set
test_loss_4, test_acc_4 = model_4.evaluate(X_test.T, Y_test.T)

finalt_loss_4, finalt_acc_4 = model_4.evaluate(X_final, Y_final)
#batch 128
# cross - loss: 0.2318 - accuracy: 0.9183
# final - loss: 0.2364 - accuracy: 0.9094

#batch 12:
# cross - loss: 0.2168 - accuracy: 0.9260
# final - loss: 0.1558 - accuracy: 0.9446

    