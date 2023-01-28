from array import array
from tensorflow import keras
import tensorflow as tf
#from tensorflow. import mnist
from tensorflow.keras.datasets import mnist
import cvHelper
import kerasHelper
import numpy as np
import initdata as init
import os
import cv2
from keras.preprocessing.image import image_utils
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


template, defects = init.initdata()
#(train_images_o, train_labels), (test_images_o, test_labels) =

folders = []


train_images_o = np.empty([80,288,352], dtype='int8')
train_labels =  np.empty([80, 1])
test_images_o =  np.empty([20, 288, 352], dtype='int8')
test_labels = np.empty([20, 1])


directory = '..\..\img'

cnt = 0
i_train = 0
cnt_train = 0
cnt_test = 0

for folder in os.listdir(directory):
    f = os.path.join(directory, folder)


    if os.path.isfile(f):
        print(f)
    else:
        folders.append(f)
        #print(folder)
        if folder == '0-Normal':
            n_train = 24
        else: 
            n_train = 8

        for pic in os.listdir(f):
            if cnt == 100:
                break
            p = os.path.join(f, pic)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)


            if cnt_train < n_train:
                train_images_o[i_train] = img
                train_labels[i_train] = ord(folder[0]) -48
                cnt_train = cnt_train + 1
                i_train = i_train + 1
            else:
                test_images_o[cnt_test] = img
                test_labels[cnt_test] = ord(folder[0]) -48 
                cnt_test = cnt_test + 1
            #print(p)
            cnt = cnt + 1
    cnt_train = 0



print(train_images_o.ndim)
print(train_images_o.shape)
print(train_images_o.dtype)

#cvHelper.imagesc(train_images_o, 'test')

#train_imgs = train_images_o.reshape((84, 288*352))
train_imgs = train_images_o
train_imgs = train_imgs.astype('float32') / 255

print(train_imgs.shape)

#test_imgs = test_images_o.reshape((16, 288*352))
test_imgs = test_images_o
test_imgs = test_imgs.astype('float32') / 255

#train_labels.astype('float32')
#test_labels.astype('float32')

from keras import models, layers
# network = models.Sequential()

# network.add(layers.Dense(128, activation='sigmoid', input_shape=(288*352,)))
# network.add(layers.Dense(56, activation = 'softmax'))

# network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# from keras.utils import plot_model
# dot_img_file='./model.png'
# plot_model(network, to_file=dot_img_file, show_shapes=True, show_layer_activations=True)


# print(train_labels)

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# network.fit(train_imgs, train_labels, epochs=12, batch_size=128)
# test_loss, test_acc = network.evaluate(test_imgs, test_labels)

s = 4

# network = models.Sequential()
# network.add(layers.Conv2D(s, (3,3), activation='relu', input_shape=(288,352,1)))
# network.add(layers.Conv2D(2*s, (3,3), activation='relu'))
# network.add(layers.MaxPooling2D((2,2)))
# network.add(layers.Conv2D(4*s, (3,3), activation='relu'))
# network.add(layers.Conv2D(4*s, (3,3), activation='relu'))
# network.add(layers.MaxPooling2D((2,2)))
# network.add(layers.Flatten())
# #network.add(layers.Dense(16*s, activation='relu'))
# #network.add(layers.Dropout(0.5))         # sets input units randomly to zero while training with a frequency of 0.5
# network.add(layers.Dense(8, activation='sigmoid'))
# print model architecture
network = models.Sequential()
network.add(layers.Conv2D(s, (10,10), activation='relu', input_shape=(288,352,1)))
network.add(layers.Conv2D(s, (4,4), activation='relu'))
network.add(layers.MaxPooling2D((3,3)))
network.add(layers.Conv2D(4*s, (2,2), activation='tanh'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Flatten())
network.add(layers.Dense(8, activation='softmax'))

network.summary()

network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])
# know train the model using trainings data
history = network.fit(train_imgs, train_labels, epochs=20, batch_size=8)




print("============================")
print("=== TEST ===================")
print("============================")

test_loss, test_acc = network.evaluate(test_imgs, test_labels)
print(test_loss)
print(test_acc)

print("============================")
print("=== PREDICT ===================")
print("============================")

#print(test_imgs)
pred = network.predict(test_imgs)
pred = pred * 100
np.set_printoptions(precision=0, suppress=True)
print(pred)

# import kerasHelper
# kerasHelper.plotAcc(history, smooth=True)
# kerasHelper.plotLoss(history, smooth=True)

