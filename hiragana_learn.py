# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 21:54:41 2018

@author: chandler
"""

import numpy as np

import tensorflow.contrib.keras as keras
import matplotlib.pyplot as plt

img_rows, img_cols = 32, 32
nb_classes = 72

hiragana_dataset = np.load('hiragana_dataset.npz')
X_train = hiragana_dataset['x_train']
X_test = hiragana_dataset['x_test']
Y_train = hiragana_dataset['y_train']
Y_test = hiragana_dataset['y_test']

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train*2-1
X_test = X_test*2-1

Y_train = np.eye(nb_classes)[np.array(Y_train).reshape(-1)]
Y_test = np.eye(nb_classes)[np.array(Y_test).reshape(-1)]

#===========================
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(nb_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=64), steps_per_epoch=144,
                    epochs=100, validation_data=(X_test, Y_test))
                    
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig('acc.png',transparent=True);
## "Loss"
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig('loss.png',transparent=True);
np.savez('result.npz',acc=history.history['acc'],val_acc=history.history['val_acc'],
         loss=history.history['loss'],val_loss=history.history['val_loss'])