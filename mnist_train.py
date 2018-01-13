# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:08:43 2017

@author: chandler
"""
import numpy as np
import tensorflow.contrib.keras as keras
import tensorflow as tf

if __name__ == "__main__":
    batch_size = 128
    num_classes = 10
    epochs = 10
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('/home/tracking/work/src/course/mnist.npz')
#    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()    
    
#    if K.image_data_format() == 'channels_first':
#        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#        input_shape = (1, img_rows, img_cols)
#    else:
#        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
#    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
#    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
    
#    y_train = tf.one_hot(y_train,num_classes)
#    y_test = tf.one_hot(y_test,num_classes)
    
    y_train = np.eye(num_classes)[np.array(y_train).reshape(-1)]
    y_test = np.eye(num_classes)[np.array(y_test).reshape(-1)]
    
    #6000 28 28 1
#    print(x_train.shape)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.metrics.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
                  
    history1=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test))
    weight = model.get_weights();
                  
#    for i in range(len(model.layers)-2):
#        model.layers[i].trainable=False;
        
    model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    x_kana = np.load('x_kana.npy')
    y_kana = np.load('y_kana.npy')
    model.set_weights(weight)
    history2 = model.fit(x_kana,y_kana,batch_size = 1,epochs=100);
    np.savez('result12.npz',acc1=history1.history['acc'],loss1=history1.history['loss'],
             acc2=history2.history['acc'],loss2=history2.history['loss'])
    
    
        

