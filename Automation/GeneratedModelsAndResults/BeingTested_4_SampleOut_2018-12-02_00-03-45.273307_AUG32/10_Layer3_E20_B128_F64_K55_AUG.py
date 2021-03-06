#*****************************************
#Author : Hencil Peter
#File Name : 10_Layer3_E20_B128_F64_K55_AUG.py
#Timestamp : 2018-12-02 00:03:45.589113
#*****************************************

import keras
from keras import callbacks
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Softmax, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop
from keras import backend as K
import numpy as np
from datetime import datetime as dt
from keras.preprocessing.image import ImageDataGenerator

#Hyper parameters
batch_size=128
epochs=20
num_classes =10
trainingTime=''

#Preprocess the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_rows, img_cols = 32, 32
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0],  img_rows, img_cols, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Construct network Layers
model = Sequential()
model.add(Conv2D(64,kernel_size=(5,5),padding='same',input_shape=(32,32,3),use_bias='True',kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Conv2D(64,kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(64,kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

#Train the model

t1 = dt.now()
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=RMSprop(lr=0.0001, decay=1e-6),metrics=['accuracy'])
model_checkpoints = callbacks.ModelCheckpoint('/content/gdrive/My Drive/CNNProject/SampleOut_2018-12-02_00-03-45.273307/10_Layer3_E20_B128_F64_K55_AUG.py_weights_{epoch:02d}_{val_loss:.2f}_Proj.h5', monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

#Data Augmentation Enabled
datagen = ImageDataGenerator( rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,steps_per_epoch=32,
validation_data=(x_test, y_test),
workers=4)
t2 = dt.now()
delta = t2 - t1
trainingTime = str(delta.total_seconds())

#Evaluate the accuracy
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
fileEvaluation = open('/content/gdrive/My Drive/CNNProject/SampleOut_2018-12-02_00-03-45.273307/EvaluationReport.txt', 'a+')
fileEvaluation.write('\nFile: 10_Layer3_E20_B128_F64_K55_AUG.py\tAccuracy : ' + str(score[1]) + '\tLoss : ' + str(score[0]) + '\tTraining Time(S) : ' + trainingTime + '')
fileEvaluation.close()