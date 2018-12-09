#*****************************************
#Author : Hencil Peter
#File Name : 8_Layer3_E30_B128_F32_K33.py
#Timestamp : 2018-12-01 01:19:35.719181
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

#Hyper parameters
batch_size=128
epochs=30
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
model.add(Conv2D(32,kernel_size=(3,3),padding='same',input_shape=(32,32,3),use_bias='True',kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Conv2D(32,kernel_size=(3,3)))
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
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=SGD(lr=0.01),
metrics=['accuracy'])
model_checkpoints = callbacks.ModelCheckpoint('/content/gdrive/My Drive/CNNProject/SampleOut_2018-12-01_01-19-35.442350/8_Layer3_E30_B128_F32_K33.py_weights_{epoch:02d}_{val_loss:.2f}_Proj.h5', monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model_log = model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[model_checkpoints])
t2 = dt.now()
delta = t2 - t1
trainingTime = str(delta.total_seconds())

#Evaluate the accuracy
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
fileEvaluation = open('/content/gdrive/My Drive/CNNProject/SampleOut_2018-12-01_01-19-35.442350/EvaluationReport.txt', 'a+')
fileEvaluation.write('\nFile: 8_Layer3_E30_B128_F32_K33.py\tAccuracy : ' + str(score[1]) + '\tLoss : ' + str(score[0]) + '\tTraining Time(S) : ' + trainingTime + '')
fileEvaluation.close()