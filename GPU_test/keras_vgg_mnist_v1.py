import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

#导入表头
from keras.layers import Conv2D,MaxPooling2D,Flatten,Softmax,Activation,Dense
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from sklearn.metrics import recall_score,f1_score,precision_score
from keras import backend as K
K.set_image_dim_ordering('th')

#导入数据
num_classes=10
img_rows,img_cols=28,28
(trainX,trainY),(testX,testY)=mnist.load_data()
if K.image_data_format()=='channels_first':
    trainX=trainX.reshape(trainX.shape[0],1,img_rows,img_cols)
    testX=testX.reshape(testX.shape[0],1,img_rows,img_cols)
    input_shape=(1,img_rows,img_cols)
else:
    trainX=trainX.reshape(trainX.shape[0],img_rows,img_cols,1)
    testX=testX.reshape(testX.shape[0],img_rows,img_cols,1)
    input_shape=(img_rows,img_cols,1)

trainX=trainX.astype('float32')
testX=testX.astype('float32')
trainX /=255.0
testX /=255.0

trainY=keras.utils.to_categorical(trainY,num_classes)
testY=keras.utils.to_categorical(testY,num_classes)



#进行推理，参考如下格式

'''
#参考keras_lenet_mnist_v2.py   
    #layer_C1
model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=input_shape))

    #layer_S2
model.add(MaxPooling2D((2,2)))
#model.add(MaxPooling2D(pool_size(2,2)))

'''

model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(500, activation='relu'))   #model.add(Dense(500,activation='relu')),256 > 500
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#model.fit(trainX, trainY, batch_size=128, epochs=20),validation_data=(testX,testY))    #model.fit(trainX,trainY,batch_size=128,epochs=20,  << batch_size=32, epochs=10

model.fit(trainX,trainY,batch_size=128,epochs=20,validation_data=(testX,testY))  #model.fit(trainX,trainY,batch_size=128,epochs=20,  << batch_size=32, epochs=10,

score = model.evaluate(testX, testY)    #score=model.evaluate(testX,testY) <<  evaluate(testX, testY, batch_size=32)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

