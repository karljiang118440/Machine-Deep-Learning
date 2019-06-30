import keras
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Softmax,Activation,Dense
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from sklearn.metrics import recall_score,f1_score,precision_score
from keras import backend as K
K.set_image_dim_ordering('th')


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

model=Sequential()
    #layer_C1
model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=input_shape))

    #layer_S2
model.add(MaxPooling2D((2,2)))
#model.add(MaxPooling2D(pool_size(2,2)))


    #layer_C3
model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
    
    #layer_S4
#model.add(MaxPooling2D(pool_size(2,2)))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

    #layer_F6
model.add(Dense(500,activation='relu'))
    
    #layer_F7
model.add(Dense(num_classes,activation='softmax'))

'''
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])
'''


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

model.fit(trainX,trainY,batch_size=128,epochs=20,
validation_data=(testX,testY))

score=model.evaluate(testX,testY)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



