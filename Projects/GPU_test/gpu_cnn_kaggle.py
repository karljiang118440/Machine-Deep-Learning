# -*- coding: utf-8 -*-
"""
Created on Thu May  3 22:52:05 2018
@author: xiaozhe
"""
import os
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import logging
from resize_pic import generate_data, load_data
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import model_from_yaml, load_model, model_from_json
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(message)s')

path = os.path.join(os.getcwd(), 'train')
# test_data, test_labels = generate_data(path, 100)
test_data, test_labels = load_data('./train', (128, 128), 100)
test_data /= 255
images, labels = load_data(path, (128, 128), 10000)
images /= 255
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2)

model = Sequential()

# the first convolution layer
model.add(Conv2D(filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                input_shape=(128, 128, 3),
                padding='same',
                data_format="channels_last"))
kernel_size = (3, 3),
strides = (1, 1),
activation = 'relu',
padding = 'same'
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# the second convolution layer
model.add(Conv2D(filters=64,
kernel_size = (3, 3),
strides = (1, 1),
activation = 'relu',
padding = 'same'))
# model.add(Conv2D(filters=64,
kernel_size = (3, 3),
strides = (1, 1),
activation = 'relu',
padding = 'same'
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# the third convolution layer
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation='relu',
                 padding='same'))
# model.add(Conv2D(filters=128,
kernel_size = (3, 3),
strides = (1, 1),
activation = 'relu',
padding = 'same'
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# the output layer
model.add(Flatten())
model.add(Dense(100, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
sgd = Adam(lr=0.001)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=15, batch_size=100, verbose=1,
          validation_data=(x_test, y_test))

predict_value = model.predict(test_data, verbose=1)
animals = np.array(['cat', 'dog'])
accurate = 0
for i in range(len(predict_value)):
    predict_temp = 0
if predict_value[i, 0] > 0.5:
    predict_temp = 1
if animals[predict_temp] == animals[test_labels[i, 0]]:
    accurate += 1
#    plt.imshow(test_data[i].astype(int))
print('predict: ', animals[predict_temp],
      'actual: ', animals[test_labels[i, 0]], sep=' ')
print('accurate: ', accurate / predict_value.shape[0])

model_json = model.to_json()
with open('./models/model_3.json', 'w') as model_file:
    model_file.write(model_json)
model.save_weights('./models/cat_dog_3.h5')


def load_cnnmodel():
    with open('models/model_2.json', 'r') as model_js:
        model_loaded = model_js.read()

    model = model_from_json(model_loaded)
    model.load_weights('models/cat_dog_2.h5')
    sgd = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

