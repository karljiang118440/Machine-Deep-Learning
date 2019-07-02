import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

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


'''
# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))

# 100张图片，每张100*100*3

y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

# 100*10

x_test = np.random.random((20, 100, 100, 3))

y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# 20*100

'''


path = os.path.join(os.getcwd(), 'train')
# test_data, test_labels = generate_data(path, 100)
test_data, test_labels = load_data('./train', (128, 128), 100)
test_data /= 255
images, labels = load_data(path, (128, 128), 10000)
images /= 255
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2)




model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3))) #input_shape=(100, 100, 3) > input_shape=(input_shape)
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100, activation='relu')) # Dense(256 > Dense(100
model.add(Dense(1, activation='sigmoid'))
model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid')) #Dense(10, activation='softmax') > Dense(1, activation='sigmoid')


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])



model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)  #categorical_crossentropy > sparse_categorical_crossentropy

#model.fit(x_train, y_train, batch_size=32, epochs=10)  # model.fit(x_train, y_train, batch_size=32, epochs=10) > next line "model.fit"

model.fit(x_train, y_train,
          epochs=15, batch_size=100, verbose=1,
          validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, batch_size=32)



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
with open('./vgg_kaggle_models/model_3.json', 'w') as model_file:
    model_file.write(model_json)
model.save_weights('./vgg_kaggle_models/cat_dog_3.h5')






