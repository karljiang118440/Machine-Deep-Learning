


'''
import tensorflow as tf
import keras
from keras.datasets import mnsit
from  keras.models import Sequential


'''


import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
'''
LeNet-5 
2018/7/15
ThomasZou

'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 输入数据为 mnist 数据集
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()

# 选取6个特征卷积核，大小为5∗5(不包含偏置),得到66个特征图，每个特征图的大小为32−5+1=2832−5+1=28，
# 也就是神经元的个数由10241024减小到了28∗28=78428∗28=784。
# 输入层与C1层之间的参数:6∗(5∗5+1)
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))

# 这一层的输入为第一层的输出，是一个28*28*6的节点矩阵。
# 本层采用的过滤器大小为2*2，长和宽的步长均为2，所以本层的输出矩阵大小为14*14*6。
model.add(MaxPooling2D(pool_size=(2, 2)))

# 本层的输入矩阵大小为14*14*6，使用的过滤器大小为5*5，深度为16.本层不使用全0填充，步长为1。
# 本层的输出矩阵大小为10*10*16。本层有5*5*6*16+16=2416个参数
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))

# 本层的输入矩阵大小10*10*16。本层采用的过滤器大小为2*2，长和宽的步长均为2，所以本层的输出矩阵大小为5*5*16。
model.add(MaxPooling2D(pool_size=(2, 2)))

# 本层的输入矩阵大小为5*5*16，在LeNet-5论文中将这一层称为卷积层，但是因为过滤器的大小就是5*5，#
# 所以和全连接层没有区别。如果将5*5*16矩阵中的节点拉成一个向量，那么这一层和全连接层就一样了。
# 本层的输出节点个数为120，总共有5*5*16*120+120=48120个参数。
model.add(Flatten())
model.add(Dense(120, activation='relu'))

# 本层的输入节点个数为120个，输出节点个数为84个，总共参数为120*84+84=10164个 (w + b)
model.add(Dense(84, activation='relu'))

# 本层的输入节点个数为84个，输出节点个数为10个，总共参数为84*10+10=850
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])