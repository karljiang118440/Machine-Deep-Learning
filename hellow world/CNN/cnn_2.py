#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/21/2018 11:21 AM
# @Author  : SkullFang
# @Contact : yzhang.private@gmail.com
# @File    : CNN_demo1.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf

#download mnist datasets
#55000 * 28 * 28 55000image
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('mnist_data',one_hot=True)#����һ���ļ�Ŀ¼�����������Ƿ�Ϊone_hot����

#one_hot is encoding format
#None means tensor �ĵ�һά�ȿ���������ά��
#/255. ����һ��
input_x=tf.placeholder(tf.float32,[None,28*28])/255.
#�����һ��one hot������
output_y=tf.placeholder(tf.int32,[None,10])

#����� [28*28*1]
input_x_images=tf.reshape(input_x,[-1,28,28,1])
#��(Test)���ݼ���ѡȡ3000����д���ֵ�ͼƬ�Ͷ�Ӧ��ǩ

test_x=mnist.test.images[:3000] #image
test_y=mnist.test.labels[:3000] #label



#���ز�
#conv1 5*5*32
#layers.conv2d parameters
#inputs ���룬��һ������
#filters ����˸�����Ҳ���Ǿ����ĺ��
#kernel_size ����˵ĳߴ�
#strides: ɨ�貽��
#padding: �߲߱�0 valid����Ҫ��0��same��Ҫ��0��Ϊ�˱�֤��������ĳߴ�һ��,�����ٲ���Ҫ֪��
#activation: �����
conv1=tf.layers.conv2d(
    inputs=input_x_images,
    filters=32,
    kernel_size=[5,5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
print(conv1)

#�������� [28*28*32]

#pooling layer1 2*2
#tf.layers.max_pooling2d
#inputs ���룬��������Ҫ���ĸ�ά��
#pool_size: �������ĳߴ�

pool1=tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2,2],
    strides=2
)
print(pool1)
#��������[?,14,14,32]

#conv2 5*5*64
conv2=tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5,5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)

#��������  [?,14,14,64]

#pool2 2*2
pool2=tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2,2],
    strides=2
)

#��������[?,7,7,64]

#flat(ƽ̹��)
flat=tf.reshape(pool2,[-1,7*7*64])


#��״�����[?,3136]

#densely-connected layers ȫ���Ӳ� 1024
#tf.layers.dense
#inputs: ����
#units�� ��Ԫ�ĸ���
#activation: �����
dense=tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)

#��������[?,1024]
print(dense)

#dropout
#tf.layers.dropout
#inputs ����
#rate ������
#training �Ƿ�����ѵ����ʱ����
dropout=tf.layers.dropout(
    inputs=dense,
    rate=0.5,
)
print(dropout)

#����㣬���ü���������ʾ���һ��ȫ���Ӳ㣩
logits=tf.layers.dense(
    inputs=dropout,
    units=10
)
#�����״[?,10]
print(logits)

#������� cross entropy�������أ�������Softmax����ٷֱȵĸ���
#tf.losses.softmax_cross_entropy
#onehot_labels: ��ǩֵ
#logits: ����������ֵ
loss=tf.losses.softmax_cross_entropy(onehot_labels=output_y,
                                     logits=logits)
# ��Adam �Ż�������С�����,ѧϰ��0.001 �����ݶ��½�
print(loss)
train_op=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


#���ȡ�����Ԥ��ֵ��ʵ�ʱ�ǩ��ƥ��̶�
#tf.metrics.accuracy
#labels����ʵ��ǩ
#predictions: Ԥ��ֵ
#Return: (accuracy,update_op)accuracy ��һ������׼ȷ�ʣ�update_op ��һ��op����������ȡ�
#���������Ǿֲ�����
accuracy_op=tf.metrics.accuracy(
    labels=tf.argmax(output_y,axis=1),
    predictions=tf.argmax(logits,axis=1)
)[1] #Ϊʲô��1 ����Ϊ���������ﲻ��Ҫ׼ȷ��������֡�����Ҫ�õ�һ��op

#�����Ự

saver = tf.train.Saver()  # defaults to saving all variables


sess=tf.Session()
#��ʼ������
#group �Ѻܶ������Ū��һ����
#��ʼ��������ȫ�֣��;ֲ�
init=tf.group(tf.global_variables_initializer(),
              tf.local_variables_initializer())
sess.run(init)

for i in range(20000):
   # for i in range(20000):
    batch=mnist.train.next_batch(50) #��Train��ѵ�������ݼ���ȡ����һ��������
    train_loss,train_op_=sess.run([loss,train_op],{input_x:batch[0],output_y:batch[1]})
    if i%100==0:
        test_accuracy=sess.run(accuracy_op,{input_x:test_x,output_y:test_y})
        print("Step=%d, Train loss=%.4f,[Test accuracy=%.2f]"%(i,train_loss,test_accuracy))

#���ԣ� ��ӡ20��Ԥ��ֵ����ʵֵ ��
test_output=sess.run(logits,{input_x:test_x[:20]})
inferenced_y=np.argmax(test_output,1)

saver.save(sess, './model_cnn2.ckpt')


print(inferenced_y,'Inferenced numbers')#�Ʋ������
print(np.argmax(test_y[:20],1),'Real numbers')
sess.close()
