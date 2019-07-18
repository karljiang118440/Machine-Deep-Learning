
#!/usr/bin/python3.5
# -*- coding: utf-8 -*-  
 
import sys
import os
import time
import random
 
import numpy as np
import tensorflow as tf
 
from PIL import Image
 
 
SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 6
iterations = 300
 
SAVER_DIR = "train-saver/province/"
 
PROVINCES = ("��","��","��","��","��","��")
nProvinceIndex = 0
 
time_begin = time.time()
 
 
# ��������ڵ㣬��Ӧ��ͼƬ����ֵ���󼯺Ϻ�ͼƬ��ǩ(�������������)
x = tf.placeholder(tf.float32, shape=[None, SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
 
x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
 
 
# ����������
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')
 
# ����ȫ���Ӳ㺯��
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
 
 
if __name__ =='__main__' and sys.argv[1]=='train':
    # ��һ�α���ͼƬĿ¼��Ϊ�˻�ȡͼƬ����
    input_count = 0
    for i in range(0,NUM_CLASSES):
        dir = './train_images/training-set/chinese-characters/%s/' % i           # ������Ըĳ����Լ���ͼƬĿ¼��iΪ�����ǩ
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                input_count += 1
 
    # �����Ӧά���͸�ά���ȵ�����
    input_images = np.array([[0]*SIZE for i in range(input_count)])
    input_labels = np.array([[0]*NUM_CLASSES for i in range(input_count)])
 
    # �ڶ��α���ͼƬĿ¼��Ϊ������ͼƬ���ݺͱ�ǩ
    index = 0
    for i in range(0,NUM_CLASSES):
        dir = './train_images/training-set/chinese-characters/%s/' % i          # ������Ըĳ����Լ���ͼƬĿ¼��iΪ�����ǩ
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # ͨ�������Ĵ���ʹ���ֵ�������ϸ�����������ʶ��׼ȷ��
                        if img.getpixel((w, h)) > 230:
                            input_images[index][w+h*width] = 0
                        else:
                            input_images[index][w+h*width] = 1
                input_labels[index][i] = 1
                index += 1
 
    # ��һ�α���ͼƬĿ¼��Ϊ�˻�ȡͼƬ����
    val_count = 0
    for i in range(0,NUM_CLASSES):
        dir = './train_images/validation-set/chinese-characters/%s/' % i           # ������Ըĳ����Լ���ͼƬĿ¼��iΪ�����ǩ
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                val_count += 1
 
    # �����Ӧά���͸�ά���ȵ�����
    val_images = np.array([[0]*SIZE for i in range(val_count)])
    val_labels = np.array([[0]*NUM_CLASSES for i in range(val_count)])
 
    # �ڶ��α���ͼƬĿ¼��Ϊ������ͼƬ���ݺͱ�ǩ
    index = 0
    for i in range(0,NUM_CLASSES):
        dir = './train_images/validation-set/chinese-characters/%s/' % i          # ������Ըĳ����Լ���ͼƬĿ¼��iΪ�����ǩ
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # ͨ�������Ĵ���ʹ���ֵ�������ϸ�����������ʶ��׼ȷ��
                        if img.getpixel((w, h)) > 230:
                            val_images[index][w+h*width] = 0
                        else:
                            val_images[index][w+h*width] = 1
                val_labels[index][i] = 1
                index += 1
    
    with tf.Session() as sess:
        # ��һ�������
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="W_conv1")
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_conv1")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')
 
        # �ڶ��������
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name="W_conv2")
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv2")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')
 
 
        # ȫ���Ӳ�
        W_fc1 = tf.Variable(tf.truncated_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20*32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)
 
 
        # dropout
        keep_prob = tf.placeholder(tf.float32)
 
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
 
        # readout��
        W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1), name="W_fc2")
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b_fc2")
 
        # �����Ż�����ѵ��op
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
 
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
        # ��ʼ��saver
        saver = tf.train.Saver()
 
        sess.run(tf.global_variables_initializer())
 
        time_elapsed = time.time() - time_begin
        print("��ȡͼƬ�ļ��ķ�ʱ�䣺%d��" % time_elapsed)
        time_begin = time.time()
 
        print ("һ����ȡ�� %s ��ѵ��ͼ�� %s ����ǩ" % (input_count, input_count))
 
        # ����ÿ��ѵ��op����������͵�������������Ϊ��֧������ͼƬ������������һ������remainder��Ʃ�磬���ÿ��ѵ��op���������Ϊ60��ͼƬ����Ϊ150�ţ���ǰ�����θ�����60�ţ����һ������30�ţ�����30��
        batch_size = 60
        iterations = iterations
        batches_count = int(input_count / batch_size)
        remainder = input_count % batch_size
        print ("ѵ�����ݼ��ֳ� %s ��, ǰ��ÿ�� %s �����ݣ����һ�� %s ������" % (batches_count+1, batch_size, remainder))
 
        # ִ��ѵ������
        for it in range(iterations):
            # ����Ĺؼ���Ҫ����������תΪnp.array
            for n in range(batches_count):
                train_step.run(feed_dict={x: input_images[n*batch_size:(n+1)*batch_size], y_: input_labels[n*batch_size:(n+1)*batch_size], keep_prob: 0.5})
            if remainder > 0:
                start_index = batches_count * batch_size;
                train_step.run(feed_dict={x: input_images[start_index:input_count-1], y_: input_labels[start_index:input_count-1], keep_prob: 0.5})
 
            # ÿ�����ε������ж�׼ȷ���Ƿ��Ѵﵽ100%���ﵽ���˳�����ѭ��
            iterate_accuracy = 0
            if it%5 == 0:
                iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
                print ('�� %d ��ѵ������: ׼ȷ�� %0.5f%%' % (it, iterate_accuracy*100))
                if iterate_accuracy >= 0.9999 and it >= 150:
                    break;
 
        print ('���ѵ��!')
        time_elapsed = time.time() - time_begin
        print ("ѵ���ķ�ʱ�䣺%d��" % time_elapsed)
        time_begin = time.time()
 
        # ����ѵ�����
        if not os.path.exists(SAVER_DIR):
            print ('������ѵ�����ݱ���Ŀ¼�����ڴ�������Ŀ¼')
            os.makedirs(SAVER_DIR)
        saver_path = saver.save(sess, "%smodel.ckpt"%(SAVER_DIR))
 
 
 
if __name__ =='__main__' and sys.argv[1]=='predict':
    saver = tf.train.import_meta_graph("%smodel.ckpt.meta"%(SAVER_DIR))
    with tf.Session() as sess:
        model_file=tf.train.latest_checkpoint(SAVER_DIR)
        saver.restore(sess, model_file)
 
        # ��һ�������
        W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
        b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')
 
        # �ڶ��������
        W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
        b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')
 
 
        # ȫ���Ӳ�
        W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
        b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20*32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)
 
 
        # dropout
        keep_prob = tf.placeholder(tf.float32)
 
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
 
        # readout��
        W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
        b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")
 
        # �����Ż�����ѵ��op
        conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 
        for n in range(1,2):
            path = "test_images/%s.bmp" % (n)
            img = Image.open(path)
            width = img.size[0]
            height = img.size[1]
 
            img_data = [[0]*SIZE for i in range(1)]
            for h in range(0, height):
                for w in range(0, width):
                    if img.getpixel((w, h)) < 190:
                        img_data[0][w+h*width] = 1
                    else:
                        img_data[0][w+h*width] = 0
            
            result = sess.run(conv, feed_dict = {x: np.array(img_data), keep_prob: 1.0})
            max1 = 0
            max2 = 0
            max3 = 0
            max1_index = 0
            max2_index = 0
            max3_index = 0
            for j in range(NUM_CLASSES):
                if result[0][j] > max1:
                    max1 = result[0][j]
                    max1_index = j
                    continue
                if (result[0][j]>max2) and (result[0][j]<=max1):
                    max2 = result[0][j]
                    max2_index = j
                    continue
                if (result[0][j]>max3) and (result[0][j]<=max2):
                    max3 = result[0][j]
                    max3_index = j
                    continue
            
            nProvinceIndex = max1_index
            print ("���ʣ�  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (PROVINCES[max1_index],max1*100, PROVINCES[max2_index],max2*100, PROVINCES[max3_index],max3*100))
            
        print ("ʡ�ݼ����: %s" % PROVINCES[nProvinceIndex])
