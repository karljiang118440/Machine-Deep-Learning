



'''
import tensorflow as tf

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    a = tf.constant(1)
    b = tf.constant(3)
    c = a + b
    print('结果是：%d\n 值为：%d' % (sess.run(c), sess.run(c)))
'''


import tensorflow as tf

with tf.device('/cpu:0'):

    a = tf.constant([1.0,2.0,3.0],shape=[3],name='a')

    b = tf.constant([1.0,2.0,3.0],shape=[3],name='b')

with tf.device('/gpu:1'):

     c = a+b

print(c)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))

sess.run(tf.global_variables_initializer())

print(sess.run(c))


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


