import tensorflow as tf
import numpy as np

import matplotlib.pyplot  as plt


sess=tf.Session()
tf.set_random_seed(5)
np.random.seed(42)



a1=tf.Variable(tf.random_normal(shape=[1,1]))
b1=tf.Variable(tf.random_uniform(shape=[1,1]))
a2=tf.Variable(tf.random_normal(shape=[1,1]))
b2=tf.Variable(tf.random_uniform(shape=[1,1]))

x=np.random.normal(2,0.1,500)
x_data=tf.placeholder(shape=[None,1],dtype=tf.float32)


sigmoid_activation=tf.simgmod(tf.add(tf.multiply(a1,x_data),b1))
relu_activation=tf.nn.relu(tf.add(tf.multiply(a2,x_data),b2))



loss1=tf.reduce_mean(tf.square((tf.sub(sigmoid_activation,0.75))))
loss2=tf.reduce_mean(tf.square((tf.sub(relu_activation,0.75))))




my_opt=tf.train.GradientDescentOptimizer(0.01)
train_step_sigmod=my_opt.minimize(loss1)
train_step_relu=my_opt.minimize(loss2)

init=tf.global_variables_initializer()
sess.run(init)


##define the loss activation val
loss_vec_sigmod=[]
loss_vec_relu=[]
activation_vec_sigmod=[]
activation_vec_relu=[]





for i in range(750):
    sess.run(train_step,feed_dict={x_data:x_val})

    a_val,b_val=(sess.run(a),sess.run(b))

    two_gata_output=sess.run(two_gate,feed_dict={x_data:x_val})

    print(str(a_val) + '*' + str(x_val)+ str(b_val)+'=' + str(two_gata_output))
