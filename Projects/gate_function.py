import tensorflow as tf


sess=tf.Session()

a=tf.Variable(tf.constant(4.0))

x_val=5

x_data=tf.placeholder(dtype=tf.float32)

multiplication=tf.multiply(a,x_data)

loss=tf.square(tf.subtract(multiplication,50.))

init=tf.global_variables_initializer()
sess.run(init)

my_opt=tf.train.GradientDescentOptimizer(0.01)

train_step=my_opt.minimize(loss)

print('optimizing a  multiplication gate output ot 50')

for i in range(10):
    sess.run(train_step,feed_dict={x_data:x_val})
    a_val=sess.run(a)
    mult_output=sess.run(multiplication,feed_dict={x_data:x_val})

    print(str(a_val) + '*' + str(x_val)+ '=' + str(mult_output))










