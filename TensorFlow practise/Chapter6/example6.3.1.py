

filter_weights=tf.get_variable(
	                          )


biases=tf.get_variable('biases',[16],initializer=tf.constant_initializer(0.1))

conv = tf.nn.conv2d(input,filter_weights,strides=[1,1,1,1],padding='SAME')

biases=tf.nn.bias_add(conv,biases)

actived_conv=tf.nn.relu(bias)

