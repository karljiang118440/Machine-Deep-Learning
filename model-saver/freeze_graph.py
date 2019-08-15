import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

sess = tf.Session()
saver = tf.train.import_meta_graph("meta文件目录")
saver.restore(sess, tf.train.latest_checkpoint("checkpoint文件所在目录"))
graph = tf.get_default_graph()

output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['输出tensor名字'])
with tf.gfile.FastGFile('pb文件保存目录', mode='wb') as f:
    f.write(output_graph_def.SerializeToString())