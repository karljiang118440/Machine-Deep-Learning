# -*- coding:utf-8 -*-
# -*- author£ºzzZ_CMing
# -*- 2018/01/24£»14:14
# -*- python3.5


import tensorflow as tf

pd_dir = "./MyModel.pb"



def main():
    x = tf.placeholder(dtype=tf.float32,shape=[None,2],name="in")
    #x = tf.constant([[1,2]],dtype=tf.float32)
    w1 = tf.get_variable("w1",dtype=tf.float32,initializer=tf.truncated_normal([2, 1], stddev=0.1))
    b1 = tf.get_variable("b1",initializer=tf.constant(.1, dtype=tf.float32, shape=[1, 1])) 

    y = tf.add(tf.matmul(x,w1),b1,name="out")
    
    with tf.Session() as sess:
        #获取计算图
        graph = tf.get_default_graph()
        #获取name和ops，这次代码并没有用到
        ret = graph.get_operations()
        r_names = []
        #获取name list
        for r in ret:
            r_names.append(r.name)

        srun = sess.run
        srun(tf.global_variables_initializer())
        print("y: ",srun(y,{x:[[1,2]]}))
        #存入输入与输出接口
        out_graph = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,["in","out"])
        saver_path = tf.train.write_graph(out_graph,".",pd_dir,as_text=True)

        
        print("saver path: ",saver_path)

if __name__ == "__main__":
    main()