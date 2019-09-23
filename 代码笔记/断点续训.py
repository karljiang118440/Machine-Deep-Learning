 
#断点续训
#之前设置数据训练次数可以减少，这样每次都能保证从最新的开始训练
 with tf.Session() as sess:

        # saver.restore(sess, './log/' + "model_savemodel.cpkt-" + str(20000))
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('./model/')  # 注意此处是checkpoint存在的目录，千万不要写成‘./log’
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 自动恢复model_checkpoint_path保存模型一般是最新
            print("Model restored...")
        else:
            print('No Model')

