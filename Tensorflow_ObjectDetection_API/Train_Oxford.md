

source /home/jcq/anaconda3/bin/activate object_detection

cd /home/jcq/models-master/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:/home/jcq/models-master/research/:/home/jcq/models-master/research/slim:/home/jcq/models-master/research/slim/nets

/home/jcq/.conda/envs/object_detection/bin/python object_detection/builders/model_builder_test.py

/home/jcq/.conda/envs/object_detection/bin/python object_detection/dataset_tools/create_pet_tf_record.py \
    --label_map_path=object_detection/data/pet_label_map.pbtxt \
    --data_dir=/media/jcq/Doc/DL_data/Oxford-IIT \
    --output_dir=//media/jcq/Doc/DL_data/Oxford-IIT/output_dir



/home/jcq/.conda/envs/object_detection/bin/python object_detection/model_main.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config \
        --train_dir=/media/jcq/Doc/DL_data/Oxford-IIT/train_checkpoint


 ## 使用的是 cpu 进行训练，速度非常着急


model_main.py 中添加如下指令即可



import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

> >> 但是输出的文件位置有问题，并不对，机会配置并没有起到任何作用。相关的训练文件也找不到。






### 重新使用 train.py 进行训练


/home/jcq/.conda/envs/object_detection/bin/python object_detection/legacy/train.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config \
        --train_dir=/media/jcq/Doc/DL_data/Oxford-IIT/train_checkpoint

# 添加以下限制：

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


> ok, train sucesss !





# 使用官方的脚本转换成.pb模型
# From tensorflow/models/research/
PIPELINE_CONFIG_PATH　＝　
TRAIN_PATH =
EXPORT_DIR =
/home/jcq/.conda/envs/object_detection/bin/python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path "/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config" \
    --trained_checkpoint_prefix "/media/jcq/Doc/DL_data/Oxford-IIT/train_checkpoint/model.ckpt-10000" \
    --output_directory "/media/jcq/Doc/DL_data/Oxford-IIT/train_checkpoint/output_pb"


# 导出模型成功



## test.py

/home/jcq/.conda/envs/object_detection/bin/python object_detection/test.py



但是貌似没有任何效果，并没有在图片上画出来标记？



## eval



/home/jcq/.conda/envs/object_detection/bin/python eval.py\
 --checkpoint_dir=train\
 --eval_dir=eval\
 --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config


 --train_dir=/media/jcq/Doc/DL_data/Oxford-IIT/train_checkpoint








### AIRuner quantizations 







