
# 1、环境
source /home/jcq/anaconda3/bin/activate object_detection

cd /home/jcq/models-master/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:/home/jcq/models-master/research/:/home/jcq/models-master/research/slim:/home/jcq/models-master/research/slim/nets

/home/jcq/.conda/envs/object_detection/bin/python object_detection/builders/model_builder_test.py





# 3、使用 train.py 进行训练


/home/jcq/.conda/envs/object_detection/bin/python object_detection/legacy/train.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200114/ssd_mobilenet_v2_quantized_300x300_coco.config \
        --train_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200114/models



## 3.1 、tensorboard 查看结果


tensorboard --logdir='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200114/models'







# 4、导出训练模型做推理
在确定要导出的候选检查点之后，从`tensorflow/models/research`运行以下命令：

```python
cd tensorflow/models/research
# From tensorflow/models/research/

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200114/ssd_mobilenet_v2_quantized_300x300_coco.config'
TRAINED_CKPT_PREFIX='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200114/models/model.ckpt-5000' 
EXPORT_DIR='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200114/models/frozen_pb'
/home/jcq/.conda/envs/object_detection/bin/python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}




# 5、 部署推理服务
  
  需要将 test.py 放在 object_detection 下面

/home/jcq/.conda/envs/object_detection/bin/python object_detection/test.py


## 完成,明显有效果

需要添加以下限制：

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))




