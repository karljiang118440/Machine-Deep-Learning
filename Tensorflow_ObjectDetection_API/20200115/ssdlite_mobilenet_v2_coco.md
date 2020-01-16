
# 1、环境
source /home/jcq/anaconda3/bin/activate object_detection

cd /home/jcq/models-master/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:/home/jcq/models-master/research/:/home/jcq/models-master/research/slim:/home/jcq/models-master/research/slim/nets

/home/jcq/.conda/envs/object_detection/bin/python object_detection/builders/model_builder_test.py





# 2、数据集

## 2.1 训练钱数据准备

此时目录结构为

```
<20180823>
├── ssdlite_mobilenet_v2_coco_2018_05_09
│   └── saved_model
│       └── variables
├── mscoco
	└── *.record
├── mscoco_label_map.pbtxt
└── ssdlite_mobilenet_v2_coco.config
```

## 2.2、coco 数据集

/media/jcq/Doc/DL_data/COCO_Data/mscoco

1、下载数据
coco2014

2、链接数据

```
cd tensorflow/models/research/
ln -s xxxx/coco `pwd`/coco
```
3、转成tfrecode

```python
# 如果使用mask，默认只使用bbox（include_masks=False）
# 设置--include_masks=True


TRAIN_IMAGE_DIR=`pwd`/coco/train2014
VAL_IMAGE_DIR=`pwd`/coco/val2014
TEST_IMAGE_DIR=`pwd`/coco/val2014 # test2014
TRAIN_ANNOTATIONS_FILE=`pwd`/coco/annotations/instances_train2014.json
VAL_ANNOTATIONS_FILE=`pwd`/coco/annotations/instances_val2014.json
TESTDEV_ANNOTATIONS_FILE=`pwd`/coco/annotations/instances_val2014.json # `pwd`/coco/annotations/instances_test2014.json
OUTPUT_DIR=`pwd`/mscoco
python3 object_detection/dataset_tools/create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"








# 3、使用 train.py 进行训练


/home/jcq/.conda/envs/object_detection/bin/python object_detection/legacy/train.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200115/ssdlite_mobilenet_v2_coco.config \
        --train_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200115/models



## 3.1 、tensorboard 查看结果


tensorboard --logdir='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200115/models'







# 4、导出训练模型做推理
在确定要导出的候选检查点之后，从`tensorflow/models/research`运行以下命令：

```python
cd tensorflow/models/research
# From tensorflow/models/research/

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200115/ssdlite_mobilenet_v2_coco.config'
TRAINED_CKPT_PREFIX='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200115/models/model.ckpt-5000' 
EXPORT_DIR='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200115/models/frozen_pb'
/home/jcq/.conda/envs/object_detection/bin/python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```



# 5、 部署推理服务
  
  需要将 test.py 放在 object_detection 下面

/home/jcq/.conda/envs/object_detection/bin/python object_detection/test.py


## 完成,明显有效果

需要添加以下限制：

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))




