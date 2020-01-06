


source /home/jcq/anaconda3/bin/activate tensorflow-object_detection

cd /home/jcq/models-master/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:/home/jcq/models-master/research/:/home/jcq/models-master/research/slim:/home/jcq/models-master/research/slim/nets

/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/builders/model_builder_test.py

/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/dataset_tools/create_pet_tf_record.py \
    --label_map_path=object_detection/data/pet_label_map.pbtxt \
    --data_dir=/media/jcq/Doc/DL_data/Oxford-IIT \
    --output_dir=//media/jcq/Doc/DL_data/Oxford-IIT/output_dir



CUDA_VISIBLE_DEVICES="" /home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/model_main.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config \
        --train_dir=/media/jcq/Doc/DL_data/Oxford-IIT/train_checkpoint


 ## 使用的是 cpu 进行训练，速度非常着急


/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/model_main.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config \
        --train_dir=/media/jcq/Doc/DL_data/Oxford-IIT/train_checkpoint

##  fail

/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/legacy/train.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config \
        --train_dir=/media/jcq/Doc/DL_data/Oxford-IIT/train_checkpoint



/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/legacy/train.py \
--pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config\
--train_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/ssd_mobilenet_v1_coco_2017_11_17 –alsologtostderr

/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/legacy/train.py --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config --train_dir=/media/jcq/Doc/DL_data/Oxford-IIT/output_dir –alsologtostderr


python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR} \







### 使用 tensorflow-gpu 环境

# 1. envroment

source /home/jcq/anaconda3/bin/activate tensorflow_gpu

cd /home/jcq/models-master/research


protoc object_detection/protos/*.proto --python_out=.


export PYTHONPATH=$PYTHONPATH:/home/jcq/models-master/research/:/home/jcq/models-master/research/slim:/home/jcq/models-master/research/slim/nets


CUDA_VISIBLE_DEVICES="" /home/jcq/.conda/envs/tensorflow_gpu/bin/python object_detection/model_main.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config \
        --train_dir=/media/jcq/Doc/DL_data/Oxford-IIT/train_checkpoint

## errors
from tensorflow.contrib.quantize.python import graph_matcher
ModuleNotFoundError: No module named 'tensorflow.contrib.quantize'



### 使用 object_detection 环境

# 1. envroment

source /home/jcq/anaconda3/bin/activate object_detection

cd /home/jcq/models-master/research


protoc object_detection/protos/*.proto --python_out=.


export PYTHONPATH=$PYTHONPATH:/home/jcq/models-master/research/:/home/jcq/models-master/research/slim:/home/jcq/models-master/research/slim/nets
/home/jcq/.conda/envs/object_detection/bin/python object_detection/builders/model_builder_test.py

/home/jcq/.conda/envs/object_detection/bin/python object_detection/model_main.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200104/pipeline.config \
        --train_dir=/media/jcq/Doc/DL_data/Oxford-IIT/train_checkpoint

