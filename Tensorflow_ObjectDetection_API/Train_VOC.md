

pip install -U tensorflow-gpu==2.0

pip install pycocotools

pip uninstall tensorflow-gpu==2.0

pip install tensorflow-gpu==1.12.0

pip install -U tensorflow-gpu==1.13.1

pip install tensorflow-gpu==1.14.0  #终于能够匹配 cuda10.1 cudnn 7.6

pip install pillow

## Protobuf Compilation

source /home/jcq/anaconda3/bin/activate tensorflow-object_detection

cd /home/jcq/models-master/research

```python
sudo apt-get install protobuf-compiler # 对于python3.5会出错
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```


## Add Libraries to PYTHONPATH

```
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/slim/nets
# 暂时有效，一旦关闭终端就无效
```
将路径写入到`~/.bashrc`,可以任何时候都有效

```
vim ~/.bashrc
# 写入以下内容（改成绝对路径）
export PYTHONPATH=$PYTHONPATH:/home/jcq/models-master/research/:/home/jcq/models-master/research/slim:/home/jcq/models-master/research/slim/nets


# 保存后
source ~/.bashrc
```
##  Testing the Installation

```
cd tensorflow/models/research/
/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/builders/model_builder_test.py
```
---

       OK ] ModelBuilderTest.test_create_ssd_mobilenet_v2_model_from_config
[ RUN      ] ModelBuilderTest.test_create_ssd_resnet_v1_fpn_model_from_config
[       OK ] ModelBuilderTest.test_create_ssd_resnet_v1_fpn_model_from_config
[ RUN      ] ModelBuilderTest.test_create_ssd_resnet_v1_ppn_model_from_config
[       OK ] ModelBuilderTest.test_create_ssd_resnet_v1_ppn_model_from_config
[ RUN      ] ModelBuilderTest.test_session
[  SKIPPED ] ModelBuilderTest.test_session
----------------------------------------------------------------------
Ran 22 tests in 0.077s












# tensorflow object detection

 '''
 
 auther : jiangchoaqing

 data  :2020.01.02
 '''


```python
# From tensorflow/models/research/
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# tar -xvf VOCtrainval_11-May-2012.tar
/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=/media/jcq/Doc/DL_data/VOC_Data/train/VOCdevkit --year=VOC2007 --set=train \
    --output_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200103/pascal_train.record


/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=/media/jcq/Doc/DL_data/VOC_Data/train/VOCdevkit --year=VOC2007 --set=val \
    --output_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200103/pascal_val.record
```
4、[train_config](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)设置

`fine_tune_checkpoint`应提供预先存在的检查点的路径（即：`“/ usr / home / username / checkpoint / model.ckpt - #####”`）。` from_detection_checkpoint`是一个布尔值。如果为`false`，则假定检查点来自对象分类检查点。请注意，从检测检查点开始通常会导致比分类检查点更快的训练工作。

可在[此处](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)找到提供的检查点列表


## train

```
cd tensorflow/models/research
mkdir 20170820
mv pascal_train.record pascal_val.record ./20170820

cp -r object_detection/data/pascal_label_map.pbtxt ./20170820
cp -r object_detection/samples/configs/faster_rcnn_resnet101_voc07.config ./20170820
```
在[此处](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)下载[faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)解压放在目录`20170820`

此时目录结构为

```
20170820/
├── faster_rcnn_resnet101_coco_2018_01_28
│   ├── checkpoint
│   ├── frozen_inference_graph.pb
│   ├── model.ckpt.data-00000-of-00001
│   ├── model.ckpt.index
│   ├── model.ckpt.meta
│   ├── pipeline.config
│   └── saved_model
│       ├── saved_model.pb
│       └── variables
├── faster_rcnn_resnet101_voc07.config
├── pascal_label_map.pbtxt
├── pascal_train.record
└── pascal_val.record
```

修改`faster_rcnn_resnet101_voc07.config`配置

```python
将PATH_TO_BE_CONFIGURED 都替换成 20170820

第 110 和 111 行内容改为：（表示不微调，从头开始训练模型）
# fine_tune_checkpoint: "20170820/model.ckpt"  （注释掉）
from_detection_checkpoint: false  （设为false）

# or 如果微调，则修改成
fine_tune_checkpoint: "20170820/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt"
from_detection_checkpoint: true

# or 如果要接着自己训练的模型继续训练（如：中途中断了）
fine_tune_checkpoint: "20170820/model/model.ckpt-1102" # 自己训练的模型保存位置
from_detection_checkpoint: true
```

运行以下命令进行train

```python
cd ../ # (cd ./models/resarch)

# From the tensorflow/models/research/
PIPELINE_CONFIG_PATH='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200103/faster_rcnn_resnet101_voc07.config'
MODEL_DIR='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200103/model' # 训练保存的model 位置
NUM_TRAIN_STEPS=10000
NUM_EVAL_STEPS=2000
/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr
```

参考：https://github.com/tensorflow/models/issues/4856

如果训练过程出现`TypeError: can't pickle dict_values objects`，需修改https://github.com/tensorflow/models/blob/master/research/object_detection/model_lib.py#L390  将`category_index.values()`改成`list(category_index.values())`


# 查看 TensorBoard

```
# 由于上面的训练过程不会打印任何训练的信息，需从tensorboard中查看
tensorboard --logdir='./20170820/model'
```

# [导出训练模型做推理](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md)

在确定要导出的候选检查点之后，从`tensorflow/models/research`运行以下命令：

```python
cd tensorflow/models/research
# From tensorflow/models/research/

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='./20170820/faster_rcnn_resnet101_voc07.config'
TRAINED_CKPT_PREFIX='./20170820/model/model.ckpt-1102' 
EXPORT_DIR='./20170820/model/frozen_pb'
/home/jcq/.conda/envs/tensorflow-object_detection/bin/python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```
注意：我们正在配置导出的模型以摄取4-D图像张量。 我们还可以将导出的模型配置为采用编码图像或序列化tf.Examples。

导出后，您应该看到包含以下内容的目录`${EXPORT_DIR}`：

- `saved_model/`，包含导出模型的已保存模型格式的目录
- `frozen_inference_graph.pb`，导出模型的冻结图格式
- `model.ckpt.*`，用于导出的模型检查点
- `checkpoint`，指定还原包含的检查点文件的文件
- `pipeline.config`，导出模型的管道配置文件


# [部署推理服务](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)

```python
# -*- coding:utf-8 -*-
"""
根据实际修改以下内容，其他的基本保持不变：
NUM_CLASSES = 20 # VOC是20类（不含背景）
PATH_TO_FROZEN_GRAPH # 指定冻结的pb文件路径
PATH_TO_LABELS # 指定label map 即标签id与标签名对应
PATH_TO_TEST_IMAGES_DIR # 指定要推理的图片路径
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from PIL import Image
import cv2

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# Object detection imports
# 以下是从对象检测模块导入的内容。
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# Model preparation
# 只需将PATH_TO_FROZEN_GRAPH更改为指向新的.pb文件，就可以在此处加载使用export_inference_graph.py工具导出的任何模型。

# 默认情况下，我们在此处使用“带有Mobilenet的SSD”模型。
# 请参阅检测模型动物园以获取其他模型的列表，这些模型可以开箱即用，具有不同的速度和精度。
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

"""
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
"""
NUM_CLASSES = 20#90

"""
# Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
"""
PATH_TO_FROZEN_GRAPH='./model/frozen_pb/frozen_inference_graph.pb'
PATH_TO_LABELS='./pascal_label_map.pbtxt'

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map
'''
标签映射将索引映射到类别名称，因此当我们的卷积网络预测5时，我们知道这对应于飞机。 
这里我们使用内部实用程序函数，但任何返回字典映射整数到适当的字符串标签的东西都可以
'''
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '../object_detection/test_images'
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]
import glob
TEST_IMAGE_PATHS=glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR,'*.jpg'))

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)

  image_np= np.concatenate((np.expand_dims(image_np[:,:,2],-1),
                            np.expand_dims(image_np[:,:,1],-1),
                            np.expand_dims(image_np[:,:,0],-1)),-1) # ==>BGR

  cv2.imwrite(image_path.replace('.jpg','_test.jpg'),
              image_np,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
  # cv2.COLOR_RGB2BGR(image_np)
  # plt.figure(figsize=IMAGE_SIZE)
  # plt.imshow(image_np)
  # plt.show()
```