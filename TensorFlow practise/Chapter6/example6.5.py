

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import  gfile

BOTTLENECK_TENSOR_SIZE=2048

BOTTLENECK_TENSOR_NAME='pool_3/_reshape:0'

JPEG_DATA_TENSOR_NAME='DecodeJpeg/contents:0'

MODEL_DIR='/path/to/model'

MODEL_FILE='classify_image_graph_def.pb'
CACHE_DIR='/tmp/bottleneck'

INPUT_DATA='/path/to/flower_data'

VALIDATION_PERCENTAGE=10





