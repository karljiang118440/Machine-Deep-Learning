(object_detection) jcq@jcq-linux:~/models-master/research$ /home/jcq/.conda/envs/object_detection/bin/python object_detection/model_main.py \
>     --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
>     --model_dir=${MODEL_DIR} \
>     --num_train_steps=${NUM_TRAIN_STEPS} \
>     --num_eval_steps=${NUM_EVAL_STEPS} \
>     --alsologtostderr

WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
If you depend on functionality not listed there, please file an issue.

2020-01-13 17:28:50.535808: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-01-13 17:28:50.644917: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-01-13 17:28:50.645362: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x561604d15670 executing computations on platform CUDA. Devices:
2020-01-13 17:28:50.645379: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce RTX 2070, Compute Capability 7.5
2020-01-13 17:28:50.666209: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600100000 Hz
2020-01-13 17:28:50.666838: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x561604de1c70 executing computations on platform Host. Devices:
2020-01-13 17:28:50.666859: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2020-01-13 17:28:50.666993: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce RTX 2070 major: 7 minor: 5 memoryClockRate(GHz): 1.62
pciBusID: 0000:29:00.0
totalMemory: 7.79GiB freeMemory: 6.89GiB
2020-01-13 17:28:50.667009: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2020-01-13 17:28:50.667583: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-01-13 17:28:50.667597: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2020-01-13 17:28:50.667605: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2020-01-13 17:28:50.667871: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6702 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:29:00.0, compute capability: 7.5)
WARNING:tensorflow:Forced number of epochs for all eval validations to be 1.
WARNING:tensorflow:Expected number of evaluation epochs is 1, but instead encountered `eval_on_train_input_config.num_epochs` = 0. Overwriting `num_epochs` to 1.
WARNING:tensorflow:Estimator's model_fn (<function create_model_fn.<locals>.model_fn at 0x7f61ad77ee18>) includes params argument, but params are not passed to Estimator.
WARNING:tensorflow:From /home/jcq/.conda/envs/object_detection/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /home/jcq/models-master/research/object_detection/builders/dataset_builder.py:80: parallel_interleave (from tensorflow.contrib.data.python.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.experimental.parallel_interleave(...)`.
WARNING:tensorflow:From /home/jcq/models-master/research/object_detection/utils/ops.py:466: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From /home/jcq/models-master/research/object_detection/inputs.py:287: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From /home/jcq/models-master/research/object_detection/core/preprocessor.py:188: sample_distorted_bounding_box (from tensorflow.python.ops.image_ops_impl) is deprecated and will be removed in a future version.
Instructions for updating:
`seed2` arg is deprecated.Use sample_distorted_bounding_box_v2 instead.
WARNING:tensorflow:From /home/jcq/models-master/research/object_detection/core/preprocessor.py:1218: calling squeeze (from tensorflow.python.ops.array_ops) with squeeze_dims is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
WARNING:tensorflow:From /home/jcq/models-master/research/object_detection/builders/dataset_builder.py:148: batch_and_drop_remainder (from tensorflow.contrib.data.python.ops.batching) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.batch(..., drop_remainder=True)`.
WARNING:root:Variable [FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_2_3x3_s2_512/weights] is available in checkpoint, but has an incompatible shape with model variable. Checkpoint shape: [[1, 1, 256, 512]], model variable shape: [[3, 3, 256, 512]]. This variable will not be initialized from the checkpoint.
WARNING:root:Variable [FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_3_3x3_s2_256/weights] is available in checkpoint, but has an incompatible shape with model variable. Checkpoint shape: [[1, 1, 128, 256]], model variable shape: [[3, 3, 128, 256]]. This variable will not be initialized from the checkpoint.
WARNING:root:Variable [FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_4_3x3_s2_256/weights] is available in checkpoint, but has an incompatible shape with model variable. Checkpoint shape: [[1, 1, 128, 256]], model variable shape: [[3, 3, 128, 256]]. This variable will not be initialized from the checkpoint.
WARNING:root:Variable [FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_5_3x3_s2_128/weights] is available in checkpoint, but has an incompatible shape with model variable. Checkpoint shape: [[1, 1, 64, 128]], model variable shape: [[3, 3, 64, 128]]. This variable will not be initialized from the checkpoint.
2020-01-13 17:29:03.582304: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2020-01-13 17:29:03.582353: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-01-13 17:29:03.582358: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2020-01-13 17:29:03.582362: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2020-01-13 17:29:03.582424: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6702 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:29:00.0, compute capability: 7.5)
WARNING:tensorflow:From /home/jcq/.conda/envs/object_detection/lib/python3.6/site-packages/tensorflow/python/training/saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.
WARNING:tensorflow:From /home/jcq/.conda/envs/object_detection/lib/python3.6/site-packages/tensorflow/python/training/saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.
WARNING:tensorflow:From /home/jcq/.conda/envs/object_detection/lib/python3.6/site-packages/tensorflow/python/training/saver.py:1070: get_checkpoint_mtimes (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file utilities to get mtimes.
WARNING:tensorflow:From /home/jcq/.conda/envs/object_detection/lib/python3.6/site-packages/tensorflow/python/training/saver.py:1070: get_checkpoint_mtimes (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file utilities to get mtimes.
WARNING:tensorflow:From /home/jcq/models-master/research/object_detection/eval_util.py:750: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From /home/jcq/models-master/research/object_detection/eval_util.py:750: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From /home/jcq/models-master/research/object_detection/utils/visualization_utils.py:429: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, use
    tf.py_function, which takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    
WARNING:tensorflow:From /home/jcq/models-master/research/object_detection/utils/visualization_utils.py:429: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, use
    tf.py_function, which takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    
2020-01-13 17:39:21.078523: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2020-01-13 17:39:21.078572: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-01-13 17:39:21.078578: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2020-01-13 17:39:21.078582: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2020-01-13 17:39:21.078643: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6702 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:29:00.0, compute capability: 7.5)
creating index...
index created!
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=362.45s).
Accumulating evaluation results...
DONE (t=61.47s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.092
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.196
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.074
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.038
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.171
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.118
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.183
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.196
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.010
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.119
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.343
2020-01-13 18:09:49.122639: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2020-01-13 18:09:49.126180: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-01-13 18:09:49.126193: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2020-01-13 18:09:49.126198: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2020-01-13 18:09:49.126957: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6702 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:29:00.0, compute capability: 7.5)
creating index...
index created!
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=356.30s).
Accumulating evaluation results...
DONE (t=56.04s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.091
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.195
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.072
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.171
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.117
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.181
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.195
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.117
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.340
2020-01-13 18:39:58.415247: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2020-01-13 18:39:58.418561: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-01-13 18:39:58.418570: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2020-01-13 18:39:58.418574: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2020-01-13 18:39:58.419115: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6702 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:29:00.0, compute capability: 7.5)
creating index...
index created!
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=371.29s).
Accumulating evaluation results...
DONE (t=57.02s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.090
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.195
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.071
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.035
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.169
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.116
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.180
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.194
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.117
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.338
2020-01-13 19:10:29.888269: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2020-01-13 19:10:29.891268: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-01-13 19:10:29.891280: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2020-01-13 19:10:29.891285: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2020-01-13 19:10:29.891821: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6702 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:29:00.0, compute capability: 7.5)
creating index...
index created!
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
Killed
