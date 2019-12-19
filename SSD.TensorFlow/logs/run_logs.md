
#1.success run train_ssd.py ,logs:

(tensorflow_gpu) jcq@jcq-linux:/media/jcq/Soft/Tensorflow/SSD.TensorFlow$ /home/jcq/.conda/envs/tensorflow_gpu/bin/python train_ssd.py
/home/jcq/.conda/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:523: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/jcq/.conda/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:524: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/jcq/.conda/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/jcq/.conda/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/jcq/.conda/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/jcq/.conda/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:532: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
2019-12-19 11:23:22.864029: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-12-19 11:23:22.966244: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-19 11:23:22.966585: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1392] Found device 0 with properties: 
name: GeForce RTX 2070 major: 7 minor: 5 memoryClockRate(GHz): 1.62
pciBusID: 0000:29:00.0
totalMemory: 7.79GiB freeMemory: 7.46GiB
2019-12-19 11:23:22.966599: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1471] Adding visible gpu devices: 0
2019-12-19 11:23:23.157388: I tensorflow/core/common_runtime/gpu/gpu_device.cc:952] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-12-19 11:23:23.157424: I tensorflow/core/common_runtime/gpu/gpu_device.cc:958]      0 
2019-12-19 11:23:23.157429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   N 
2019-12-19 11:23:23.157495: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1084] Created TensorFlow device (/device:GPU:0 with 7185 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:29:00.0, compute capability: 7.5)
WARNING:tensorflow:From train_ssd.py:460: replicate_model_fn (from tensorflow.contrib.estimator.python.estimator.replicate_model_fn) is deprecated and will be removed after 2018-05-31.
Instructions for updating:
Please use `tf.contrib.distribute.MirroredStrategy` instead.
2019-12-19 11:23:23.225658: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1471] Adding visible gpu devices: 0
2019-12-19 11:23:23.225706: I tensorflow/core/common_runtime/gpu/gpu_device.cc:952] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-12-19 11:23:23.225711: I tensorflow/core/common_runtime/gpu/gpu_device.cc:958]      0 
2019-12-19 11:23:23.225716: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   N 
2019-12-19 11:23:23.225798: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1084] Created TensorFlow device (/device:GPU:0 with 7185 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:29:00.0, compute capability: 7.5)
INFO:tensorflow:Replicating the `model_fn` across ['/device:GPU:0'].  Variables are going to be placed on ['/device:GPU:0'].  Consolidation device is going to be /device:GPU:0.
INFO:tensorflow:Using config: {'_model_dir': '/media/jcq/Doc/DL_data/VOC/logs', '_tf_random_seed': 20180503, '_save_summary_steps': 2000, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 1200, '_session_config': gpu_options {
  per_process_gpu_memory_fraction: 1.0
}
allow_soft_placement: true
, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 500, '_train_distribute': None, '_device_fn': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f3b78ba00b8>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
Starting a training cycle.
INFO:tensorflow:Calling model_fn.
WARNING:tensorflow:From /media/jcq/Soft/Tensorflow/SSD.TensorFlow/net/ssd_net.py:114: calling reduce_sum (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From train_ssd.py:421: TowerOptimizer.__init__ (from tensorflow.contrib.estimator.python.estimator.replicate_model_fn) is deprecated and will be removed after 2018-05-31.
Instructions for updating:
Please use `tf.contrib.distribute.MirroredStrategy` instead.
/home/jcq/.conda/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/ops/gradients_impl.py:100: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
INFO:tensorflow:Ignoring --checkpoint_path because a checkpoint already exists in /media/jcq/Doc/DL_data/VOC/logs.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
2019-12-19 11:23:25.718043: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1471] Adding visible gpu devices: 0
2019-12-19 11:23:25.718077: I tensorflow/core/common_runtime/gpu/gpu_device.cc:952] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-12-19 11:23:25.718081: I tensorflow/core/common_runtime/gpu/gpu_device.cc:958]      0 
2019-12-19 11:23:25.718083: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   N 
2019-12-19 11:23:25.718146: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1084] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7185 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:29:00.0, compute capability: 7.5)
INFO:tensorflow:Restoring parameters from /media/jcq/Doc/DL_data/VOC/logs/model.ckpt-0
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into /media/jcq/Doc/DL_data/VOC/logs/model.ckpt.
INFO:tensorflow:lr=0.000100, ce=22.110928, loc=3.040825, loss=31.556402, l2=6.404650, acc=0.000000
INFO:tensorflow:loss = 31.556402, step = 0
INFO:tensorflow:global_step/sec: 8.42877
INFO:tensorflow:lr=0.000100, ce=5.227589, loc=2.342388, loss=13.974075, l2=6.404099, acc=0.467647
INFO:tensorflow:loss = 13.974075, step = 500 (59.320 sec)
INFO:tensorflow:global_step/sec: 8.50477
INFO:tensorflow:lr=0.001000, ce=18.224215, loc=5.480471, loss=30.307096, l2=6.602411, acc=0.558765
INFO:tensorflow:loss = 30.307096, step = 1000 (58.791 sec)
INFO:tensorflow:global_step/sec: 8.47424
INFO:tensorflow:lr=0.001000, ce=4.683057, loc=2.281278, loss=13.558760, l2=6.594425, acc=0.583910
INFO:tensorflow:loss = 13.55876, step = 1500 (59.002 sec)
INFO:tensorflow:global_step/sec: 8.42799
INFO:tensorflow:lr=0.001000, ce=11.457021, loc=2.562667, loss=20.603548, l2=6.583861, acc=0.606716
INFO:tensorflow:loss = 20.603548, step = 2000 (59.327 sec)
INFO:tensorflow:global_step/sec: 8.38772
INFO:tensorflow:lr=0.001000, ce=4.882475, loc=2.288802, loss=13.745257, l2=6.573980, acc=0.617403
INFO:tensorflow:loss = 13.745257, step = 2500 (59.611 sec)
INFO:tensorflow:global_step/sec: 8.35991
INFO:tensorflow:lr=0.001000, ce=8.909486, loc=2.543677, loss=18.017097, l2=6.563936, acc=0.640909
INFO:tensorflow:loss = 18.017097, step = 3000 (59.809 sec)
INFO:tensorflow:global_step/sec: 8.26849
INFO:tensorflow:lr=0.001000, ce=8.162116, loc=2.371217, loss=17.087543, l2=6.554209, acc=0.652439
INFO:tensorflow:loss = 17.087543, step = 3500 (60.471 sec)
INFO:tensorflow:global_step/sec: 8.21916
INFO:tensorflow:lr=0.001000, ce=5.316886, loc=2.518495, loss=14.379858, l2=6.544477, acc=0.668367
INFO:tensorflow:loss = 14.379858, step = 4000 (60.834 sec)
INFO:tensorflow:global_step/sec: 8.47325
INFO:tensorflow:lr=0.001000, ce=5.763558, loc=2.694732, loss=14.993493, l2=6.535202, acc=0.675000
INFO:tensorflow:loss = 14.993493, step = 4500 (59.009 sec)
INFO:tensorflow:global_step/sec: 8.46243
INFO:tensorflow:lr=0.001000, ce=5.262539, loc=2.203122, loss=13.991657, l2=6.525996, acc=0.683518
INFO:tensorflow:loss = 13.991657, step = 5000 (59.085 sec)
INFO:tensorflow:global_step/sec: 8.47018
INFO:tensorflow:lr=0.001000, ce=6.590781, loc=2.193778, loss=15.301476, l2=6.516917, acc=0.687008
INFO:tensorflow:loss = 15.3014765, step = 5500 (59.030 sec)
INFO:tensorflow:global_step/sec: 8.47422
INFO:tensorflow:lr=0.001000, ce=4.288447, loc=2.426745, loss=13.223043, l2=6.507850, acc=0.690959
INFO:tensorflow:loss = 13.223043, step = 6000 (59.003 sec)
INFO:tensorflow:global_step/sec: 8.46649
INFO:tensorflow:lr=0.001000, ce=4.292162, loc=2.062066, loss=12.853159, l2=6.498932, acc=0.694444
INFO:tensorflow:loss = 12.853159, step = 6500 (59.056 sec)
INFO:tensorflow:global_step/sec: 8.45645
INFO:tensorflow:lr=0.001000, ce=5.570697, loc=1.911748, loss=13.972604, l2=6.490158, acc=0.697137
INFO:tensorflow:loss = 13.972604, step = 7000 (59.126 sec)
INFO:tensorflow:global_step/sec: 8.44256
INFO:tensorflow:lr=0.001000, ce=4.703139, loc=1.862069, loss=13.046595, l2=6.481386, acc=0.701219
INFO:tensorflow:loss = 13.046595, step = 7500 (59.224 sec)
INFO:tensorflow:global_step/sec: 8.18063
INFO:tensorflow:lr=0.001000, ce=3.666051, loc=2.231963, loss=12.371534, l2=6.473520, acc=0.703891
INFO:tensorflow:loss = 12.371534, step = 8000 (61.120 sec)
INFO:tensorflow:global_step/sec: 8.25916
INFO:tensorflow:lr=0.001000, ce=4.442044, loc=2.021635, loss=12.928457, l2=6.464778, acc=0.705390
INFO:tensorflow:loss = 12.928457, step = 8500 (60.538 sec)
INFO:tensorflow:global_step/sec: 8.26493
INFO:tensorflow:lr=0.001000, ce=3.258704, loc=1.282698, loss=10.997750, l2=6.456348, acc=0.707623
INFO:tensorflow:loss = 10.99775, step = 9000 (60.497 sec)
INFO:tensorflow:global_step/sec: 8.26281
INFO:tensorflow:lr=0.001000, ce=4.095664, loc=1.894388, loss=12.437846, l2=6.447794, acc=0.709407
INFO:tensorflow:loss = 12.437846, step = 9500 (60.512 sec)
INFO:tensorflow:global_step/sec: 8.25935
INFO:tensorflow:lr=0.001000, ce=4.580778, loc=1.632417, loss=12.652540, l2=6.439345, acc=0.711239
INFO:tensorflow:loss = 12.65254, step = 10000 (60.538 sec)
INFO:tensorflow:Saving checkpoints for 10026 into /media/jcq/Doc/DL_data/VOC/logs/model.ckpt.
INFO:tensorflow:global_step/sec: 8.12282
INFO:tensorflow:lr=0.001000, ce=4.913309, loc=1.623711, loss=12.968016, l2=6.430995, acc=0.713453
INFO:tensorflow:loss = 12.968016, step = 10500 (61.555 sec)
INFO:tensorflow:global_step/sec: 8.33201
INFO:tensorflow:lr=0.001000, ce=5.017112, loc=2.198732, loss=13.638383, l2=6.422539, acc=0.716831
INFO:tensorflow:loss = 13.638383, step = 11000 (60.010 sec)
INFO:tensorflow:global_step/sec: 8.44815
INFO:tensorflow:lr=0.001000, ce=5.341780, loc=2.067878, loss=13.823874, l2=6.414217, acc=0.718997
INFO:tensorflow:loss = 13.8238735, step = 11500 (59.184 sec)
INFO:tensorflow:global_step/sec: 8.46326
INFO:tensorflow:lr=0.001000, ce=4.575591, loc=1.615974, loss=12.597551, l2=6.405987, acc=0.720083
INFO:tensorflow:loss = 12.597551, step = 12000 (59.079 sec)
INFO:tensorflow:global_step/sec: 8.3614
INFO:tensorflow:lr=0.001000, ce=4.255994, loc=1.556083, loss=12.209937, l2=6.397860, acc=0.720570
INFO:tensorflow:loss = 12.209937, step = 12500 (59.799 sec)
INFO:tensorflow:global_step/sec: 8.45368
INFO:tensorflow:lr=0.001000, ce=5.169243, loc=1.830459, loss=13.389471, l2=6.389770, acc=0.722061
INFO:tensorflow:loss = 13.389471, step = 13000 (59.146 sec)
INFO:tensorflow:global_step/sec: 8.41471
INFO:tensorflow:lr=0.001000, ce=4.830331, loc=1.897859, loss=13.109994, l2=6.381804, acc=0.721971
INFO:tensorflow:loss = 13.109994, step = 13500 (59.420 sec)
INFO:tensorflow:global_step/sec: 8.45117
INFO:tensorflow:lr=0.001000, ce=5.088928, loc=1.164768, loss=12.627699, l2=6.374002, acc=0.722376
INFO:tensorflow:loss = 12.627699, step = 14000 (59.164 sec)
INFO:tensorflow:global_step/sec: 8.44589
INFO:tensorflow:lr=0.001000, ce=5.339112, loc=2.483387, loss=14.188589, l2=6.366090, acc=0.723587
INFO:tensorflow:loss = 14.188589, step = 14500 (59.200 sec)
INFO:tensorflow:global_step/sec: 8.45008
INFO:tensorflow:lr=0.001000, ce=4.038191, loc=1.675401, loss=12.072199, l2=6.358607, acc=0.723745
INFO:tensorflow:loss = 12.072199, step = 15000 (59.171 sec)
INFO:tensorflow:global_step/sec: 8.43763
INFO:tensorflow:lr=0.001000, ce=4.879896, loc=1.410182, loss=12.641006, l2=6.350928, acc=0.724226
INFO:tensorflow:loss = 12.641006, step = 15500 (59.259 sec)
INFO:tensorflow:global_step/sec: 8.44171
INFO:tensorflow:lr=0.001000, ce=4.897944, loc=1.877890, loss=13.119187, l2=6.343353, acc=0.725340
INFO:tensorflow:loss = 13.119187, step = 16000 (59.229 sec)
INFO:tensorflow:global_step/sec: 8.45758
INFO:tensorflow:lr=0.001000, ce=3.907465, loc=1.654767, loss=11.898302, l2=6.336070, acc=0.725729
INFO:tensorflow:loss = 11.898302, step = 16500 (59.118 sec)
INFO:tensorflow:global_step/sec: 8.44551
INFO:tensorflow:lr=0.001000, ce=5.647048, loc=2.119690, loss=14.095480, l2=6.328742, acc=0.726816
INFO:tensorflow:loss = 14.09548, step = 17000 (59.203 sec)
INFO:tensorflow:global_step/sec: 8.39636
INFO:tensorflow:lr=0.001000, ce=4.844564, loc=1.379867, loss=12.546010, l2=6.321579, acc=0.727068
INFO:tensorflow:loss = 12.54601, step = 17500 (59.549 sec)
INFO:tensorflow:global_step/sec: 8.4572
INFO:tensorflow:lr=0.001000, ce=3.451889, loc=1.285977, loss=11.052299, l2=6.314434, acc=0.727845
INFO:tensorflow:loss = 11.0522995, step = 18000 (59.121 sec)
INFO:tensorflow:global_step/sec: 8.45673
INFO:tensorflow:lr=0.001000, ce=3.433673, loc=1.522650, loss=11.263727, l2=6.307403, acc=0.729281
INFO:tensorflow:loss = 11.263727, step = 18500 (59.125 sec)
INFO:tensorflow:global_step/sec: 8.42045
INFO:tensorflow:lr=0.001000, ce=3.741315, loc=1.369030, loss=11.410939, l2=6.300594, acc=0.729668
INFO:tensorflow:loss = 11.410939, step = 19000 (59.379 sec)
INFO:tensorflow:global_step/sec: 8.43533
INFO:tensorflow:lr=0.001000, ce=2.975537, loc=1.098941, loss=10.368244, l2=6.293766, acc=0.730322
INFO:tensorflow:loss = 10.368244, step = 19500 (59.274 sec)
INFO:tensorflow:Saving checkpoints for 20000 into /media/jcq/Doc/DL_data/VOC/logs/model.ckpt.
INFO:tensorflow:Loss for final step: 13.278389.
