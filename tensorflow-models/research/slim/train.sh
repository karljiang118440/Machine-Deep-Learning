
# Train inception models in flowers

DATA_DIR=/home/jcq/models-master/research/slim/tmp/data/flowers
python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"





CHECKPOINT_DIR=/home/jcq/models-master/research/slim/tmp/checkpoints
mkdir ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvf inception_v3_2016_08_28.tar.gz
mv inception_v3.ckpt ${CHECKPOINT_DIR}
rm inception_v3_2016_08_28.tar.gz




DATASET_DIR=/home/jcq/models-master/research/slim/tmp/data/flowers
TRAIN_DIR=/home/jcq/models-master/research/slim/tmp/flowers-models/inception_v3
CHECKPOINT_PATH=/home/jcq/models-master/research/slim/tmp/checkpoints/inception_v3.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits



# Evaluating performance of a model
# rename the validation files

CHECKPOINT_FILE =/home/jcq/models-master/research/slim/tmp/checkpoints/inception_v3.ckpt
 python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=/home/jcq/models-master/research/slim/tmp/checkpoints/inception_v3.ckpt \
    --dataset_dir=/home/jcq/models-master/research/slim/tmp/data/flowers \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_v3



# Exporting the Inference Graph


/home/jcq/.conda/envs/tensorflow_gpu/bin/python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=/home/jcq/models-master/research/slim/tmp/inception_v3_inf_graph.pb

python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v1 \
  --image_size=224 \
  --output_file=/tmp/mobilenet_v1_224.pb




# Freezing the exported Graph

bazel build /home/jcq/.conda/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/inception_v3_inf_graph.pb \
  --input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
  --input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1





/home/jcq/.conda/envs/tensorflow_gpu/bin/python -u /home/jcq/.conda/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py  --input_graph=/home/jcq/models-master/research/slim/tmp/inception_v3_inf_graph.pb   --input_checkpoint=/home/jcq/models-master/research/slim/tmp/checkpoints/inception_v3.ckpt \--input_binary=true --output_graph=/home/jcq/models-master/research/slim/tmp/frozen_inception_v3.pb   --output_node_name=InceptionV3/Predictions/Reshape_1

## run oks





















