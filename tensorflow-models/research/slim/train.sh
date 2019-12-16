
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
