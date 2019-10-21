


python /home/karljiang/TensorFlow/Project/Vehicles_detection/dataset/create_pascal_tf_record.py \
--data_dir=/home/karljiang/TensorFlow/data/Vehicles_Detection/VOCdevkit \
--year=VOC2012 \
--set=train \
--output_path=/home/karljiang/TensorFlow/Project/Vehicles_detection/record/pascal_train.record \
--label_map_path=/home/karljiang/TensorFlow/Project/Vehicles_detection/dataset/pascal_label_map.pbtxt



python /home/karljiang/TensorFlow/Project/Vehicles_detection/dataset/create_pascal_tf_record.py \
--data_dir=/home/karljiang/TensorFlow/data/Vehicles_Detection/VOCdevkit \
--year=VOC2012 \
--set=val \
--output_path=/home/karljiang/TensorFlow/Project/Vehicles_detection/record/pascal_val.record \
--label_map_path=/home/karljiang/TensorFlow/Project/Vehicles_detection/dataset/pascal_label_map.pbtxt

