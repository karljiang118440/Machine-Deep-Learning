##_*_coding:utf-8_*_ 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import hashlib
import io
import logging
import os
 
from lxml import etree
import PIL.Image
import tensorflow as tf
 
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
 
#执行参数，我们可以在这里修改，也可以在执行时候带上参数修改，建议带上参数修改
flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS
 
SETS = ['train', 'val', 'trainval', 'test']
#增加我们自己的数据集my_VOC
#YEARS = ['VOC2007', 'VOC2012', 'my_VOC']
YEARS = 'VOC2012'
 
 
def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  #确定照片的路径，这个我调试了很久，一直找未找此路径的文件，建议将路径输出，看看是否正确。
  #这里有个大坑，官方的XML标注里面，filename字段是后面有文件类型的，但是用labelImg标注是没有的
  #我们在img_path里手动拼接   +'.jpg'  
  img_path = os.path.join('VOC2012',data['folder'], data['filename']+'.jpg')
  full_path = os.path.join(dataset_directory, img_path)
  #手动输入查看路径是否正确
  print('full_path',full_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()
 
  width = int(data['size']['width'])
  height = int(data['size']['height'])
 
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue
 
    difficult_obj.append(int(difficult))
 
    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))
 
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example
 
 
def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))
 
  data_dir = FLAGS.data_dir
  #新增我们的数据集到years中
  #years = ['VOC2007','VOC2012','my_VOC']
  years = 'VOC2012'

  if FLAGS.year != 'merged':
    years = [FLAGS.year]
  print('data_dir=',data_dir)
  print('years=',years)
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
 
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
 
  for year in years:
    logging.info('Reading from PASCAL %s dataset.', year)
    #修改成如下代码，这里只需要用到Main下面的train.txt，val.txt等4个文件
    #原来的代码是用了官方下面的XX_train.txt等文件
    examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main/'
                                  + FLAGS.set + '.txt')
    annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples_list))
      path = os.path.join(annotations_dir, example + '.xml')
      with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
 
      tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances)
      writer.write(tf_example.SerializeToString())
 
  writer.close()
 
 
if __name__ == '__main__':
  tf.app.run()

