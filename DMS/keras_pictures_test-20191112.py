import os
import matplotlib.pyplot as plt
import random

from keras import backend as K
from matplotlib import pyplot as plt
from keras.models import load_model
import cv2
import  tensorflow as tf
import  numpy as np
import time

import glob

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import *

model_image_size = (224, 224)
fine_tune_layer = 22
final_layer = 24
visual_layer = 21

model = load_model("models/vgg19-imagenet-finetune{}.h5".format(fine_tune_layer))
print("load successed")

#SVG(model_to_dot(model).create(prog='dot', format='svg'))


img_height = 224 # Height of the input images  # karl-5: 修改尺寸大小300 > 224
img_width = 224 # Width of the input images    # karl-5: 修改尺寸大小480 > 224




    # add 模型

weights = model.layers[final_layer].get_weights()[0]
layer_output = model.layers[visual_layer].output
model2 = Model(model.input, [layer_output, model.output])
print("layer_output {0}".format(layer_output))
print("weights shape {0}".format(weights.shape))



test_dir="./test"
image_files = glob.glob(os.path.join(test_dir,"*"))
print(len(image_files))

plt.figure(figsize=(12, 24))
for i in range(10):
	plt.subplot(5, 2, i+1)
	#img = cv2.imread(image_files[2000*i+113])
	image = cv2.imread(image_files[i])
	img = cv2.resize(image,  (model_image_size[1],model_image_size[0]))
	x = img.copy()
	x.astype(np.float32)
	out, predictions = model2.predict(np.expand_dims(x, axis=0))   #model_show ==》 model2 修改调用的模型
	predictions = predictions[0]
	out = out[0]
	
	max_idx = np.argmax(predictions)
	prediction = predictions[max_idx]
    
	print("c=%d' %(max_idx) ")
    
    
	status = ["safe driving",  " texting - right",  "phone - right",  "texting - left",  "phone - left",  
                "operation radio", "drinking", "reaching behind", "hair and makeup", "talking"]
	plt.title('c%d |%s| %.2f%%' % (max_idx , status[max_idx], prediction*100))
	print("c%d |%s| %.2f%% "%(max_idx, status[max_idx], prediction*100))

    
	cv2.imshow("img",image)  
	#plt.imshow(out)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	time.sleep(1)





