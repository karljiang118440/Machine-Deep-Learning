# -*- coding: utf-8 -*-
"""
Created on Wed May  2 23:03:57 2018
@author: xiaozhe
"""

import cv2
import os
import numpy as np
import _thread as thread
import logging
import random
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.utils import np_utils

logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def resize_pic(path, width, height):
    # path = '.\\pics_dog'
    # width = 300
    # height = 200
    for num, pic in enumerate(os.listdir(path), 1):
        logging.debug(pic) if num % 100 == 0 else None
        pic = os.path.join(path, pic)
        img = cv2.imread(pic, cv2.IMREAD_UNCHANGED)
        try:
            img = cv2.resize(img, (width, height))
            cv2.imwrite(pic, img)

        except Exception as e:
            logging.debug(e)
            os.remove(pic)


def generate_data(path, samples_num):
    logging.debug('Prepare data')
    #    path = '.\\pics_dog'
    #    samples_num = 1000
    data = np.empty((samples_num, 200, 300, 3))
    labels = np.empty((samples_num,))
    path_list = os.listdir(path)
    random.shuffle(path_list)
    #    ii, jj = 0, 0
    for i, pic in enumerate(path_list):
        if i >= samples_num:
            break
        pic = os.path.join(path, pic)
        img = cv2.imread(pic, cv2.IMREAD_UNCHANGED)
        if os.path.split(pic)[-1].split('.')[0] == 'dog':
            labels[i] = 1
            data[i] = img
        #            print(labels[i])
        #            ii += 1
        elif os.path.split(pic)[-1].split('.')[0] == 'cat':
            labels[i] = 0
            data[i] = img
        #            print(labels[i])
        #            jj += 1
        print(i, pic, labels[i])
    labels = np_utils.to_categorical(labels, 2)
    return data, labels


def load_data(path, image_size, sample_num):
    #    path = './data/train/'
    files = os.listdir(path)
    random.shuffle(files)
    files = files[:sample_num]
    images = []
    labels = []
    for i, f in enumerate(files, 1):

        img_path = os.path.join(path, f)
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img)
        images.append(img_array)

        if 'cat' in f:
            labels.append(0)
        else:
            labels.append(1)
        logging.debug(str(i) + ' ' + f + ': ' +
                      str(labels[-1])) if i % 500 == 0 else None
    data = np.array(images)
    labels = np.array(labels).reshape(len(labels), 1)
    #
    #    labels = np_utils.to_categorical(labels, 2)
    return data, labels


if __name__ == '__main__':
    path_dog = os.path.join(os.getcwd(), 'train')
    path_cat = os.path.join(os.getcwd(), 'test1')
    thread.start_new_thread(resize_pic, (path_cat, 128, 128,))
    thread.start_new_thread(resize_pic, (path_dog, 128, 128,))
    # resize_pic(path_cat, 224, 224)
    # resize_pic(path_dog, 224, 224)
    print('Finished')
