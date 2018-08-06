#####################################################################################################
# testing VGG face model using a pre-trained model
#
#####################################################################################################

import os
from vgg_face import vgg_face
from scipy.misc import imread, imresize
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# build the graph
graph = tf.Graph()
with graph.as_default():
    input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
    output, average_image, class_names = vgg_face('vgg-face.mat', input_maps)


root_path = "/data1/datasets/PeopleInfo/day2"

with tf.Session(graph=graph) as sess:

    for dir in os.listdir(root_path):
        dir_path = root_path + '/' + dir
        npy_name = dir + '.npy'
        print dir_path
        feature_list = []

        for file in os.listdir(dir_path + '/Screensave'):
            image_path = dir_path + '/Screensave/' + file
            try:
                img = imread(image_path, mode='RGB')
            except:
                print("cannot open image file %s") % image_path
                continue

            img = img[0:250, :, :]
            img = imresize(img, [224, 224])
            img = img - average_image

            out = sess.run(output, feed_dict={input_maps: [img]})
            feature = np.squeeze(out['relu7'])
            assert len(feature) == 4096
            feature_list.append(feature)

        num = len(feature_list)
        feature_array = np.zeros((num, 4096))
        for i in range(num):
            feature_array[i,:] = feature_list[i]
        np.save(dir_path + '/' + npy_name, feature_array)

