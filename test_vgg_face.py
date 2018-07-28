#####################################################################################################
# testing VGG face model using a pre-trained model
#
#####################################################################################################

import os
from vgg_face import vgg_face
from scipy.misc import imread, imresize
import tensorflow as tf
import numpy as np
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# build the graph
graph = tf.Graph()
with graph.as_default():
    input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
    output, average_image, class_names = vgg_face('vgg-face.mat', input_maps)

with tf.Session(graph=graph) as sess:

    for dir in os.listdir('PeopleInfo'):
        feature_list = []
        for file in os.listdir('PeopleInfo/' + dir):
            image_path = 'PeopleInfo/' + dir + '/' + file
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
        feature_array = np.zeros((len(feature_list), 4096))
        for i in range(len(feature_list)):
            feature_array[i,:] = feature_list[i]
        print "save .npy file of " + dir
        np.save('data/%s.npy' % dir, feature_array)

