'''

   Generates a dataset of encodings from real images using the trained encoder.

'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.misc as misc
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import cPickle as pickle
import numpy as np
import argparse
import random
import ntpath
import glob
import time
import sys
import cv2
import os

sys.path.insert(0, '../ops/')

from tf_ops import *
import data_ops

def activate(x, ACTIVATION):
   if ACTIVATION == 'lrelu': return lrelu(x)
   if ACTIVATION == 'relu':  return relu(x)
   if ACTIVATION == 'elu':   return elu(x)
   if ACTIVATION == 'swish': return swish(x)

'''
   Encoder
'''
def encZ(x, ACTIVATION):

   conv1 = tcl.conv2d(x, 64, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv1')
   conv1 = activate(conv1, ACTIVATION)
   
   conv2 = tcl.conv2d(conv1, 128, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv2')
   conv2 = activate(conv2, ACTIVATION)

   conv3 = tcl.conv2d(conv2, 256, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv3')
   conv3 = activate(conv3, ACTIVATION)

   conv4 = tcl.conv2d(conv3, 512, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv4')
   conv4 = activate(conv4, ACTIVATION)

   conv4_flat = tcl.flatten(conv4)

   fc1 = tcl.fully_connected(conv4_flat, 4096, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc1')
   fc1 = activate(fc1, ACTIVATION)

   fc2 = tcl.fully_connected(fc1, 100, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc2')
   
   print 'input:',x
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print 'fc1:',fc1
   print 'fc2:',fc2
   print 'END ENCODER\n'
   
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', fc1)
   tf.add_to_collection('vars', fc2)

   return fc2


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   parser.add_argument('--DATASET',    required=False,help='The DATASET to use',      type=str,default='celeba')
   parser.add_argument('--DATA_DIR',   required=False,help='Directory where data is', type=str,default='./')
   parser.add_argument('--OUTPUT_DIR', required=False,help='Directory to save data', type=str,default='./')
   parser.add_argument('--ACTIVATION', required=False,help='Activation function',     type=str,default='lrelu')
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   OUTPUT_DIR     = a.OUTPUT_DIR
   ACTIVATION     = a.ACTIVATION

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   images      = tf.placeholder(tf.float32, shape=(1, 64, 64, 3), name='images')

   encoded = encZ(images, ACTIVATION)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   print 'Loading data...'

   # images and annots: _, __
   train_images, train_annots, test_images, test_annots = data_ops.load_celeba(DATA_DIR)

   test_images = train_images
   test_annots = train_annots
   
   test_len = len(test_annots)
   print 'test num:',test_len

   # want to write out a file with the image path and z vector
   for image_path in test_images:

      img          = misc.imread(image_path).astype('float32')
      batch_images = np.expand_dims(img, 0)

      encoding = sess.run([encoded], feed_dict={images:batch_images})[0][0]
      print image_path
      print encoding
      exit()
    
