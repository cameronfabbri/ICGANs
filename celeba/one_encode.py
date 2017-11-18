'''

   Generates just one encoded image

'''
import matplotlib.pyplot as plt
from tqdm import tqdm
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
from nets import *

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   parser.add_argument('--IMAGE',   required=False,help='Directory where data is', type=str,default='./')
   parser.add_argument('--ACTIVATION', required=False,help='Activation function',     type=str,default='lrelu')
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   IMAGE           = a.IMAGE
   OUTPUT_DIR     = 'one/'
   ACTIVATION     = a.ACTIVATION
   
   try: os.makedirs(OUTPUT_DIR)
   except: pass

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

   image = a.IMAGE
   # bald, bangs, black_hair, blond_hair, eyeglasses, heavy_makeup, male, pale_skin, smiling
   label = np.asarray([1., 0., 1., 0., 1., 0., 1., 0., 0.])

   info = {}
   img              = misc.imread(image).astype('float32')
   batch_images     = np.expand_dims(img, 0)
   encoding         = sess.run([encoded], feed_dict={images:batch_images})[0][0]
   info[image]      = [encoding, label]

   # write out dictionary to pickle file
   p = open(OUTPUT_DIR+'data.pkl', 'wb')
   data = pickle.dumps(info)
   p.write(data)
   p.close()

