'''

   Generates a dataset of encodings from real images using the trained encoder.

'''
import tensorflow.contrib.layers as tcl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.misc as misc
import tensorflow as tf
import cPickle as pickle
from tqdm import tqdm
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

def batch(iterable, n=1):
   l = len(iterable)
   for ndx in range(0, l, n):
      yield iterable[ndx:min(ndx + n, l)]

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
  
   BATCH_SIZE = 64

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   images      = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='images')

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

   test_len = len(test_annots)
   print 'test num:',test_len

   info = {}
   
   for x in batch(train_images, BATCH_SIZE):
      if x.shape[0] < 64: break
      batch_images = []
      for im in x:
         img = misc.imread(im).astype('float32')
         batch_images.append(img)
      batch_images = np.asarray(batch_images)
      encoding = sess.run([encoded], feed_dict={images:batch_images})[0]
      for ip,e in zip(x,encoding):
         info[ip] = e

   '''   
   # want to write out a file with the image path and z vector
   for image_path,label in tqdm(zip(test_images, test_annots)):
      img          = data_ops.normalize(misc.imread(image_path))
      batch_images = np.expand_dims(img, 0)
      encoding     = sess.run([encoded], feed_dict={images:batch_images})[0]
      info[image_path] = [encoding, label]
   '''
   
   # write out dictionary to pickle file
   p = open(OUTPUT_DIR+'data.pkl', 'wb')
   data = pickle.dumps(info)
   p.write(data)
   p.close()

