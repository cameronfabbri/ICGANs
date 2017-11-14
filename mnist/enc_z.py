'''

   Encoder that encodes an image to z

'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.misc as misc
import tensorflow as tf
import tensorflow.contrib.layers as tcl
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

def activate(x, ACTIVATION):
   if ACTIVATION == 'lrelu': return lrelu(x)
   if ACTIVATION == 'relu':  return relu(x)
   if ACTIVATION == 'elu':   return elu(x)
   if ACTIVATION == 'swish': return swish(x)

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--DATASET',    required=False,help='The DATASET to use',      type=str,default='celeba')
   parser.add_argument('--DATA_DIR',   required=False,help='Directory where data is', type=str,default='./')
   parser.add_argument('--MAX_STEPS',  required=False,help='Maximum training steps',  type=int,default=100000)
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',              type=int,default=64)
   parser.add_argument('--ACTIVATION', required=False,help='Activation function',     type=str,default='lrelu')
   a = parser.parse_args()

   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   MAX_STEPS      = a.MAX_STEPS
   BATCH_SIZE     = a.BATCH_SIZE
   ACTIVATION     = a.ACTIVATION

   CHECKPOINT_DIR = 'checkpoints/encoder_z/DATASET_'+DATASET+'/ACTIVATION_'+ACTIVATION+'/'
   
   try: os.makedirs(CHECKPOINT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   images      = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name='images')
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   lr          = tf.placeholder(tf.float32, name='learning_rate')

   encoded = encZ(images, ACTIVATION)

   # l2 loss on encoded and actual latent representation, z
   loss = tf.nn.l2_loss(encoded-z)

   tf.summary.scalar('loss', loss)
   merged_summary_op = tf.summary.merge_all()

   train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())

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
   
   ########################################### training portion

   step = sess.run(global_step)

   print 'Loading data...'

   mimages = np.load(DATA_DIR+'images.npy')
   latents = np.load(DATA_DIR+'latents.npy')

   train_len = len(latents)
   print 'train num:',train_len

   lr_ = 1e-4

   while step < MAX_STEPS:

      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_images = mimages[idx]
      batch_z      = latents[idx]

      if step > 25000: lr_ = 1e-5

      _,l = sess.run([train_op, loss], feed_dict={images:batch_images, z:batch_z, lr:lr_})
      print 'step:',step,'loss:',l
      step += 1
    
   print 'Saving model...'
   saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
   saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
