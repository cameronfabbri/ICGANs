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

   encoded = encZ(images, ACTIVATION)

   # l2 loss on encoded and actual latent representation, z
   loss = tf.nn.l2_loss(encoded-z)

   tf.summary.scalar('loss', loss)
   merged_summary_op = tf.summary.merge_all()

   train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

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

   while step < MAX_STEPS:

      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_images = mimages[idx]
      batch_z      = latents[idx]

      _,l = sess.run([train_op, loss], feed_dict={images:batch_images, z:batch_z})
      print 'step:',step,'loss:',l
      step += 1
    
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
