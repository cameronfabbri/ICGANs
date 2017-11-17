'''

   Encoder that encodes an image to z

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
from nets import *

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--DATASET',    required=False,help='The DATASET to use',      type=str,default='galaxy')
   parser.add_argument('--LOSS',       required=False,help='What type of GAN',        type=str,default='wgan')
   parser.add_argument('--DATA_DIR',   required=False,help='Directory where data is', type=str,default='./')
   parser.add_argument('--EPOCHS',  required=False,help='Maximum training steps',  type=int,default=200)
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',              type=int,default=64)
   parser.add_argument('--ACTIVATION', required=False,help='Activation function',     type=str,default='lrelu')
   a = parser.parse_args()

   LOSS           = a.LOSS
   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   EPOCHS      = a.EPOCHS
   BATCH_SIZE     = a.BATCH_SIZE
   ACTIVATION     = a.ACTIVATION

   CHECKPOINT_DIR = 'checkpoints/encoder_z/DATASET_'+DATASET+'/ACTIVATION_'+ACTIVATION+'/LOSS_'+LOSS+'/'
   
   try: os.makedirs(CHECKPOINT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   images      = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='images')
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
   pkl_file = open(DATA_DIR+'data.pkl')
   
   # dictionary of [image_name] = [batch_y,batch_z]
   data           = pickle.load(pkl_file)
   #images_        = data.keys()
   #t              = data.values()

   latents = []
   images_ = []

   for k,v in data.iteritems():
      images_.append(k)
      latents.append(v[1])

   images_   = np.asarray(images_)
   latents   = np.asarray(latents)
   train_len = len(latents)
   
   print 'train num:',train_len

   epoch_num = step/(train_len/BATCH_SIZE)

   lr_ = 1e-3
   while epoch_num < EPOCHS:
      
      epoch_num = step/(train_len/BATCH_SIZE)

      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_z      = latents[idx]
      batch_img    = images_[idx]
      batch_images = np.empty((BATCH_SIZE, 64, 64, 3), dtype=np.float32)
      i = 0
      for img in batch_img:
         img = data_ops.normalize(misc.imread(img))
         r = random.random()
         if r < 0.5: img = np.fliplr(img) # randomly flip left right half the time.
         batch_images[i, ...] = img
         i+=1

      if step > 100000 and step < 200000: lr_ = 1e-4
      if step > 200000 and step < 300000: lr_ = 1e-5
      if step > 300000 and step < 400000: lr_ = 1e-6
      if step > 400000: lr_ = 1e-7

      _,l = sess.run([train_op, loss], feed_dict={images:batch_images, z:batch_z, lr:lr_})
      if step%10==0: print 'epoch:',epoch_num,'step:',step,'loss:',l
      step += 1

      if step % 1000 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
