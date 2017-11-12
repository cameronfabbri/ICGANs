'''

   ICGAN

   Run this after all other models are trained in this order:
      - mnist_cgan: generates conditional mnist images from noise
      - enc_z: encodes images to z, i.e the noise vectors from cgan
      - enc_y: encodes images to labels y

   I believe all I have to do is now load up the saved model for the generator,
   then load up an image, encode it, (or directly save out test sample z), and
   swap out the attribute and boom

'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.misc as misc
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import cPickle as pickle
from tqdm import tqdm
import numpy as np
import argparse
import random
import ntpath
import time
import sys
import cv2
import os

sys.path.insert(0, '../ops/')

from tf_ops import *
import data_ops

'''
   Generator network
   batch norm before activation function
'''
def netG(z, y, BATCH_SIZE):

   # concat attribute y onto z
   z = tf.concat([z,y], axis=1)
   print 'z:',z

   z = tcl.fully_connected(z, 4*4*512, activation_fn=tf.identity, scope='g_z')
   z = tf.reshape(z, [BATCH_SIZE, 4, 4, 512])
   z = tcl.batch_norm(z)
   z = tf.nn.relu(z)
   
   conv1 = tcl.convolution2d_transpose(z, 256, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')
   conv2 = tcl.convolution2d_transpose(conv1, 128, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')
   conv3 = tcl.convolution2d_transpose(conv2, 64, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')
   conv4 = tcl.convolution2d_transpose(conv3, 3, 5, 2, activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv4')

   print 'z:',z
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print
   print 'END G'
   print
   return conv4


if __name__ == '__main__':
   
   BATCH_SIZE = 1

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=False,help='The generator checkpoint to load',type=str,default='mnist')
   parser.add_argument('--DATASET',        required=False,help='The dataset to use',              type=str,default='mnist')
   parser.add_argument('--DATA_DIR',       required=False,help='Directory where data is',         type=str,default='./')
   parser.add_argument('--OUT_DIR',        required=True,help='Directory to save data in',        type=str)
   parser.add_argument('--NUM_GEN',        required=True,help='Directory to save data in',        type=str)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   OUT_DIR        = a.OUT_DIR
   NUM_GEN        = a.NUM_GEN
   IMAGES_DIR     = OUT_DIR
   
   try: os.makedirs(IMAGES_DIR)
   except: pass

   # placeholders for data going into the network
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 9), name='y')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE)

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


   print 'Loading data...'

   '''
      When the encoder is done training, create a test set of x:z by encoding
      a bunch of test images. Save out the latent vectors and path to image,
      no need to save the image again.
   '''

   pkl_file = open(DATA_DIR+'data.pkl')
   
   data     = pickle.load(pkl_file)
   images_  = data.keys()
   t        = data.values()

   #encodings, labels = zip(*t)
   labels, encodings = zip(*t)

   test_len = len(images_)

   images_ = np.asarray(images_)
   encodings = np.asarray(encodings)
   labels = np.asarray(labels)

   print 'Done'
   print

   print 'Generating data...'
   for n in tqdm(range(int(NUM_GEN))):

      idx = np.random.choice(np.arange(test_len), 1, replace=False)

      original_image = images_[idx]
      label          = labels[idx]
      z_             = encodings[idx]

      original_image = misc.imread(original_image[0])
      #original_image = data_ops.normalize(original_image)

      label = np.reshape(label, (1,9))
      z_    = np.reshape(z_, (1,100))

      reconstruction = np.squeeze(sess.run(gen_images, feed_dict={z:z_, y:label}))

      misc.imsave(IMAGES_DIR+str('000')+str(n)+'_o.png', original_image)
      misc.imsave(IMAGES_DIR+str('000')+str(n)+'_r.png', reconstruction)

      print label

      # bald, bangs, black_hair, blond_hair, brown_hair, eyeglasses, goatee, gray_hair, heavy_makeup, male, mustache, no_beard, smiling, wearing_hat, wearing_necklace
      new_y = np.zeros((9))
      new_y[-4] = 1 # no beard
      new_y[5] = 1 # glasses
      new_y = np.expand_dims(new_y, 0)
      print new_y
      
      new_image = np.squeeze(sess.run(gen_images, feed_dict={z:z_, y:new_y}))
      misc.imsave(IMAGES_DIR+str('000')+str(n)+'_n.png', new_image)

      exit()

      new_y = np.expand_dims(np.zeros((10)),0)
      r = random.randint(0,9)
      new_y[0][r] = 1
      true_index = np.argmax(label[0])
      new_index  = np.argmax(new_y[0])

      while new_index == true_index:
         new_y = np.expand_dims(np.zeros((10)),0)
         r = random.randint(0,9)
         new_y[0][r] = 1
         true_index = np.argmax(label[0])
         new_index  = np.argmax(new_y[0])

      #print 'label:',label
      #print 'new_y:',new_y
      
      new_gen = np.squeeze(sess.run(gen_images, feed_dict={z:z_, y:new_y}))
      plt.imsave(IMAGES_DIR+str('000')+str(n)+'_n.png', np.squeeze(new_gen))

      #print 'should be a',np.argmax(new_y[0]),'!'
      #print

   


