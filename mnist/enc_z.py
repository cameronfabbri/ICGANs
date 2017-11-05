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

'''
   Encoder
'''
def encZ(x, BATCH_SIZE):

   conv1 = tcl.conv2d(x, 64, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
   conv1 = lrelu(conv1)
   
   conv2 = tcl.conv2d(conv1, 128, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
   conv2 = lrelu(conv2)

   conv3 = tcl.conv2d(conv2, 256, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
   conv3 = lrelu(conv3)

   conv4 = tcl.conv2d(conv3, 512, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
   conv4 = lrelu(conv4)

   conv5 = tcl.conv2d(conv4, 1, 4, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')

   print 'input images:',input_images
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print 'conv5:',conv5
   print 'END ENCODER\n'
   exit()

   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', conv5)

   return conv5



if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--DATASET',    required=False,help='The DATASET to use',      type=str,default='celeba')
   parser.add_argument('--DATA_DIR',   required=False,help='Directory where data is', type=str,default='./')
   parser.add_argument('--MAX_STEPS',  required=False,help='Maximum training steps',  type=int,default=100000)
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',              type=int,default=64)
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   a = parser.parse_args()

   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   MAX_STEPS      = a.MAX_STEPS
   BATCH_SIZE     = a.BATCH_SIZE
   CHECKPOINT_DIR = a.CHECKPOINT_DIR

   IMAGES_DIR     = CHECKPOINT_DIR+'images/'
   
   try: os.makedirs(IMAGES_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name='images')
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')

   encoded = encZ(images, BATCH_SIZE)

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

   m_images = glob.glob('output_mnist/*.png')
   sorted(m_images)
   # TODO ended here. Need to get all images and their encoded value z


   train_len = len(annots)
   test_len  = len(test_annots)

   print 'train num:',train_len

   while step < MAX_STEPS:
      
      start = time.time()

      # train the discriminator
      for critic_itr in range(n_critic):
         idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = annots[idx]
         batch_images = images[idx]
         sess.run(D_train_op, feed_dict={z:batch_z, y:batch_y, real_images:batch_images})
      
      # now train the generator once! use normal distribution, not uniform!!
      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      batch_y      = annots[idx]
      batch_images = images[idx]
      sess.run(G_train_op, feed_dict={z:batch_z, y:batch_y, real_images:batch_images})

      # now get all losses and summary *without* performing a training step - for tensorboard and printing
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={z:batch_z, y:batch_y, real_images:batch_images})
      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')

         idx          = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = test_annots[idx]
         batch_images = test_images[idx]
         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:batch_y, real_images:batch_images})[0][0]

         num = np.argmax(batch_y[0])
         plt.imsave(IMAGES_DIR+'step_'+str(step)+'_num_'+str(num)+'.png', np.squeeze(gen_imgs), cmap=cm.gray)



