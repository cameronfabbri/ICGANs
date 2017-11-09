'''
   conditional gan
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
import time
import sys
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
   tf.add_to_collection('vars', z)
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   return conv4

'''
   Discriminator network. No batch norm
'''
def netD(input_images, y, BATCH_SIZE, reuse=False):

   print 'DISCRIMINATOR reuse = '+str(reuse)
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      y_dim = int(y.get_shape().as_list()[-1])

      # reshape so it's batchx1x1xy_size
      y = tf.reshape(y, shape=[BATCH_SIZE, 1, 1, y_dim])
      input_ = conv_cond_concat(input_images, y)

      conv1 = tcl.conv2d(input_, 64, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      conv1 = lrelu(conv1)
      
      conv2 = tcl.conv2d(conv1, 128, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      conv2 = lrelu(conv2)

      conv3 = tcl.conv2d(conv2, 256, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      conv3 = lrelu(conv3)

      conv4 = tcl.conv2d(conv3, 512, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      conv4 = lrelu(conv4)

      conv5 = tcl.conv2d(conv4, 1, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')

      print 'input images:',input_images
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5
      print 'END D\n'

      tf.add_to_collection('vars', conv1)
      tf.add_to_collection('vars', conv2)
      tf.add_to_collection('vars', conv3)
      tf.add_to_collection('vars', conv4)
      tf.add_to_collection('vars', conv5)

      return conv5



if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--LOSS',       required=False,help='Type of GAN loss to use', type=str,default='wgan')
   parser.add_argument('--MODE',       required=False,help='train/test/val',          type=str,default='train')
   parser.add_argument('--DATASET',    required=False,help='The DATASET to use',      type=str,default='celeba')
   parser.add_argument('--DATA_DIR',   required=False,help='Directory where data is', type=str,default='./')
   parser.add_argument('--MAX_STEPS',  required=False,help='Maximum training steps',  type=int,default=200000)
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',              type=int,default=64)
   a = parser.parse_args()

   LOSS           = a.LOSS
   MODE           = a.MODE
   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   BATCH_SIZE     = a.BATCH_SIZE
   MAX_STEPS      = a.MAX_STEPS

   CHECKPOINT_DIR = 'checkpoints/DATASET_'+DATASET+'/LOSS_'+LOSS+'/'
   IMAGES_DIR     = CHECKPOINT_DIR+'images/'
   
   try: os.makedirs(IMAGES_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='real_images')
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 15), name='y')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE)

   # get the output from D on the real and fake data
   errD_real = netD(real_images, y, BATCH_SIZE)
   errD_fake = netD(gen_images, y, BATCH_SIZE, reuse=True)

   # cost functions
   errD = tf.reduce_mean(errD_real) - tf.reduce_mean(errD_fake)
   errG = tf.reduce_mean(errD_fake)

   # gradient penalty
   epsilon = tf.random_uniform([], 0.0, 1.0)
   x_hat = real_images*epsilon + (1-epsilon)*gen_images
   d_hat = netD(x_hat, y, BATCH_SIZE, reuse=True)
   gradients = tf.gradients(d_hat, x_hat)[0]
   slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
   gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
   errD += gradient_penalty

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # optimize G
   G_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.0,beta2=0.9).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.0,beta2=0.9).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())

   tf.add_to_collection('G_train_op', G_train_op)
   tf.add_to_collection('D_train_op', D_train_op)
   
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

   n_critic = 5

   print 'Loading data...'
   images, annots, test_images, test_annots = data_ops.load_celeba(DATA_DIR)

   train_len = len(annots)
   test_len  = len(test_annots)

   print 'train num:',train_len
   print 'test num:',test_len

   while step < MAX_STEPS:
      
      start = time.time()

      # train the discriminator
      for critic_itr in range(n_critic):
         idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = annots[idx]
         batch_img    = images[idx]

         batch_images = np.empty((BATCH_SIZE, 64, 64, 3), dtype=np.float32)
         i = 0
         for img in batch_img:
            img = data_ops.normalize(misc.imread(img))
            batch_images[i, ...] = img
            i+=1
         sess.run(D_train_op, feed_dict={z:batch_z, y:batch_y, real_images:batch_images})
      
      # now train the generator once! use normal distribution, not uniform!!
      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      batch_y      = annots[idx]
      batch_img    = images[idx]
      batch_images = np.empty((BATCH_SIZE, 64, 64, 3), dtype=np.float32)
      # gotta read the batch of images
      i = 0
      for img in batch_img:
         img = data_ops.normalize(misc.imread(img))
         batch_images[i, ...] = img
         i+=1

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
         batch_img    = test_images[idx]
         batch_images = np.empty((BATCH_SIZE, 64, 64, 3), dtype=np.float32)
         '''
         idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = annots[idx]
         batch_img    = images[idx]
         batch_images = np.empty((BATCH_SIZE, 64, 64, 3), dtype=np.float32)
         '''

         i = 0
         for img in batch_img:
            img = data_ops.normalize(misc.imread(img))
            batch_images[i, ...] = img
            i+=1

         # comes out as (1, batch, 64, 64, 3), so squeezing it
         gen_imgs = np.squeeze(np.asarray(sess.run([gen_images], feed_dict={z:batch_z, y:batch_y, real_images:batch_images})))

         num = 0
         for img,atr in zip(gen_imgs, batch_y):
            img = (img+1.)
            img *= 127.5
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.reshape(img, (64, 64, -1))
            misc.imsave(IMAGES_DIR+'step_'+str(step)+'_num_'+str(num)+'.png', img)
            with open(IMAGES_DIR+'attrs.txt', 'a') as f:
               f.write('step_'+str(step)+'_num_'+str(num)+','+str(atr)+'\n')
            num += 1
            if num == 5: break
