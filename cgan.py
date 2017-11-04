'''
   conditional gan
'''
import scipy.misc as misc
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tf_ops import *
import numpy as np
import argparse
import data_ops
import random
import ntpath
import time
import sys
import cv2
import os

'''
   Generator network
   batch norm before activation function
'''
def netG(z, y, BATCH_SIZE):

   # concat attribute y onto z
   z = tf.concat([z,y], axis=1)
   print 'z:',z

   z = tcl.fully_connected(z, 4*4*1024, activation_fn=tf.identity, scope='g_z')
   z = tf.reshape(z, [BATCH_SIZE, 4, 4, 1024])
   z = tcl.batch_norm(z)
   z = tf.nn.relu(z)
   
   conv1 = tcl.convolution2d_transpose(z, 512, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')
   conv2 = tcl.convolution2d_transpose(conv1, 256, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')
   conv3 = tcl.convolution2d_transpose(conv2, 128, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')
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
   parser.add_argument('--LOSS',    required=False,help='Type of GAN loss to use',default='wgan')
   parser.add_argument('--DATASET',    required=False,help='The DATASET to use',default='celeba')
   parser.add_argument('--DATA_DIR',   required=False,help='Directory where data is',default='./')
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',type=int,default=64)
   parser.add_argument('--MAX_STEPS', required=False,help='Maximum training iterations',type=int,default=100000)
   a = parser.parse_args()

   LOSS           = a.LOSS
   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   BATCH_SIZE     = a.BATCH_SIZE
   MAX_STEPS      = a.MAX_STEPS

   CHECKPOINT_DIR = 'checkpoints/DATASET_'+DATASET+'/LOSS_'+LOSS+'/'
   IMAGES_DIR     = CHECKPOINT_DIR+'images/'
   
   try: os.makedirs(IMAGES_DIR)
   except: pass

   # size of the attribute depends on the dataset
   if DATASET == 'celeba': y_size = 15
   if DATASET == 'mnist':  y_size = 1

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='real_images')
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_size), name='z')

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
   
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   n_critic = 5

   print 'Loading data...'
   images, annots = data_ops.load_celeba(DATA_DIR, mode='train')

   while step < MAX_STEPS:
      
      start = time.time()

      # train the discriminator
      for critic_itr in range(n_critic):
         batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y = random.sample(annots, BATCH_SIZE)
         sess.run(D_train_op, feed_dict={z:batch_z, y:batch_y})

      print 'here'
      exit()
      # now train the generator once! use normal distribution, not uniform!!
      batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      sess.run(G_train_op, feed_dict={z:batch_z})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={z:batch_z})
      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
         batch_z  = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z})

         data_ops.saveImage(gen_imgs[0], step, IMAGES_DIR)
         print 'Done saving'



