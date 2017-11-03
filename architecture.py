import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys
from tf_ops import *


def conv_cond_concat(x, y):
   """Concatenate conditioning vector on feature map axis."""
   x_shapes = x.get_shape()
   y_shapes = y.get_shape()
   return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

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

      # repeat y so it is 64x64 and concatenate it as a channel
      input_images = conv_cond_concat(input_images, y)
      print 'input_images:',input_images
      exit()

      conv1 = tcl.conv2d(input_images, 64, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
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
