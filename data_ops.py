'''

   Operations commonly used in tensorflow

'''

import tensorflow as tf
import numpy as np
import math




'''
   Converts a single image from [0,255] range to [-1,1]
'''
def preprocess(image):
   with tf.name_scope('preprocess'):
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      return (image/127.5)-1.0

'''
   Converts a single image from [-1,1] range to [0,255]
'''
def deprocess(image):
   with tf.name_scope('deprocess'):
      return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)
   
'''
   Converts a batch of images from [-1,1] range to [0,255]
'''
def batch_deprocess(images):
   with tf.name_scope('batch_deprocess'):
      return tf.map_fn(deprocess, images, dtype=tf.uint8)

'''
   Converts a batch of images from [0,255] to [-1,1]
'''
def batch_preprocess(images):
   with tf.name_scope('batch_preprocess'):
      return tf.map_fn(preprocess, images, dtype=tf.float32)
