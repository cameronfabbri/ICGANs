'''

   Operations commonly used in tensorflow

'''

import tensorflow as tf
import numpy as np
import math
import glob
import scipy.misc as misc
import ntpath
from tqdm import tqdm
import os
import requests
import gzip
import cPickle as pickle

'''
   Helper function that returns string names for the attributes

   0     1      2           3           4           5           6       7          8             9     10        11        12       13           14
   bald, bangs, black_hair, blond_hair, brown_hair, eyeglasses, goatee, gray_hair, heavy_makeup, male, mustache, no_beard, smiling, wearing_hat, wearing_necklace
   4,5,8,9,11,15,16,17,18,20,22,24,31,35,37

'''
def get_attr_name(attr):
   s = ''
   if attr[0] == -1: s+='not bald, '
   else: s+='bald, '
   
   if attr[1] == -1: s+='no bangs, '
   else: s+='bangs, '
   
   if attr[2] == -1: s+='no black hair, '
   else: s+='black hair, '
   
   if attr[3] == -1: s+='no blonde hair, '
   else: s+='black hair, '
   
   if attr[4] == -1: s+='no brown hair, '
   else: s+='brown hair, '
   
   if attr[5] == -1: s+='no eyeglasses, '
   else: s+='eyeglasses, '
   
   if attr[6] == -1: s+='no goatee, '
   else: s+='goatee, '
   
   if attr[7] == -1: s+='no gray hair, '
   else: s+='gray hair, '
   
   if attr[8] == -1: s+='no heavy makeup, '
   else: s+='heavy makeup, '
   
   if attr[9] == -1: s+='not male, '
   else: s+='male, '
   
   if attr[10] == -1: s+='no mustache, '
   else: s+='mustache, '
   
   if attr[11] == -1: s+='beard, '
   else: s+='no beard, '
   
   if attr[12] == -1: s+='not smiling, '
   else: s+='smiling, '
   
   if attr[13] == -1: s+='not wearing hat, '
   else: s+='wearing hat, '
   
   if attr[14] == -1: s+='not wearing necklace, '
   else: s+='wearing necklace'
   
   return s

def load_mnist(data_dir, mode='train'):

   url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
   # check if it's already downloaded
   if not os.path.isfile(data_dir+'/mnist.pkl.gz'):
      print 'Downloading mnist...'
      with open('mnist.pkl.gz', 'wb') as f:
         r = requests.get(url)
         if r.status_code == 200:
            f.write(r.content)
         else:
            print 'Could not connect to ', url

   print 'opening mnist'
   f = gzip.open('mnist.pkl.gz', 'rb')
   train_set, val_set, test_set = pickle.load(f)
   
   if mode == 'train':
      mnist_train_images = []
      mnist_train_labels = []
      for t,l in zip(*train_set):
         label = np.zeros((10))
         label[l] = 1
         mnist_train_images.append(np.reshape(t, (28, 28, 1)))
         mnist_train_labels.append(label)
      return np.asarray(mnist_train_images), np.asarray(mnist_train_labels)

   if mode == 'val':
      mnist_val_images = []
      mnist_val_labels = []
      for t,l in zip(*val_set):
         label = np.zeros((10))
         label[l] = 1
         mnist_val_images.append(np.reshape(t, (28, 28, 1)))
         mnist_val_labels.append(label)
      return np.asarray(mnist_val_images), np.asarray(mnist_val_labels)

   if mode == 'test':
      mnist_test_images  = []
      mnist_test_labels = []
      for t,l in zip(*test_set):
         label = np.zeros((10))
         label[l] = 1
         mnist_test_images.append(np.reshape(t, (28, 28, 1)))
         mnist_test_labels.append(label)
      return np.asarray(mnist_test_images), np.asarray(mnist_test_labels)

   return 'mode error'


'''
   mode can be train/test/val
'''
def load_celeba(data_dir, mode='train'):

   train_ids = []
   test_ids  = []
   val_ids   = []
   
   with open(data_dir+'list_eval_partition.txt', 'r') as f:
      for line in f:
         line = line.rstrip().split()
         img = line[0]
         id_ = int(line[1])
         if id_ == 0: train_ids.append(img)
         if id_ == 1: test_ids.append(img)
         if id_ == 2: val_ids.append(img)
   
   len_train = len(train_ids)
   len_test  = len(test_ids)
   len_val   = len(val_ids)

   # load up annotations
   '''
      0  5_o_Clock_Shadow
      1  Arched_Eyebrows
      2  Attractive
      3  Bags_Under_Eyes
      4  Bald
      5  Bangs
      6  Big_Lips
      7  Big_Nose
      8  Black_Hair
      9  Blond_Hair
      10 Blurry
      11 Brown_Hair
      12 Bushy_Eyebrows
      13 Chubby
      14 Double_Chin
      15 Eyeglasses
      16 Goatee
      17 Gray_Hair
      18 Heavy_Makeup
      19 High_Cheekbones
      20 Male
      21 Mouth_Slightly_Open
      22 Mustache
      23 Narrow_Eyes
      24 No_Beard
      25 Oval_Face
      26 Pale_Skin
      27 Pointy_Nose
      28 Receding_Hairline
      29 Rosy_Cheeks
      30 Sideburns
      31 Smiling
      32 Straight_Hair
      33 Wavy_Hair
      34 Wearing_Earrings
      35 Wearing_Hat
      36 Wearing_Lipstick
      37 Wearing_Necklace
      38 Wearing_Necktie
      39 Young

      only considering: bald, bangs, black_hair, blond_hair, eyeglasses, heavy_makeup, male, pale_skin, smiling
      4, 5, 8, 9, 15, 18, 20, 26, 31
   '''
   dum = 0
   train_image_attr = {}
   test_image_attr  = {}
   i = 0
   print 'Loading attributes...'
   with open(data_dir+'list_attr_celeba.txt', 'r') as f:
      for line in tqdm(f):
         line = line.rstrip().split()
         if dum < 2:
            dum += 1
            continue
         image_id = line[0]
         if image_id in train_ids or image_id in val_ids:
            attr = line[1:]
            #attr = np.asarray(list(attr[x] for x in [4,5,8,9,11,15,16,17,18,20,22,24,31,35,37]), dtype=np.float32)
            attr = np.asarray(list(attr[x] for x in [ 4, 5, 8, 9, 15, 18, 20, 26, 31]), dtype=np.float32)
            attr = np.asarray([0 if x == -1 else 1 for x in attr])
            train_image_attr[data_dir+'img_align_celeba_cropped/'+image_id] = attr
         if image_id in test_ids:
            attr = line[1:]
            attr = np.asarray(list(attr[x] for x in [4, 5, 8, 9, 15, 18, 20, 26, 31]), dtype=np.float32)
            attr = np.asarray([0 if x == -1 else 1 for x in attr])
            test_image_attr[data_dir+'img_align_celeba_cropped/'+image_id] = attr

         i += 1
         #if i == 70: break

   train_images = train_image_attr.keys()
   train_attrs  = train_image_attr.values()
   test_images  = test_image_attr.keys()
   test_attrs   = test_image_attr.values()
   '''
   if mode == 'train':
      train_images = train_image_attr.keys()
      train_attrs  = train_image_attr.values()
      return np.asarray(train_images), np.asarray(train_attrs)

   if mode == 'test':
      test_images = test_image_attr.keys()
      test_attrs  = test_image_attr.values()
      return np.asarray(test_images), np.asarray(test_attrs)
   '''
   return np.asarray(train_images), np.asarray(train_attrs), np.asarray(test_images), np.asarray(test_attrs)

def normalize(image):
   return (image/127.5)-1.0

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
