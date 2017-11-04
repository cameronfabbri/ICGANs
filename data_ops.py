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

   len_train = 100
   len_test = 100

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

      only considering: bald, bangs, black_hair, blond_hair, brown_hair, eyeglasses, goatee, gray_hair, heavy_makeup, male, mustache, no_beard, smiling, wearing_hat, wearing_necklace
      4,5,8,9,11,15,16,17,18,20,22,24,31,35,37
   '''
   i = 0
   j = 0
   dum = 0
   count = 0
   train_attr = []
   print 'Loading attributes...'
   with open(data_dir+'list_attr_celeba.txt', 'r') as f:
      for line in tqdm(f):
         line = line.rstrip().split()
         if dum < 2:
            dum += 1
            continue
         image_id = line[0]
         if image_id in train_ids:
            attr = line[1:]
            attr = np.asarray(list(attr[x] for x in [4,5,8,9,11,15,16,17,18,20,22,24,31,35,37]), dtype=np.float32)
            train_attr.append(attr)
         if image_id in test_ids:
            attr = line[1:]
            attr = np.asarray(list(attr[x] for x in [4,5,8,9,11,15,16,17,18,20,22,24,31,35,37]), dtype=np.float32)
            test_attr.append(attr)
         if count == 100: break
         count += 1

   train_images = np.empty((len_train, 64, 64, 3), dtype=np.float32)
   test_images  = np.empty((len_test, 64, 64, 3), dtype=np.float32)

   images = glob.glob(data_dir+'img_align_celeba_resized/*.jpg')

   i = 0
   j = 0
   count = 0
   print 'Loading images...'
   for image in tqdm(images):
      image_id = ntpath.basename(image)
      if image_id in train_ids:
         img = misc.imread(image)
         train_images[i,...] = img
         i+=1
      if image_id in test_ids:
         img = misc.imread(image)
         train_images[j,...] = img
         j+=1
      if count == 100: break
      count += 1

   if mode == 'train': return train_images, train_attr
   if mode == 'test': return train_images, train_attr
   return 'mode error'

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
