import os
import ntpath
import numpy as np
from tqdm import tqdm
from glob import glob
import scipy.misc as misc

images = glob('img_align_celeba/*.jpg')

def crop_center(img,cropx=64,cropy=64):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

for i in tqdm(images):
   filename = ntpath.basename(i)
   img = misc.imread(i)
   img = misc.imresize(img, (96,96))
   img = crop_center(img)
   misc.imsave('img_align_celeba_cropped/'+filename, img)
