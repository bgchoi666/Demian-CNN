from PIL import Image
import numpy as np
import os

#im = Image.open('test.png')
#imGray_sacle = im.convert('L')
#im_arr = np.array(imGray_sacle)/255

#print(im)

inputs = []
for root, directory, files in os.walk('image_files'):
  for fname in files:
    im = Image.open(root + "/" + fname)
    imGray_sacle = im.convert('L') # 흑백으로 전환
    im_arr = np.array(imGray_sacle)
    inputs.append(np.reshape(im_arr, [-1])/255)

features = np.reshape(inputs, [-1, len(inputs[0])])


# read label file, make labels
labels = []

