# Copyright 2018 Bimghi Choi. All Rights Reserved.
# ==============================================================================


"""Utilities for read and parsing s&p500 text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
import random as rd
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler as StandardScaler

def read_image_file(config, data_path=None):
  """
  in given directory, read all files, convert them to a normalize dataset

  :param config: a configuration object
  :param data_path: not used
  :return: tuple (train_data, valid_data, test_data)
    where each of the data objects can be passed to producer function.
  """

  inputs = []
  for root, directory, files in os.walk('image_files'):
      for fname in files:
        im = Image.open(root + "/" + fname)
        imGray_sacle = im.convert('L') # 흑백으로 전환
        im_arr = np.array(imGray_sacle)
        norm_img = normalize(im_arr)
        inputs.append(np.reshape(norm_img[:25, :25], [-1]))

  features = np.reshape(inputs, [-1, len(inputs[0])])

  #read data file, and make dataframe
  df = pd.read_csv("label" + ".csv", encoding="ISO-8859-1")

  # create train dataset
  labels = np.array(np.reshape(df.values[:, 0], [-1]))

  inputs = []
  for root, directory, files in os.walk('image_files_test'):
      for fname in files:
        im = Image.open(root + "/" + fname)
        imGray_sacle = im.convert('L') # 흑백으로 전환
        im_arr = np.array(imGray_sacle)
        norm_img = normalize(im_arr)
        inputs.append(np.reshape(norm_img[:25, :25], [-1]))

  features_test = np.reshape(inputs, [-1, len(inputs[0])])

  #read data file, and make dataframe
  df = pd.read_csv("label_test" + ".csv", encoding="ISO-8859-1")

  # create train dataset
  labels_test = np.array(np.reshape(df.values[:, 0], [-1]))

  return features, labels, features_test, labels_test

def normalize(df):
  # normalize df, (df.mean())/ standard deviation
  normal_proc = StandardScaler().fit(df)
  transformed_df = normal_proc.transform(df)
  return transformed_df

def denormalize(df):
  # denormalize the normalized data back to the original
  normal_proc = StandardScaler().fit(df)
  inverse_trans_df = normal_proc.inverse_transform(df)
  return inverse_trans_df
