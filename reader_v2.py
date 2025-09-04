# Copyright 2018 Bimghi Choi. All Rights Reserved.
# ==============================================================================
"""Utilities for read and parsing kospi200f, s&p500 text files.
2D cnn을 마치 1D cnn처럼 사용하여 regression을 하는 예제

version1과의 차이 :  version1은 상승/하락 확률, version2는 regression 그리고
  test set이 전체 data set 중간에 있음
"""
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
        inputs.append(np.reshape(im_arr, [-1])/255)

  features = np.reshape(inputs, [-1, len(inputs[0])])

  # read label file, make labels
  labels = []

  return features, labels

def read_file(config, data_path=None):
  """Loadkosp kospi200f or s&p 500 raw data from data directory "data_path".

  kospi200 futures 5, 20, 65days-forward  predictions text files with xx input features,
  and performs mini-batching of the inputs. consider the time-series data  as an image

  Args:
    data_path: not use

  Returns:
    tuple (train_data, valid_data, test_data)
    where each of the data objects can be passed to producer function.
  """

  #read data file, and make dataframe
  raw_df = pd.read_csv("raw_data_" + config.file_name + ".csv", encoding="ISO-8859-1")
  df = normalize(raw_df.values[:, 1:config.input_size+1])

  #calculate the index for test_start_date
  test_start_index = len(raw_df[raw_df['date'] <= config.test_start]) - 1 - config.far_predict
  test_end_index = len(raw_df[raw_df['date'] < config.test_end]) - 1 - config.far_predict

  input_target_list = list(range(1, config.input_size))

  #the first train input 데이터 생성
  train_data1 =  df[0: test_start_index - config.far_predict, :config.input_size]
  # the second train target data created
  train_target_raw = raw_df.values[0: test_start_index, config.input_size + 1]
  train_target1 = []
  for i in range(len(train_target_raw) - config.far_predict):
    if config.conversion == 'diff': train_target1.append(train_target_raw[i+config.far_predict]-train_target_raw[i])
    if config.conversion == 'rate': train_target1.append((train_target_raw[i+config.far_predict]-train_target_raw[i])/train_target_raw[i]*100)
    if config.conversion == 'norm': train_target1.append((train_target_raw[i+config.far_predict] - 67)/30)
  # input + target train data created
  train_target1 = np.reshape(train_target1, (-1, 1))
  train_data1 = np.concatenate((train_data1, train_target1), axis=1)

  #the second train input 데이터 생성
  train_data2 =  df[test_end_index + 1 - config.step_interval * (config.num_steps-1) :
                    - config.step_interval * (config.num_steps-1) - config.far_predict, :config.input_size]
  # the second train target data created
  train_target_raw = raw_df.values[test_end_index + 1:  - config.step_interval * (config.num_steps - 1), config.input_size + 1]
  train_target2 = []
  for i in range(len(train_target_raw) - config.far_predict):
    if config.conversion == 'diff': train_target2.append(train_target_raw[i+config.far_predict]-train_target_raw[i])
    if config.conversion == 'rate': train_target2.append((train_target_raw[i+config.far_predict]-train_target_raw[i])/train_target_raw[i]*100)
    if config.conversion == 'norm': train_target2.append((train_target_raw[i+config.far_predict] - 67)/30)
  # input + target train data created
  train_target2 = np.reshape(train_target2, (-1, 1))
  train_data2 = np.concatenate((train_data2, train_target2), axis=1)

  # test data 생성
  test_data = df[test_start_index - config.step_interval * (config.num_steps - 1): test_end_index + 1, :config.input_size]

  # test target data created
  test_target_raw = raw_df.values[test_start_index - config.step_interval * (config.num_steps - 1): test_end_index + 1 + config.far_predict, config.input_size + 1]
  test_target = []
  for i in range(len(test_target_raw) - config.far_predict):
    if config.conversion == 'diff': test_target.append(test_target_raw[i+config.far_predict]-test_target_raw[i])
    if config.conversion == 'rate': test_target.append((test_target_raw[i+config.far_predict]-test_target_raw[i])/test_target_raw[i]*100)
    if config.conversion == 'norm': test_target.append((test_target_raw[i + config.far_predict] - 67) / 30)
  # input + target test data created
  test_target = np.reshape(test_target, (-1, 1))
  test_data = np.concatenate((test_data, test_target), axis=1)
  predict_data = test_data

  #test 시도일 및 지수, test 목표일
  today = raw_df.values[test_start_index: test_end_index + 1, 0] # date 0번 column
  today_index = raw_df.values[test_start_index:test_end_index  + 1, config.input_size + 1] # 종가 지수 4번 column
  target_date = raw_df.values[test_start_index + config.far_predict:test_end_index + config.far_predict + 1, 0] # row(target_date) = row(today_index) + far_predict

  return train_data1, train_data2, test_data, predict_data, today, today_index, target_date


def producer(raw_data, input_size, output_size, far_predict, interval, steps, name=None):
  """produce time-series data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data with 5, 20, 65-after futures data.
    far_predict : the predict term. (5, 20, 65)
    interval : the interval between steps
    steps : the number of serial steps for input data
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, input_size * steps]. The second element
    of the tuple is the target data(5, 20, 65days-forward  values).

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """

  #input = np.reshape(raw_data[:, 1:input_size + 1], [-1, input_size]) #date column 제거
  #target = np.reshape(raw_data[:, input_size + 1], [-1])

  # input data만 normalize
  #norm_df = normalize(input)

  # input과 target 분리하여 반환
  dataX, dataY = [], []

  # training data size 계산
  size = len(raw_data)  - interval * (steps - 1)

  for i in range(size):
    input_list = list(range(i, i + steps * interval, interval)) # i, i + j, i + 2j, . . . , i + (k-1)j
    a = np.reshape(raw_data[input_list, :input_size], [steps*input_size])
    dataX.append(a)

    b = np.reshape(raw_data[i + interval * (steps - 1), input_size], [1]) # the target value at the last date of serial input
    dataY.append(b)

  x = np.array(dataX).reshape(-1, steps*input_size)
  y = np.array(dataY).reshape(-1)


  return x, y

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
