# Copyright 2018 Bimghi Choi. All Rights Reserved.
# ==============================================================================
"""Utilities for read and parsing kospi200f s&p500 text files.
1D cnn을 사용하여 일정 기간(1일 ~ 1년)후의 지수 상승/하락 확률을  output으로 하는 시스템
cnn input의 각 변수들에 대한 time sequence data를 한 줄로 연결한다.
csv로 되어 있는 해당 지수에 대한 기술적 지표 데이터를 읽어 cnn의 1 dimensional input data로 변환한다.
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
  """Load PTB raw data from data directory "data_path".

  kospi200 futures 65days-forward  predictions text files with 35 input features,
  and performs mini-batching of the inputs.

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
  test_end_index = len(raw_df[raw_df['date'] <= config.test_end]) - 1 - config.far_predict

  input_target_list = list(range(1, config.input_size))

  #train input 데이터 생성
  train_data =  df[0: test_start_index - config.far_predict, :config.input_size]
  # train target data created
  train_target_raw = raw_df.values[0: test_start_index, config.input_size + 1]
  train_target = []
  for i in range(len(train_target_raw) - config.far_predict):
    if train_target_raw[i + config.far_predict] - train_target_raw[i] > 0: train_target.append(1)
    else: train_target.append(0)
  # input + target train data created
  train_target = np.reshape(train_target, (-1, 1))
  train_data = np.concatenate((train_data, train_target), axis=1)

  # test data 생성
  test_data = df[test_start_index - config.step_interval * (config.num_steps - 1): test_end_index  + 1, :config.input_size]

  # test target data created
  test_target_raw = raw_df.values[test_start_index - config.step_interval * (config.num_steps - 1): test_end_index + 1 + config.far_predict, config.input_size + 1]
  test_target = []
  for i in range(len(test_target_raw) - config.far_predict):
    if test_target_raw[i + config.far_predict] - test_target_raw[i] > 0: test_target.append(1)
    else: test_target.append(0)
  # input + target test data created
  test_target = np.reshape(test_target, (-1, 1))
  test_data = np.concatenate((test_data, test_target), axis=1)

  predict_data = test_data

  #test 시도일 및 지수, test 목표일
  today = raw_df.values[test_start_index:test_end_index + 1, 0] # date 0번 column
  today_index = raw_df.values[test_start_index:test_end_index + 1, config.input_size+1] # 종가 지수
  target_date = raw_df.values[test_start_index + config.far_predict:test_end_index + config.far_predict + 1, 0] # row(target_date) = row(today_index) + far_predict

  return train_data, test_data, predict_data, today, today_index, target_date


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
  size = len(raw_data) - interval * (steps - 1)

  for i in range(size):
    input_list = list(range(i, i + steps * interval, interval)) # i, i + j, i + 2j, . . . , i + (k-1)j
    a = np.reshape(raw_data[input_list, :input_size].T, [steps*input_size])
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
