# Copyright 2018 Bimghi Choi. All Rights Reserved.
#
# upgraded from v3.0
#  - base date : today's close --> next day's open
#  - RRL input : target prices --> difference between target and base prices
#  - include RRL : optional
#
# this program  is a test for many-to-many (PTB)
#     -- # of steps  ----  --- # of steps  ----
#   b ----------------------------------------------------------
#   a -                  -                    -
#   t - the first batch  -   the second batch -
#   c -                  -                    -
#   h -----------------------------------------------------------

# but in DeepMoney,
#    -- # of steps  in first batch -----
#   00-00-01 ---------------------------
#   00-00-02 -                         -
#   00-00-03 - the first batch         -
#   00-00-04 -                  -
#   00-00-05 ---------------------------
#   --- # of steps  in second batch ----
#   00-00-02 ---------------------------
#   00-00-03 -                         -
#   00-00-04 - the second batch         -
#   00-00-05 -                         -
#   00-00-06 ---------------------------
# ==============================================================================

# to eliminate a term[x, y] from training
#
# train_start1 = the first date of data
# train_start2 = y + 1
# train_end1 = x - 1
# train_end2 = test_start - extra days for elimination

"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import statistics as stat
import path as path
from statistics import stdev

def norm_data(config):
  """
  read normalized data

  kospi200 futures 5, 20, 65days-forward  predictions text files with 943 input features,
  and performs mini-batching of the inputs.

  Args:
    config : configuration class
    data_path: not use

  Returns:
    tuple (train_data, valid_data, test_data)
    where each of the data objects can be passed to producer function.
  """
  # if norm_days > 0, read raw data directory and normalize with norm days
  if (config.norm_days > 0): return raw_data(config)

  raw_df = pd.read_csv(path.file_name, encoding="ISO-8859-1")
  #raw_df = raw_df[:-config.predict_term]

  # start : the predicted date,  start_index : the date to try to predict
  test_start_index = len(raw_df[raw_df['date'] <= config.test_start]) - 1 - config.predict_term
  test_end_index = len(raw_df[raw_df['date'] <= config.test_end]) - 1 - config.predict_term

  train_start1_index = len(raw_df[raw_df['date'] <= config.train_start1]) - 1 - config.predict_term
  train_end1_index = len(raw_df[raw_df['date'] <= config.train_end1]) - 1 - config.predict_term

  train_start2_index = len(raw_df[raw_df['date'] <= config.train_start2]) - 1 - config.predict_term
  train_end2_index = len(raw_df[raw_df['date'] <= config.train_end2]) - 1 - config.predict_term

  input_target_list = list(range(1, 1 + (config.input_size + 1)))

  #create train input data 1
  train_data1 = raw_df.values[max(0, train_start1_index - config.step_interval*(config.num_steps - 1) - config.predict_term): train_end1_index  + 1 - config.predict_term, input_target_list]

  #train target data created
  train_target_raw = raw_df.values[max(0, train_start1_index - config.step_interval*(config.num_steps - 1) - config.predict_term): train_end1_index + 1, config.input_size+1]

  train_target = []
  for i in range(len(train_target_raw)-config.predict_term):
    train_target.append(train_target_raw[i+config.predict_term])

  #input + target train data created
  train_target = np.reshape(train_target, (-1, 1))
  train_data1 = np.concatenate((train_data1, train_target), axis=1)

  #create train input data 2
  train_data2 = raw_df.values[max(0, train_start2_index - config.step_interval*(config.num_steps - 1) - config.predict_term) : train_end2_index + 1 - config.predict_term, input_target_list]

  #train target data created
  train_target_raw = raw_df.values[max(0, train_start2_index - config.step_interval*(config.num_steps - 1) - config.predict_term): train_end2_index + 1, config.input_size+1]

  train_target = []
  for i in range(len(train_target_raw)-config.predict_term):
    train_target.append(train_target_raw[i+config.predict_term])

  #input + target train data created
  train_target = np.reshape(train_target, (-1, 1))
  train_data2 = np.concatenate((train_data2, train_target), axis=1)

  #test data
  if test_start_index > 0 and test_start_index < test_end_index: test_data = raw_df.values[test_start_index - config.step_interval * (config.num_steps-1) : test_end_index + 1, input_target_list]
  else: test_data = []

  #test target data created
  if test_start_index > 0 and test_start_index < test_end_index:
    test_target_raw = raw_df.values[test_start_index - config.step_interval * (config.num_steps-1): test_end_index + 1 + config.predict_term, config.input_size+1]
    test_target = []
    for i in range(len(test_target_raw)-config.predict_term):
      test_target.append(test_target_raw[i + config.predict_term])
    #input + target test data created
    test_target = np.reshape(test_target, (-1, 1))
    test_data = np.concatenate((test_data, test_target), axis=1)
  else: test_data = []

  predict_data = test_data

  return train_data1, train_data2, test_data, predict_data, test_start_index

def raw_data(config):
  """
  read non-normlized data, adnd return normalized dataset

  kospi200 futures 5, 20, 65days-forward  predictions text files with 943 input features,
  and performs mini-batching of the inputs.

  Args:
    config : configuration class
    data_path: not use

  Returns:
    tuple (train_data, valid_data, test_data)
    where each of the data objects can be passed to producer function.
  """

  raw_df = pd.read_csv(path.market + "_raw.csv", encoding="ISO-8859-1")
  #raw_df = raw_df[:-config.predict_term]

  # column 0: date, if 20days normalization, normalization start at index 19
  test_start_index = len(raw_df[raw_df['date'] <= config.test_start]) - (config.norm_days-1) - config.predict_term - 1
  test_end_index = len(raw_df[raw_df['date'] < config.test_end]) - (config.norm_days-1) - config.predict_term - 1

  normXY = make_norm(raw_df.values[:, 1:],  config.conversion, config.predict_term, config.input_size, config.norm_days)
  #normXY = normXY[:-config.predict_term]

  train_data = normXY[:test_start_index - config.predict_term, :]
  test_data = normXY[test_start_index - config.step_interval * (config.num_steps-1) : test_end_index + 1, :]
  predict_data = test_data

  return train_data, test_data, predict_data, test_start_index

def make_index_date(test_data,config):

  test_data_size = len(test_data) - config.step_interval * (config.num_steps-1)

  index_df = pd.read_csv(path.file_name, encoding = "ISO-8859-1")
  test_start_index = len(index_df[index_df['date'] <= config.test_start]) - 1 - config.predict_term

  # base date list
  z = index_df.values[test_start_index: test_start_index + test_data_size, 0]
  z = list(z)

  # index list at base date = today's close prices
  index_close = list(index_df.values[test_start_index: test_start_index + test_data_size, config.input_size + 1])

  # index list at base date = next day's open prices
  index_open = list(index_df.values[test_start_index + 1: test_start_index + test_data_size + 1, config.input_size + 2])

  # predicted date list
  #date = index_df.values[test_start_index + config.predict_term: test_start_index +  config.predict_term + test_data_size, 0]
  # eliminate weekend: if predict_term is 65, 65/5 = 13weeks, 13weeks * 7 = 91(3months)
  #date = []
  #for k in range(0, len(z)):
  #  basedate = z[k]
  #  for_date = pd.date_range(start=basedate, periods=config.predict_term / 5 * 7 + 1, freq='D')[int(config.predict_term / 5 * 7)]
  #  for_date = for_date.strftime("%Y-%m-%d")
  #  date.append(for_date)

  date = list(index_df.values[test_start_index + config.predict_term: test_start_index + test_data_size + config.predict_term, 0])

  #standard deviation
  std = []
  for k in range(0, len(date)):
    idx = len(index_df[index_df['date'] <= date[k]])-1
    std.append(stdev(index_df.values[idx - config.predict_term:idx + 1, config.input_size + 1]))

  #std = index_df.values[test_start_index + config.predict_term: test_start_index +  config.predict_term + test_data_size, 7]

  return index_close, index_open, date, z, std

def producer(raw_data, num_steps, step_interval, input_size, output_size):
  """produce time-series data.

  This chunks up raw_data into series of examples and returns Tensors that
  are drawn from these series.

  Args:
    raw_data: one of the raw data outputs from futures data.
    num_steps: int, the number of unrolls.
    step_interval : days between steps
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is base index values and the target index(n days-forward kospi200 futures values).

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """

  # split input and target, return
  dataX, dataY = [], []

  # calc the number of time series data
  size = len(raw_data) - (num_steps-1) * step_interval

  for i in range(size):
    input_list = list(range(i, i + num_steps * step_interval, step_interval))
    a = np.float32(raw_data[input_list, :input_size + 2])
    dataX.append(a)
    b = np.float32(raw_data[input_list, input_size: input_size + 2])
    dataY.append(b)

  # --> 3-D array dimension
  x = np.array(dataX).reshape(-1, num_steps, input_size + 2)
  y = np.array(dataY).reshape(-1, num_steps, 2)

  return x, y

def raw_target_producer(config):
  """produce time series data of real target index
  """
  raw_df = pd.read_csv(path.file_name, encoding="ISO-8859-1")

  # start : the predicted date,  start_index : the date to try to predict
  test_start_index = len(raw_df[raw_df['date'] <= config.test_start]) - 1 - config.predict_term
  test_end_index = len(raw_df[raw_df['date'] <= config.test_end]) - 1 - config.predict_term

  if test_start_index > 0 and test_start_index < test_end_index:
    raw_data = raw_df.values[test_start_index - config.step_interval * (config.num_steps-1): test_end_index + 1 + config.predict_term, config.input_size+1]

  # split input and target, return
  dataX = []
  dataY = []

  # calc the number of time series data
  size = len(raw_data) - (config.num_steps-1) * config.step_interval

  for i in range(size):
    input_list = list(range(i, i + config.num_steps * config.step_interval, config.step_interval))
    a = np.float32(raw_data[input_list, 0])
    dataX.append(a)

    input_list = list(range(i + config.predict_term, i + config.num_steps * config.step_interval + config.predict_term, config.step_interval))
    b = np.float32(raw_data[input_list, 0])
    dataY.append(b)

  # --> 3-D array dimension
  x = np.array(dataX).reshape(-1, config.num_steps, config.output_size)
  y = np.array(dataY).reshape(-1, config.num_steps, config.output_size)

  return x, y

def make_norm(raw_data, conversion, predict_term, input_size, days):
  norm_data = []
  for i in range(days-1, raw_data.shape[0]):
    r = []
    for j in range(input_size):
      data_norm = raw_data[i - (days-1):i + 1, j]
      std_norm = stat.stdev(data_norm)
      if std_norm != 0:
        c = np.float64((raw_data[i, j] - sum(data_norm) / days) / std_norm)
      else:
        c = 0
      if not (abs(c) < 4.3): c = 0
      r.append(c)
    if i + predict_term < raw_data.shape[0]:
      if conversion == 'diff':
        r.append(raw_data[i + predict_term, input_size] - raw_data[i, input_size])
      if conversion == 'rate':
        r.append((raw_data[i + predict_term, input_size] - raw_data[i, input_size]) / raw_data[i, input_size] * 100)
      if conversion == 'norm': r.append((raw_data[i + predict_term, input_size] - 67) / 30)
    else: r.append(0)
    norm_data.append(r)
  return np.reshape(norm_data, [-1, input_size+1])

def get_test_target(config):
  raw_df = pd.read_csv(path.file_name, encoding="ISO-8859-1")

  # start : the predicted date,  start_index : the date to try to predict
  test_start_index = len(raw_df[raw_df['date'] <= config.test_start]) - 1 - config.predict_term
  test_end_index = len(raw_df[raw_df['date'] <= config.test_end]) - 1 - config.predict_term

  input_target_list = list(range(1, 1 + (config.input_size + 1)))

  #test data
  if test_start_index > 0 and test_start_index < test_end_index: test_data = raw_df.values[test_start_index - config.step_interval * (config.num_steps-1) : test_end_index + 1, config.input_size  + 1]
  else: test_data = []

  #test target data created
  if test_start_index > 0 and test_start_index < test_end_index:
    test_target_raw = raw_df.values[test_start_index - config.step_interval * (config.num_steps-1): test_end_index + 1 + config.predict_term, config.input_size+1]
    test_target = []
    for i in range(len(test_target_raw)-config.predict_term):
      test_target.append(test_target_raw[i + config.predict_term])
    #input + target test data created
    test_target = np.reshape(test_target, (-1, 1))
    test_data = np.concatenate((test_data, test_target), axis=1)
  else: test_data = []

  return test_data