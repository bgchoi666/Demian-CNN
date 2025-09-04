#  Copyright 2018 Shinhan Financial Group All Rights Reserved.
# ==============================================================================
"""Utilities for read and parsing kospi200f, s&p500 text files.
2D cnn을 마치 1D cnn처럼 사용하여 softmaxt로 상승/하락 확률을 산출하는 예제
cnn input의 width는 변수들이고 height는 time sequence 의 step들이다.
kernel의  width를 1로 한다면 변수들의 특성을 유지하면서 time sequence 를 적절히
압축하는 효과를 얻는다.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import reader
import pandas as pd
import shutil, os
from datetime import datetime
import time

tf.logging.set_verbosity(tf.logging.INFO)

class Config:
    step_interval = 20
    num_steps = 20
    input_size = 38
    output_size = 1
    far_predict = 65
    batch_size = 50
    iter_steps = 2000
    filters = 32
    flat_size = 4 * 2 * 32
    kernel_width = 5
    kernel_height = 5
    kernel_stride_width = 1
    kernel_stride_height = 1
    pool_width = 2
    pool_height = 2
    pool_stride_width = 2
    pool_stride_height = 2
    file_name = "nikey225"
    test_start = '2016-07-01'
    test_end ='2017-08-31'
    model_reset = True

def data_type():
    return tf.float32

def cnn_model_fn(features, labels, mode, params):
  """Model function for CNN."""

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # images are 65x34 input-output pairs, and have one color channel
  config = Config()
  input_layer = tf.cast(tf.reshape(features["x"], [-1, config.num_steps, config.input_size, 1]), tf.float32)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=config.filters,
      kernel_size=[config.kernel_height, config.kernel_width],
      strides = [config.kernel_stride_height, config.kernel_stride_width],
      padding="valid",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[config.pool_height, 1], strides=[config.pool_stride_height, config.pool_stride_width])

  # Convolutional Layer #2
  # Computes 16 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=config.filters,
      kernel_size=[config.kernel_height, 1],
      strides = [config.kernel_stride_height, config.kernel_stride_width],
      padding="valid",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[config.pool_height, 1], strides=[config.pool_stride_height, config.pool_stride_width])

  # Convolutional Layer #3
  # Computes 8 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  #conv3 = tf.layers.conv2d(
  #    inputs=pool2,
  #    filters=config.filters,
  #    kernel_size=[config.kernel_height, 1],
  #    strides = [config.kernel_stride_height, config.kernel_stride_width],
  #    padding="valid",
  #    activation=tf.nn.relu)

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  #pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[config.pool_height, 1], strides=[config.pool_stride_height, config.pool_stride_width])

  # Convolutional Layer #4
  # Computes 8 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  #conv4 = tf.layers.conv2d(
  #    inputs=pool3,
  #    filters=config.filters,
  #    kernel_size=[config.kernel_height, config.kernel_width],
  #    strides=[config.kernel_stride_height, config.kernel_stride_width],
  #    padding="valid",
  #    activation=tf.nn.relu)

  # Pooling Layer #4
  # Second max pooling layer with a 2x2 filter and stride of 2
  #pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[config.pool_height, config.pool_width], strides=[config.pool_stride_height, config.pool_stride_width])


  # Convolutional Layer #5
  # Computes 8 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  #conv5 = tf.layers.conv2d(
  #    inputs=pool4,
  #    filters=config.filters,
  #    kernel_size=[config.kernel_height, config.kernel_width],
  #    strides=[config.kernel_stride_height, config.kernel_stride_width],
  #    padding="valid",
  #    activation=tf.nn.relu)

  # Pooling Layer #5
  # Second max pooling layer with a 2x2 filter and stride of 2
  #pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[config.pool_height, config.pool_width], strides=[config.pool_stride_height, config.pool_stride_width])

  # Flatten tensor into a batch of vectors
  if mode == tf.estimator.ModeKeys.TRAIN: batch_size = config.batch_size
  else: batch_size = 1
  pool_flat = tf.reshape(pool2, [-1, 2*9*32])

  # Add dropout operation; 0.6 probability that element will be kept
  dropout_cnn = tf.layers.dropout(
      inputs=pool_flat, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer 1
  #dense = tf.layers.dense(inputs=dropout_cnn, units=2000, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),
  #                        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001)
  #)
  # Add dropout operation; 0.6 probability that element will be kept
  #dropout = tf.layers.dropout(
  #    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # dense layer 2
  #dense2 = tf.layers.dense(inputs=dropout_cnn, units=1000, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),
  #                        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001)
  #)
  # Add dropout operation; 0.6 probability that element will be kept
  #dropout2 = tf.layers.dropout(
  #    inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  #dense layer 3
  #dense3 = tf.layers.dense(inputs=dropout2, units=500, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),
  #                        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001)
  #)
  # Add dropout operation; 0.8 probability that element will be kept
  #dropout3 = tf.layers.dropout(
  #    inputs=dense3, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  #dense layer 4
  dense4 = tf.layers.dense(inputs=dropout_cnn, units=50, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001)
  )
  # Add dropout operation; 0.8 probability that element will be kept
  dropout4 = tf.layers.dropout(
      inputs=dense4, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  outputs = tf.layers.dense(inputs=dropout4, units=2)
  # for regression
  #logits = tf.layers.dense(inputs=outputs, units=1, activation=None)
  #logits = tf.cast(tf.reshape(logits, [-1]), tf.float32)

  classes = tf.argmax(input=outputs, axis=1, name="classes")
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": classes,
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(outputs, name="softmax_tensor"),
      "logits" : outputs
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes) for regression
  # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  #labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)
  #loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

  # Calculate Loss (for both TRAIN and EVAL modes)
  #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=config.output_size)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.cast(labels, tf.int32), logits=outputs, name='cross_entropy_per_example')
  loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  #eval_metric_ops = {
  #    "accuracy": tf.metrics.accuracy(
  #        labels=labels, predictions=predictions["logits"])}
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(labels, tf.int64), classes)}

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data

  config = Config()
  (train_data, test_data, predict_data, today, index, date) = reader.read_file(config)
  (train_inputs, train_labels) = reader.producer(train_data, config.input_size, config.output_size, config.far_predict, config.step_interval, config.num_steps)
  (test_inputs, test_labels) = reader.producer(test_data, config.input_size, config.output_size, config.far_predict, config.step_interval, config.num_steps)

  # Create the Estimator
  model_dir = "model_dir/" + "CNN_kospi200f_2d_prob" + str(config.far_predict) + "_" + str(config.step_interval) + "_" + \
              str(config.num_steps) + "_" + str(config.filters) + "_kernel_" + str(config.kernel_height) + "_" + str(config.kernel_width) + \
              "_" + str(config.kernel_stride_height) + "_" + str(config.kernel_stride_width) + str(config.batch_size) + "_" + str(config.test_start)
  if config.model_reset == True:
      try:
          shutil.rmtree(model_dir)
      except OSError as e:
          if e.errno == 2:
              # 파일이나 디렉토리가 없음!
              pass
          else:
              raise
  cnn = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=model_dir)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"classes": "classes"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_inputs},
      y=train_labels,
      batch_size=config.batch_size,
      num_epochs=None,
      shuffle=True)
  cnn.train(
      input_fn=train_input_fn,
      hooks=[logging_hook],
      steps=config.iter_steps
      )

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_inputs},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = cnn.evaluate(input_fn=eval_input_fn)
  print(eval_results)


  # Predict the model and save results
  predictions = cnn.predict(input_fn=eval_input_fn)

  # save  the results
  #target = list(test_labels)
  #pred_values = []
  #class_values = []
  #for i, p in enumerate(predictions):
  #      pred_values.append(p["logits"])
  #      class_values.append(p["classes"])
  #comp_results = {"date": test_date, "target": target, "predict": pred_values, "classes": class_values}
  #pd.DataFrame.from_dict(comp_results).to_csv("target-predict-file.csv", index=False)

  save_results(predictions, config, test_labels, index, today, date, eval_results)

def save_results(predictions, config, target, index_today, today, date, ev):
    pred_values = []
    target_values = []
    class_values = []
    for i, p in enumerate(predictions):
        target_values.append(target[i])
        k = p["probabilities"]
        pred_values.append(k[1])
        class_values.append(p["classes"])

    accuracy, _, _, _ = calculate_recall_precision(target_values, class_values)
    comp_results = {"date" : today,  "pred_date" : date,
                    "index_today" : index_today, "pred_class": class_values,
                    "real" : target_values, "prediction" : pred_values}
    result_file = "results/CNN_prob_" + config.file_name + "_" + str(config.far_predict) + "_" + str(config.step_interval) + "_" + str(config.num_steps) + \
                  "_k" + str(config.kernel_height) + "." + str(config.kernel_width) + "_p" + str(config.pool_height) + "." + str(config.pool_width) + \
                   "_" + str(config.batch_size) + "_"  + str(config.test_start) + "_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H-%M-%S') + ".csv"
    pd.DataFrame.from_dict(comp_results).to_csv(result_file, index=False)
    r = open(result_file, 'a')
    r.write("accuracy, RMSE\n" + str(accuracy) + ", " + str(ev["rmse"]))

def calculate_recall_precision(label, prediction):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(0, len(label)):
        if prediction[i] == 1:
            if label[i] == 1:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if label[i] == 0:
                true_negatives += 1
            else:
                false_negatives += 1

    # a ratio of correctly predicted observation to the total observations
    accuracy = (true_positives + true_negatives) \
               / (true_positives + true_negatives + false_positives + false_negatives)

    # precision is "how useful the search results are"
    precision = true_positives / (true_positives + false_positives)
    # recall is "how complete the results are"
    recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 / ((1 / precision) + (1 / recall))

    return accuracy, precision, recall, f1_score

if __name__ == "__main__":
  tf.app.run()
