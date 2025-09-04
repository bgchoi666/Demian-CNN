#  Copyright 2018 Bumghi Choi All Rights Reserved.
#  1-D CNN

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import reader_img as reader
import pandas as pd
import shutil, os
from datetime import datetime
import time

tf.logging.set_verbosity(tf.logging.INFO)

class Config:
    width = 25
    height = 25
    output_size = 2
    batch_size = 10
    iter_steps = 5000
    kernel_size = 5
    pool_size = 2
    strides = 2
    filters = 16
    flat_size = 78*16
    directory_name = "image_files"
    model_reset = False

def data_type():
    return tf.float32

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  config = Config()
  input_layer = tf.cast(tf.reshape(features["x"], [-1, config.width * config.height, 1]), tf.float32)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=config.filters,
      kernel_size=config.kernel_size,
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=config.pool_size, strides=config.strides)

  # Convolutional Layer #2
  # Computes 16 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=config.filters,
      kernel_size=config.kernel_size,
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=config.pool_size, strides=config.strides)

  # Convolutional Layer #3
  # Computes 8 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  conv3 = tf.layers.conv1d(
      inputs=pool2,
      filters=config.filters,
      kernel_size=config.kernel_size,
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=config.pool_size, strides=config.strides)

  # Convolutional Layer #4
  # Computes 8 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  #conv4 = tf.layers.conv2d(
  #    inputs=pool3,
  #    filters=32,
  #    kernel_size=config.kernel_size,
  #    padding="same",
  #    activation=tf.nn.relu)

  # Pooling Layer #5
  # Second max pooling layer with a 2x2 filter and stride of 2
  #pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=config.pool_size, strides=2)

  # Convolutional Layer #5
  # Computes 8 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  #conv5 = tf.layers.conv2d(
  #    inputs=pool2,
  #    filters=32,
  #    kernel_size=config.kernel_size,
  #    padding="same",
  #    activation=tf.nn.relu)

  # Pooling Layer #5
  # Second max pooling layer with a 2x2 filter and stride of 2
  #pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=config.pool-size, strides=2)

  # Flatten tensor into a batch of vectors
  pool3_flat = tf.reshape(pool3, [-1, config.flat_size])

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=pool3_flat, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer
  # Densely connected layer with 1024 neurons
  dense = tf.layers.dense(inputs=dropout, units=1024, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001)
  )

  # Add dropout operation; 0.8 probability that element will be kept
  dropout2 = tf.layers.dropout(
      inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer
  # Densely connected layer with 1024 neurons
  dense2 = tf.layers.dense(inputs=dropout2, units=100, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001)
  )

  # Add dropout operation; 0.8 probability that element will be kept
  dropout3 = tf.layers.dropout(
      inputs=dense2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # dense1 Tensor Shape: [batch_size, 1024]
  # dense2 Tensor Shape: [batch_size, 100]
  # output Tensor Shape: [batch_size, 2]
  logits = tf.layers.dense(inputs=dropout3, units=2, activation=None)

  classes = tf.argmax(input=logits, axis=1, name="classes")
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": classes, #tf.argmax(input=logits, axis=1, name="classes"),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "logits" : logits
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=config.output_size)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
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
          labels, classes)}

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data

  config = Config()
  (train_inputs, train_labels, test_inputs, test_labels) = reader.read_image_file(config)

  # Create the Estimator
  model_dir = "model_dir/" + "CNN_" + str(config.kernel_size) + "_" + str(config.filters) + "_" + str(config.batch_size)
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

  # Train the model using train_data1
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

  save_results(predictions, config, test_labels, eval_results)

def save_results(predictions, config, target, ev):
    target_values = []
    class_values = []
    numbers = []
    for i, p in enumerate(predictions):
        target_values.append(target[i])
        class_values.append(p["classes"])
        numbers.append(str(i))
    index_real = []
    index_pred = []
    profits = []

    accuracy, _, _, _ = calculate_recall_precision(target_values, class_values)
    comp_results = {"num" : numbers, "real" : target_values, "prediction" : class_values}
    result_file = "results/CNN_" + str(config.kernel_size) + "_" + str(config.filters) + "_" + str(config.batch_size) + "_"  + \
                   datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H-%M-%S') + ".csv"
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
