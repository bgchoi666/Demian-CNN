# Copyright 2018 Bumghi Choi. All Rights Reserved.
#

"""Index predicting Many-to-Many LSTM model.

# upgraded from v3.0
#  - base date : today's close --> next day's open
#  - RRL input : target prices --> difference between target and base prices
#  - include RRL : optional

 The hyperparameters used in the model:
 - init_scale - the initial scale of the weights
 - learning_rate - the initial value of the learning rate
 - num_layers - the number of LSTM layers
 - num_steps - the number of unrolled steps of LSTM
 - hidden_size - the number of LSTM units
 - batch_size - the batch size
 - input_size - the number of input nodes
 - file_name
 - conversion - the conversion type of target data: rate, difference, 20days norm
 - grad_train_term - [train_start_date1, train_end_date1(=test_start_date1=train_start_date2), train__end_date2(=test_end_date1=test_start_date2=train_start_date3), . . . ]

 The data required for this example is in the data/ dir of the
    - kosfi200f-943.csv : for prediction after 1 week with 943 input features

 To run:

 $ python deepnmoney_v2.1.1.py

 target : diff - values of today + predict_term - today
          rate - (value of today + predict_term - today)/values of today * 100
          norm - (values of today _ predict_term - 67)/30
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# imperative running
#tf.enable_eager_execution()

import pandas as pd

import reader
import path
from datetime import datetime
import time
import shutil
import os
import random as rd
import logging
from datetime import datetime

# from tensorflow.models.rnn.ptb import reader

#flags = tf.flags
logging = tf.logging

#flags.DEFINE_string("data_path", "DeepMoney_v2.0",
#                    "Where the training/test data is stored.")
#flags.DEFINE_string("rnn_mode", "BASIC",
#                    "The low level implementation of lstm cell: one of CUDNN, "
#                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
#                    "and lstm_block_cell classes.")
#FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
GRU = "gru"

#log file setup


tf.logging.set_verbosity(tf.logging.INFO)

def data_type():
    return tf.float32

def model_setup(config):
    model_params = {"learning_rate": config.learning_rate,
                    "rnn_mode": config.rnn_mode,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "batch_size": config.batch_size,
                    "num_steps": config.num_steps,
                    "output_size": config.output_size,
                    "input_size": config.input_size,
                    "init_scale": config.init_scale,
                    "predict_term": config.predict_term,
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "shuffle": config.shuffle}
    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20 * 60,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max=1,  # Retain the 10 most recent checkpoints.
    )
    return model_params, my_checkpointing_config

def model_fn(features, labels, mode, params):
    #customized many to many LSTM model function

    rnn_mode = params["config"].rnn_mode
    hidden_size1 = params["config"].hidden_size1
    hidden_size2 = params["config"].hidden_size2
    num_layers = params["config"].num_layers
    batch_size = params["config"].batch_size
    num_steps = params["config"].num_steps
    output_size = params["config"].output_size
    learning_rate = params["config"].learning_rate
    alpha = params["config"].alpha
    beta = params["config"].beta
    shuffle = params["config"].shuffle
    config = params["config"]

    if not (mode == tf.estimator.ModeKeys.TRAIN):
        batch_size = 1;

    # make one cell(LSTM layer) for ibdex prediction
    def make_cell():
        if rnn_mode == BASIC:
            #return tf.contrib.rnn.LSTMCell(
            #    hidden_size, use_peepholes=True, initializer=tf.contrib.layers.xavier_initializer(),
            #    forget_bias=0.0, state_is_tuple=True,
            #    reuse=False, activation=tf.nn.tanh)
            return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size1)
        if rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                hidden_size1, forget_bias=0.0, activation=tf.nn.tanh)
        if rnn_mode == GRU:
            #return tf.contrib.rnn.GRUCell(
            #    hidden_size, activation=tf.nn.tanh)
            return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(hidden_size1)
        raise ValueError("rnn_mode %s not supported" % params["rnn_mode"])
        return cell

    # make one cell(LSTM layer) for asset rebalancing
    def make_cell2():
        if rnn_mode == BASIC:
            #return tf.contrib.rnn.LSTMCell(
            #    hidden_size, use_peepholes=True, initializer=tf.contrib.layers.xavier_initializer(),
            #    forget_bias=0.0, state_is_tuple=True,
            #    reuse=False, activation=tf.nn.tanh)
            return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size2)
        if rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(hidden_size2, forget_bias=0.0, activation=tf.nn.tanh)
        if rnn_mode == GRU:
            #return tf.contrib.rnn.GRUCell(
            #    hidden_size, activation=tf.nn.tanh)
            return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(hidden_size2)
        raise ValueError("rnn_mode %s not supported" % params["rnn_mode"])
        return cell

    # create LSTM layers as many as num_layer
    #endoder cell
    cell_1 = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(num_layers)], state_is_tuple=True)
    #decoder cell
    cell_2 = tf.contrib.rnn.MultiRNNCell(
        [make_cell2() for _ in range(num_layers)], state_is_tuple=True)

    # state initialization
    state = cell_1.zero_state(batch_size, data_type())

    input_targets = features["x"]
    inputs = input_targets[:, :, :config.input_size]

    outputs = []
    # endocer: recurrently created steps, input shape is inputs[batch_size, num_steps, input_size]
    # the last state --> initial state of decoder
    with tf.variable_scope("RNN1"):
         for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
                #for i in range(batch_size):
                #    for j in range(input_size):
                #        inputs[i, time_step, j].assign(out[i, j])
            (cell_output, state) = cell_1(inputs[:, time_step, :], state)
            #else:
            #    (cell_output, state) = cell(inputs[:, 0, :], state)
            #out = tf.layers.dense(cell_output, units=input_size, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="dense2")
            #out1 = tf.layers.dense(cell_output, units=5, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="dense1")
            #outputs = tf.layers.dense(cell_output, units=output_size, activation=None, reuse=tf.AUTO_REUSE, name="dense2")
            outputs.append(cell_output)
         #output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)


    # many-to-one option... need only the last step
    #logit = tf.reshape(logits[num_steps-1], [-1])

    # after appending, shape be (num_steps, batch_size, hidden_size)
    # reshaped to ( batch_size, num_steps, hidden_size), again to (batch_size*num_steps, hidden_size)....
    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size1])


    #1 dense hidden layer and 1 output layer
    #net = tf.layers.dense(output, units=100, activation=tf.nn.relu)
    #net2 = tf.layers.dense(net, units=10, activation=tf.nn.relu)
    logits = tf.layers.dense(output, units=output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

    logit = tf.reshape(logits, [-1])

    targets = input_targets[:, :, config.input_size:config.input_size + 2]

    # produce real, predicted index values
    base_prices = tf.reshape(targets[:, :, 0], [batch_size, num_steps, 1])
    predict_prices = tf.reshape(tf.math.add(logits, tf.reshape(base_prices, [-1, 1])), [batch_size, num_steps, 1])
    target_prices = tf.reshape(targets[:, :, 1], [batch_size, num_steps, 1])
    diff = tf.subtract(target_prices, base_prices)

    # state initialization
    state = cell_2.zero_state(batch_size, data_type())

    # asset rebalancing -1 sell, 1 buy
    # input: target prices
    outputs2 = []
    with tf.variable_scope("RNN2"):
         for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
                #for i in range(batch_size):
                #    for j in range(input_size):
                #        inputs[i, time_step, j].assign(out[i, j])
            (cell_output, state) = cell_2(diff[:, time_step, :], state)
            #else:
            #    (cell_output, state) = cell(inputs[:, 0, :], state)
            #out = tf.layers.dense(cell_output, units=input_size, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="dense2")
            #out1 = tf.layers.dense(cell_output, units=5, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="dense1")
            #outputs = tf.layers.dense(cell_output, units=output_size, activation=None, reuse=tf.AUTO_REUSE, name="dense2")
            outputs2.append(cell_output)
         #output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)


    # after appending, shape be (num_steps, batch_size, hidden_size)
    # reshaped to ( batch_size, num_steps, hidden_size), again to (batch_size*num_steps, hidden_size)....
    output2 = tf.reshape(tf.concat(outputs2, 1), [-1, hidden_size2])


    #1 dense hidden layer and 1 output layer
    #net = tf.layers.dense(output, units=100, activation=tf.nn.relu)
    #net2 = tf.layers.dense(net, units=10, activation=tf.nn.relu)
    logits2 = tf.layers.dense(output2, units=output_size, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

    logit2 = tf.reshape(logits2, [-1])

    """
    tv = tf.trainable_variables()
    first_layer = 0
    second_layer = 0
    dense_layer = 0

    for i in range(1443):
        first_layer += tv[0][i, 10]
    for i in range(1000):
        second_layer += tv[2][i, 10]
    for i in range(500):
        dense_layer += tv[4][i, 0]

    tf.summary.scalar("first_layer", first_layer)
    tf.summary.scalar("second_layer", second_layer)
    tf.summary.scalar("dense_layer", dense_layer)
    """
    #for i in range(400): tf.summary.scalar("second_layer_weight", tv[5][200,i])

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "results": logit,
        "buy_sell": logit2
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # when only the last step of target data used ...
    #targets = tf.slice(targets, [0, num_steps - 1, 0], [batch_size, 1, 1])

    target = tf.reshape(labels, [-1])

    # optimizer = tf.train.GradientDescentOptimizer(self._lr)
    # self._train_op = optimizer.minimize(loss=loss, global_step=tf.contrib.framework.get_or_create_global_step())

    # loss is the average of squared sum of difference from target
    loss = tf.losses.mean_squared_error(target, logit)

    #the cost for lstm regularization
    #lstm_trainable_weights = cell.trainable_weights
    #regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in lstm_trainable_weights])
    #loss = loss + regularization_cost


    # add RRL cost - maximize downside sharp ratio
    r = [] # price difference
    for i in range(batch_size):
        r.append(0) # the first step has no difference
        for j in range(1, num_steps):
            r.append(target_prices[i][j][0] * 30 - target_prices[i][j-1][0] * 30)
    r = tf.reshape(r, [batch_size, num_steps, 1])

    # reshape the asset rebalancing outputs of RRL network
    F = tf.reshape(logit2, [batch_size, num_steps, 1])

    # calc returns from each step in batches
    R = []
    for i in range(batch_size):
        R.append(0)
        for j in range(1, num_steps):
            R.append(r[i, j, 0] * F[i, j - 1, 0] - 0.00003 * abs(F[i, j, 0] - F[i, j - 1, 0]))
    R = tf.reshape(R, [batch_size, num_steps, 1])

    # calc downside sharp ratio

    # downside returns
    DR = []
    for i in range(batch_size):
        for j in range(num_steps):
            DR.append(tf.minimum(0.0, R[i, j, 0]))
    DR = tf.reshape(DR, [batch_size, num_steps, 1])

    # calc. downside sharp ratio
    s = []
    for i in range(batch_size):
       s.append(tf.reduce_mean(R[i, :, 0])/tf.keras.backend.std(DR[i, :, 0]))
    loss2 = tf.reduce_mean(s)

    # minimize the loss of prediction network and maximize downside sharp ratio of RRL
    if config.RRL: loss = loss - loss2

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(target, logit),
        "accuracy": tf.metrics.accuracy(tf.sign(target), tf.sign(logit))
    }

    if not (mode == tf.estimator.ModeKeys.TRAIN):
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

def create_estimator(config):
    #configuration parameter --> model parameter
    model_params = {"config": config}

    # only the last checkpoint saved
    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20 * 60,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max=1,  # Retain the 10 most recent checkpoints.
    )
    # model directory for new creation mode  - random number added to the end of directory
    model_dir = path.model_dir +  "/" + path.market + "_" + str(config.predict_term) + "_" + \
                str(config.step_interval) + "_" + str(config.num_steps) + "_" + str(config.rnn_mode) + "_" + \
                str(config.hidden_size1) + "_" + str(config.batch_size) + "_" + str(config.num_layers) + "layers" + datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    # existing directory reused
    model_dir_cont = path.model_dir +  "/" + path.market + "_" + str(config.predict_term) + "_" + str(config.step_interval) + "_" + str(config.num_steps) + "_" + \
                     str(config.rnn_mode) + "_" + str(config.hidden_size1) + "_" + str(config.batch_size) + "_" + str(config.num_layers) + "layers"
    if config.model_reset == True: # if True, new model creation mode
        try:
            shutil.rmtree(model_dir)
        except OSError as e:
            if e.errno == 2:
                # not existing file or directory
                pass
            else:
                raise
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params, config=my_checkpointing_config)
    if os.path.isdir(model_dir_cont) : # reuse existing model
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir_cont, params=model_params, config=my_checkpointing_config)
    else :
        os.makedirs(model_dir_cont)
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir_cont, params=model_params, config=my_checkpointing_config)

def train(model, train_data, config):
    #model training

    #train input, target data
    inputs, targets = reader.producer(train_data, config.num_steps, config.step_interval, config.input_size, config.output_size)

    # target values created asscording to conversion type
    # create target values according to given conversion type
    target = []
    for i in range(len(targets)):
        for step in range(config.num_steps):
            if path.Config().conversion == 'diff': target.append(targets[i, step, 1] - targets[i, step, 0])
            if path.Config().conversion == 'rate': target.append(
                (y[i, step, 1] - y[i, step, 0]) / targets[i, step, 0] * 100)
            if path.Config().conversion == 'norm': target.append((targets[i, step, 0] - 67) / 30)
    target_values = np.array(target).reshape(-1, config.num_steps, 1)

    #train input function setup
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(inputs)},
        y=np.array(target_values),
        batch_size=config.batch_size,
        num_epochs=None,
        shuffle=config.shuffle)

    model.train(input_fn=train_input_fn, steps=config.iter_steps)

def test(model, test_data, config):
    # Score accuracy

    # testinput, target data
    inputs, targets = reader.producer(test_data, config.num_steps, config.step_interval, config.input_size, config.output_size)

    # target values created asscording to conversion type
    # create target values according to given conversion type
    target = []
    for i in range(len(targets)):
        for step in range(config.num_steps):
            if path.Config().conversion == 'diff': target.append(targets[i, step, 1] - targets[i, step, 0])
            if path.Config().conversion == 'rate': target.append(
                (y[i, step, 1] - y[i, step, 0]) / targets[i, step, 0] * 100)
            if path.Config().conversion == 'norm': target.append((targets[i, step, 0] - 67) / 30)
    target_values = np.array(target).reshape(-1, config.num_steps, 1)

    # test input function setup
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(inputs)},
        y=np.array(target_values),
        batch_size=1,
        num_epochs=1,
        shuffle=False)

    ev = model.evaluate(input_fn=test_input_fn)
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error : %s" % ev["rmse"])
    print("accuracy : %s" % ev["accuracy"])
    return ev["loss"]

def predict(model, predict_data, config, test_start_index):
    # Print out predictions

    # test input, target data
    inputs, targets = reader.producer(predict_data, config.num_steps, config.step_interval, config.input_size, config.output_size)

    # target values created asscording to conversion type
    # create target values according to given conversion type
    target = []
    for i in range(len(targets)):
        for step in range(config.num_steps):
            if path.Config().conversion == 'diff': target.append(targets[i, step, 1] - targets[i, step, 0])
            if path.Config().conversion == 'rate': target.append(
                (targets[i, step, 1] - targets[i, step, 0]) / targets[i, step, 0] * 100)
            if path.Config().conversion == 'norm': target.append((targets[i, step, 0] - 67) / 30)

    # test input function setup
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": inputs},
        batch_size=1,
        num_epochs=1,
        shuffle=False)

    predictions = model.predict(input_fn=predict_input_fn)

    index_close, index_open, date, today, std = reader.make_index_date(predict_data, config)

    pred_values = []
    target_values = []
    asset_rebal = []
    for i, p in enumerate(predictions):
        if i % config.num_steps == config.num_steps - 1:
            pred_values.append(p["results"])
            target_values.append(target[i])
        if i % config.num_steps == config.num_steps - int(config.predict_term / config.step_interval) - 1:
            asset_rebal.append(p["buy_sell"])

    index_real = []
    index_pred = []
    profits = []

    # restore to original index values according to market, conversion type
    for i in range(len(index_close)):
        if target[i] != 'Nan':
            if config.conversion == 'rate':
                index_real.append(target_values[i]*index_close[i]/100 + index_close[i])
                index_pred.append(pred_values[i]*index_close[i]/100 + index_close[i])
            if config.conversion == 'diff':
                index_real.append(target_values[i] + index_close[i])
                index_pred.append(pred_values[i] + index_close[i])
            if config.conversion == 'norm':
                index_real.append(target_values[i]*30 + 67)
                index_pred.append(pred_values[i]*30 + 67)
            if (index_open[i] - index_real[i])*(index_open[i] - index_pred[i]) > 0: profits.append(abs(index_open[i] - index_real[i]))
            else: profits.append(-abs(index_open[i] - index_real[i]))
        else:
            index_real.append('NaN')
            profits.append('Nan')
            if config.conversion == 'rate':
                index_pred.append(pred_values[i]*index_close[i]/100 + index_close[i])
            if config.conversion == 'diff':
                index_pred.append(pred_values[i] + index_close[i])
            if config.conversion == 'norm':
                index_pred.append(pred_values[i]*30 + 67)
    return today, date, index_pred, index_real, index_open, profits, pred_values, target_values, std, asset_rebal



