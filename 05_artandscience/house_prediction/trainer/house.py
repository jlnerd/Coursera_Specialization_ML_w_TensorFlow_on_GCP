import os
import math
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

def train(output_dir, batch_size, learning_rate):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Read dataset and split into train and eval
  df = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")
  df['num_rooms'] = df['total_rooms'] / df['households']
  msk = np.random.rand(len(df)) < 0.8
  traindf = df[msk]
  evaldf = df[~msk]

  # Train and eval input functions
  SCALE = 100000
  
  train_input_fn = tf.estimator.inputs.pandas_input_fn(x = traindf[["num_rooms"]],
                                                       y = traindf["median_house_value"] / SCALE,  # note the scaling
                                                       num_epochs = None,
                                                       batch_size = batch_size, # note the batch size
                                                       shuffle = True)
  
  eval_input_fn = tf.estimator.inputs.pandas_input_fn(x = evaldf[["num_rooms"]],
                                                      y = evaldf["median_house_value"] / SCALE,  # note the scaling
                                                      num_epochs = 1,
                                                      batch_size = len(evaldf),
                                                      shuffle=False)
  
  # Define feature columns
  features = [tf.feature_column.numeric_column('num_rooms')]
  
  def train_and_evaluate(output_dir):
    # Compute appropriate number of steps
    num_steps = (len(traindf) / batch_size) / learning_rate  # if learning_rate=0.01, hundred epochs

    # Create custom optimizer
    myopt = tf.train.FtrlOptimizer(learning_rate = learning_rate) # note the learning rate

    # Create rest of the estimator as usual
    estimator = tf.estimator.LinearRegressor(model_dir = output_dir, 
                                             feature_columns = features, 
                                             optimizer = myopt)
    #Add rmse evaluation metric
    def rmse(labels, predictions):
      pred_values = tf.cast(predictions['predictions'],tf.float64)
      return {'rmse': tf.metrics.root_mean_squared_error(labels*SCALE, pred_values*SCALE)}
    estimator = tf.contrib.estimator.add_metrics(estimator,rmse)

    train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn,
                                        max_steps = num_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn,
                                      steps = None)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  # Run the training
  shutil.rmtree(output_dir, ignore_errors=True) # start fresh each time
  train_and_evaluate(output_dir)
    
if __name__ == '__main__' and "get_ipython" not in dir():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type = float, 
      default = 0.01
  )
  parser.add_argument(
      '--batch_size',
      type = int, 
      default = 30
  ),
  parser.add_argument(
      '--job-dir',
      help = 'GCS location to write checkpoints and export models.',
      required = True
  )
  args = parser.parse_args()
  print("Writing checkpoints to {}".format(args.job_dir))
  train(args.job_dir, args.batch_size, args.learning_rate)