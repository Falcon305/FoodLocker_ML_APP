from math import sqrt
from numpy import array
from numpy import mean
from numpy import std
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from django.shortcuts import render, redirect
import joblib
import os
import random
from processing.models import Alg, Project
from django.contrib.auth.decorators import login_required
from ast import *


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
  return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
  df = DataFrame(data)
  cols = list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
  # put it all together
  agg = concat(cols, axis=1)
  # drop rows with NaN values
  agg.dropna(inplace=True)
  return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
  return sqrt(mean_squared_error(actual, predicted))

# difference dataset
 
def difference(data, order):
  return [data[i] - data[i - order] for i in range(order, len(data))]

# fit a model
def model_fit(train, config):
  # unpack config
  n_input, n_nodes, n_epochs, n_batch, n_diff = config
  # prepare data
  if n_diff > 0:
    train = difference(train, n_diff)
  # transform series into supervised format
  data = series_to_supervised(train, n_in=n_input)
  # separate inputs and outputs
  train_x, train_y = data[:, :-1], data[:, -1]
  # define model
  model = Sequential()
  model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='adam')
  # fit model
 
  model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
  return model

# forecast with the fit model
def model_predict(model, history, config):
  # unpack config
  n_input, _, _, _, n_diff = config
  # prepare data
  correction = 0.0
  if n_diff > 0:
    correction = history[-n_diff]
    history = difference(history, n_diff)
  # shape input for model
  x_input = array(history[-n_input:]).reshape((1, n_input))
  # make forecast
  yhat = model.predict(x_input, verbose=0)
  print(correction + yhat[0])
  # correct forecast if it was differenced
  return correction + yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
  predictions = list()
  # split dataset
  train, test = train_test_split(data, n_test)
  # fit model
  model = model_fit(train, cfg)
  # seed history with training dataset
  history = [x for x in train]
  # step over each time-step in the test set
  for i in range(len(test)):
    # fit model and make forecast for history
    yhat = model_predict(model, history, cfg)
    # store forecast in list of predictions
    predictions.append(yhat)
    # add actual observation to history for the next loop
    history.append(test[i])
  # estimate prediction error
  error = measure_rmse(test, predictions)
  print(' > %.3f' % error)
  return error

# score a model, return None on failure
def repeat_evaluate(data, config, n_test, n_repeats=10):
  # convert config to a key
  key = str(config)
  # fit and evaluate the model n times
  scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
  # summarize score
  result = mean(scores)
  print('> Model[%s] %.3f' % (key, result))
  return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test):
  # evaluate configs
  scores = scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
  # sort configs by error, asc
  scores.sort(key=lambda tup: tup[1])
  return scores

# create a list of configs to try
def model_configs():
  # define scope of configs
  n_input = [12]
  n_nodes = [50, 100]
  n_epochs = [100]
  n_batch = [1, 150]
  n_diff = [0, 12]
  # create configs
  configs = list()
  for i in n_input:
    for j in n_nodes:
      for k in n_epochs:
        for l in n_batch:
          for m in n_diff:
            cfg = [i, j, k, l, m]
            configs.append(cfg)
  print('Total configs: %d' % len(configs))
  return configs

def randString(length=8):
    # put your letters in the following string
    your_letters='abcdefghiz4789234KNVSDFtrgfhjklm'
    return ''.join((random.choice(your_letters) for i in range(length)))


@login_required
def MLP_GRID(request):
    if request.method == 'POST':
        try:
            file_name = request.POST['filename']
            my_file = "media/user_{0}/processed_csv/{1}".format(request.user, file_name)
            series = read_csv(my_file, header=0, index_col=0)
            data = series.values
            # data split
            n_test = int(request.POST['ratio'])
            print('MLP (Grid Search)')

            cfg_list = model_configs()
            # grid search
            scores = grid_search(data, cfg_list, n_test)
            print('done')
            # list top 3 configs
            train = data[:-n_test]
            fo = scores[0][0]
            fo = literal_eval(fo)
            hh = model_fit(train, fo)
            history = [x for x in train]
            next_p = model_predict(model_fit(train, fo), history, fo)
            print(next_p)
        
            md = "MLP Grid Search"
            ems='RMSE'
            score = scores[0][1]
            if not os.path.exists("media/user_{}/trained_model".format(request.user)):
                   os.makedirs("media/user_{}/trained_model".format(request.user))
            strt = randString()
            nw = strt + file_name.split('.')[0]
            hh.save("media/user_{0}/trained_model/{1}".format(request.user, nw + '.h5'))
            # download_link = "media/user_{0}/trained_model/{1}".format(request.user,  'fool.pkl')
            # joblib.dump(model_fit(train,config), download_link)
            return render(request, 'models/result.html', {"md": md,                      
                                                              "score": score,
                                                              'ems': ems,
                                                              'nw': nw,
                                                              'next_p':next_p})

        except Exception as e:
            return render(request, 'models/result.html', {"Error": e})