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
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
  return data[:-n_test], data[-n_test:]

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

# fit a model
def model_fit(train, config):
  # unpack config
  n_input, n_nodes, n_epochs, n_batch = config
  # prepare data
  data = series_to_supervised(train, n_input)
  train_x, train_y = data[:, :-1], data[:, -1]
  # define model
  model = Sequential()
  model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='adam')
  # fit
  model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
  return model

# forecast with a pre-fit model
def model_predict(model, history, config):
  # unpack config
  n_input, _, _, _ = config
  # prepare data
  x_input = array(history[-n_input:]).reshape(1, n_input)
  # forecast
  yhat = model.predict(x_input, verbose=0)
  return yhat[0]

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
  # print(' > %.3f' % error)
  return error

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=30):
  # fit and evaluate the model n times
  scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
  return scores

# summarize model performance
def summarize_scores(name, scores):
  # print a summary
  scores_m, score_std = mean(scores), std(scores)
  # print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
  # box and whisker plot
  pyplot.boxplot(scores)
  #pyplot.show()
  return score_std

def randString(length=8):
    # put your letters in the following string
    your_letters='abcdefghiz4789234KNVSDFtrgfhjklm'
    return ''.join((random.choice(your_letters) for i in range(length)))

def MLP(request):
    if request.method == 'POST':
        try:
            file_name = request.POST['filename']
            n_input = int(request.POST['n_input'])
            n_nodes = int(request.POST['n_nodes'])
            n_epoches = int(request.POST['n_epoches'])
            n_batch = int(request.POST['n_batch'])
            my_file = "media/user_{0}/processed_csv/{1}".format(request.user, file_name)
            series = read_csv(my_file, header=0, index_col=0)
            data = series.values
            # data split
            n_test = int(request.POST['ratio'])
            print('MLP')
            # define config
            config = [n_input, n_nodes, n_epoches, n_batch]
            print(n_nodes)
            # grid search
            scores = repeat_evaluate(data, config, n_test)
            # summarize scores
            score = summarize_scores('MLP', scores)

            series_to_supervised(data, n_in=3, n_out=1)
            train = data[:-n_test]
            test = data[-n_test:]
            walk_forward_validation(data, n_test, config)
            history = [x for x in train]
            next_p = model_predict(model_fit(train, config), history, config)
            md = "MLP"
            ems='RMSE'
            hh = model_fit(train,config)
            
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