import warnings
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from django.shortcuts import render, redirect
 
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order, rat):
	# prepare training dataset
	train, test = X[0:-rat], X[-rat:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values, rat):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order, rat)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    #return best_score


def ARIMA_M(request):
    if request.method == 'POST':
        try:
            file_name = request.POST['filename']
            my_file = "media/user_{0}/processed_csv/{1}".format(request.user, file_name)
            series = read_csv(my_file, header=0, index_col=0, squeeze=True)
            p_values = [0, 1, 2, 4, 6, 8, 10]
            d_values = range(0, 3)
            q_values = range(0, 3)
            rat = int(request.POST['ratio'])
            warnings.filterwarnings("ignore")
            print('ARIMA')
            score = evaluate_models(series.values, p_values, d_values, q_values,rat)
            md = "ARIMA"
            ems = 'MSE'
            return render(request, 'models/result.html', {"md": md,                      
                                                              "score": score,
                                                              'ems': ems})

        except Exception as e:
            return render(request, 'models/result.html', {"Error": e})