import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code


# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
def pegasos_alg(X,Y,lambd,max_iter):
	t = 0
	w = np.zeros(X.shape[1]+1)
	print w.shape
	iterations = 0
	while iterations < max_iter:
		for i in xrange(len(X)):
			t += 1
			learning_rate = 1.0/(t*lambd)
			y_i = Y[i]
			x_i = X[i]
			theta = w[1:]
			theta_0 = w[0]
			if y_i*(np.dot(theta,x_i) + theta_0) < 1:

				theta = (1-learning_rate*lambd)*theta + learning_rate*y_i*x_i
				theta_0 = theta_0 + learning_rate*y_i
				w = np.concatenate((theta_0,theta),axis=0)
			else:
				theta = (1-learning_rate*lambd)*theta + learning_rate*y_i*x_i
				w = np.concatenate((theta_0,theta),axis=0)
		iterations += 1

	return w

w = pegasos_alg(X,Y,10,10**4)
print w

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###
def predict_linearSVM(x):
	theta_0 = w[0]
	theta = w[1:]
	return np.sign(np.dot(theta,x)+theta_0)

# plot training results
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
pl.show()

