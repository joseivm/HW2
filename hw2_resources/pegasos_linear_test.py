import numpy as np
from plotBoundary import *
import pylab as pl

def kernel(x1,x2):
    return np.dot(x1, x2)

def g_kernel_maker(gamma):
	def g_kernel(x1,x2):
		term = -gamma*np.linalg.norm(x1-x2)**2
		return np.exp(term)
	return g_kernel

def pegasos_alg(X,Y,lambd,max_iter):
	t = 0
	w = np.zeros(X.shape[1]+1)
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
				w = np.insert(theta,0,theta_0)
			else:
				theta = (1-learning_rate*lambd)*theta
				w = np.insert(theta,0,theta_0)
		iterations += 1
	return w

def kernelized_pegasos_alg(X,Y,lambd,K,max_iter):
	t = 0
	alphas = np.zeros(X.shape[0])
	iterations = 0
	while iterations < max_iter:
		for i in xrange(len(X)):
			t += 1
			learning_rate = 1.0/(t*lambd)
			y_i = Y[i]
			alpha_i = alphas[i]

			summ = 0
			for j in range(len(alphas)):
				summ += alphas[j]*K[j][i]

			if y_i*summ < 1:
				alphas[i] = (1-learning_rate*lambd)*alpha_i + learning_rate*y_i 

			else:
				alphas[i] = (1-learning_rate*lambd)*alpha_i 
		iterations += 1
	return alphas
def constructPredictPegasosLinearSVM(w):
	def predict_linearSVM(x):
		theta_0 = w[0]
		theta = w[1:]
		return np.dot(theta,x)+theta_0
	return predict_linearSVM

if __name__ == "__main__":
	# load data from csv files
	name = '1'
	train = loadtxt('data/data'+name+'_train.csv')
	X = train[:,0:2]
	Y = train[:,2:3]

	for i in range(-10,3):
		lambd = 2**-i
		print 'lambda = 2^',i
		max_iter = 500
		w = pegasos_alg(X,Y,lambd,max_iter)
		predict = constructPredictPegasosLinearSVM(w)

		plotDecisionBoundary(X, Y, predict, [-1,0,1], title = 'Linear SVM')
		pl.show()

