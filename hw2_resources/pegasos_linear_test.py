import numpy as np
from plotBoundary import *
import pylab as pl

# import your LR training code


# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
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
				# print 'theta',theta
				theta = (1-learning_rate*lambd)*theta + learning_rate*y_i*x_i
				theta_0 = theta_0 + learning_rate*y_i
				w = np.insert(theta,0,theta_0)
				# w = np.concatenate((theta_0,theta),axis=0)
			else:
				theta = (1-learning_rate*lambd)*theta
				# print theta_0, theta
				# w = np.concatenate(([theta_0],theta),axis=0)
				w = np.insert(theta,0,theta_0)
		# print w
		iterations += 1
	print w
	print 1.0/np.linalg.norm(w)
	return w

def pegasos_alg_2(X,Y,lambd,max_iter):
	t = 0
	ones = np.ones([X.shape[0],1])
	new_X = np.concatenate((ones,X),axis=1)
	w = np.zeros(new_X.shape[1])
	iterations = 0
	while iterations < max_iter:
		for i in xrange(len(X)):
			t += 1
			learning_rate = 1.0/(t*lambd)
			y_i = Y[i]
			x_i = new_X[i]
			theta = w.copy()
			theta_0 = w[0]
			# print theta_0
			if y_i*(np.dot(theta,x_i)) < 1:
				# print 'theta',theta
				theta = (1-learning_rate*lambd)*theta + learning_rate*y_i*x_i
				theta_0 = theta_0 + learning_rate*y_i
				theta[0] = theta_0
				w = theta
			else:
				theta = (1-learning_rate*lambd)*theta 
				# print theta_0, theta
				theta[0] = theta_0
				w = theta
		# print w
		iterations += 1
	print w
	print 1.0/np.linalg.norm(w)
	return w

def kernelized_pegasos_alg(X,Y,lambd,K,max_iter):
	t = 0
	# w = np.zeros(X.shape[1])
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

def build_K_matrix(X,kernel):
	n = X.shape[0]
	K = np.zeros([n,n])
	for i in xrange(n):
		for j in xrange(n):
			K[i][j] = kernel(X[i],X[j])
	return K
# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###


def predict_linearSVM_kernelized(x):
	summ = 0
	for i in alphas:
		summ += alphas[i]*kernel(X[i],x)
	return np.sign(summ)

def predict_linearSVM(x):
	theta_0 = w[0]
	theta = w[1:]
	return np.dot(theta,x)+theta_0

# # plot training results
for i in xrange(-2,3):


# plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
# pl.show()

