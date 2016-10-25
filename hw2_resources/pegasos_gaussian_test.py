from plotBoundary import *
import pylab as pl
import numpy as np
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
epochs = 500;
lmbda = .02;
# gamma = 2e-2;

# K = zeros((n,n));
### TODO: Compute the kernel matrix ###
def g_kernel_maker(gamma):
	def g_kernel(x1,x2):
		term = -gamma*np.linalg.norm(x1-x2)**2
		return np.exp(term)
	return g_kernel

def build_K_matrix(X,kernel):
	n = X.shape[0]
	K = np.zeros([n,n])
	for i in xrange(n):
		for j in xrange(n):
			K[i][j] = kernel(X[i],X[j])
	return K


def predict_gaussianSVM(x):
	summ = 0
	for i in range(len(alphas)):
		summ += alphas[i]*g_kernel(X[i],x)
	return summ


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

def train_gaussianSVM(X,Y,K,lmbda,epochs):
	return kernelized_pegasos_alg(X,Y,lmbda,K,epochs)

for i in [0,1,2]:
	print '2^',i
	g_kernel = g_kernel_maker(2**i)
	K = build_K_matrix(X,g_kernel)


	alphas = train_gaussianSVM(X, Y, K, lmbda, epochs);


	# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
	### TODO:  define predict_gaussianSVM(x) ###

	# plot training results
	plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
	pl.show()
