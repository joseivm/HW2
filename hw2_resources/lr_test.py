# from numpy import *
import numpy as np
from plotBoundary import *
import pylab as pl
from sklearn.linear_model import LogisticRegression
from scipy.optimize import fmin
# import your LR training code


def sigmoid(x):
	return 1.0/(1+np.exp(-x))

def batch_gradient_descent(obj_func,gradient_func,init_guess,step_size,convergence_criterion):
	size = init_guess.shape[0]
	w_old = init_guess
	w_new = np.zeros(size)
	alpha = step_size
	old_cost = obj_func(w_old)
	new_cost = np.inf
	iterations = 0
	converged = False
	while (not converged):

		# print iterations
		w_new = w_old - alpha*gradient_func(w_old)
		new_cost = obj_func(w_new)
		
		if np.absolute(new_cost - old_cost) < convergence_criterion:
			converged = True

		old_cost = new_cost
		w_old = w_new

		
		iterations+=1
		

	# print 'minimum occurs at: ', w_new
	print "min val", new_cost
	print iterations
	return w_new

def LR_obj_maker_batch1(X,Y,lambd):
	def LR(w):
		theta = w[1:]
		theta_0 = w[0]
		ind_1 = (Y+1)/2
		ind_0 = (1-Y)/2
		log_s = np.log(sigmoid(np.dot(theta,X.T)+theta_0))
		reg_term = lambd*np.linalg.norm(theta)**2
		return - np.sum(ind_1*log_s+ind_0*(1-log_s))

	return LR

def LR_grad_maker_batch1(X,Y,lambd):
	def LR_grad(w):
		theta_0 = w[0]
		theta = w[1:]
		new_y = (Y+1)/2
		const_term = np.ones([1,X.shape[0]]) # (1,400)
		new_X = np.concatenate((const_term,X.T),axis=0) # (400,) (2,400)
		sig_arg = np.dot(w,new_X)
		R = (sigmoid(sig_arg)-new_y) #temp variable
		big_array = (R*new_X).T
		reg_term = 2*lambd*theta
		summ = np.sum(big_array,axis=0) + np.insert(reg_term,0,0)
		return summ

	return LR_grad

def trainBatchGradientDescent(X,Y,step_size, convergence_criterion):
	guess = np.random.random(3)
	LR_obj = LR_obj_maker_batch1(X,Y,1)
	LR_grad = LR_grad_maker_batch1(X,Y,1)
	w = batch_gradient_descent(LR_obj,LR_grad,guess,step_size,convergence_criterion)
	return w

def trainLRL1norm(X,Y,C):
	model = LogisticRegression(penalty='l1',C=C)
	model.fit(X,Y)
	w = np.concatenate((model.intercept_,model.coef_[0]))
	return w

def trainLRL2norm(X,Y,C):
	model = LogisticRegression(C=C)
	model.fit(X,Y)
	w = np.concatenate((model.intercept_,model.coef_[0]))
	return w

def constructPredictLR(w):
	def predictLR(x):
		weight_vector = w[1:]
		w_0 = w[0]
		exp_term = exp(-(dot(weight_vector,x)+w_0))
		s = 1/(1+exp_term)
		return s - 0.5
	return predictLR


if __name__ == "__main__":
	name = '3'

	# print '======Training======'
	# load data from csv files
	train = loadtxt('data/data'+name+'_train.csv')
	X = train[:,0:2] # (400,2)
	Y = train[:,2:3] #(400,1)

	Y = Y.ravel()

	C = 10**0
	# Cs = [.0001,.001,.01,.1,1,10,100,1000,10000]
	step_size = 10**-3
	convergence_criterion = 10**-8

	# w = trainBatchGradientDescent(X, Y, step_size, convergence_criterion)
	# w = trainLRL1norm(X,Y,C)
	w = trainLRL2norm(X,Y,C)

	predictLR = constructPredictLR(w)

	# plot training results
	plotDecisionBoundary(X, Y, predictLR, [0], title = 'LR Train')
	pl.show()


	# print '======Validation======'
	# # load data from csv files
	# validate = loadtxt('data/data'+name+'_validate.csv')
	# X = validate[:,0:2]
	# Y = validate[:,2:3]

	# # plot validation results
	# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
	# pl.show()
