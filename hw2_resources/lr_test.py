# from numpy import *
import numpy as np
from plotBoundary import *
import pylab as pl
from sklearn.linear_model import LogisticRegression
from scipy.optimize import fmin
# import your LR training code


def sigmoid(x):
	return 1.0/(1+np.exp(-x))

def batch_gradient_descent(obj_func,gradient_func,init_guess,step_size,convergence_criterion,max_iter):
	I = []
	S = []
	size = init_guess.shape[0]
	w_old = init_guess
	w_new = np.zeros(size)
	alpha = step_size
	old_cost = obj_func(w_old)
	new_cost = np.inf
	iterations = 0
	converged = False
	while (not converged and iterations<max_iter):

		if iterations%1000==0: print iterations, np.linalg.norm(w_new)

		w_new = w_old - alpha*gradient_func(w_old)
		new_cost = obj_func(w_new)
		
		if np.absolute(new_cost - old_cost) < convergence_criterion:
			converged = True

		old_cost = new_cost
		w_old = w_new

		# if iterations%7==0: print w_new, np.linalg.norm(w_new)
		I.append(iterations)
		S.append(np.linalg.norm(w_new))
		iterations+=1
		

	print w_new
	print "min val", new_cost
	print iterations
	return w_new, I,S # FIX THIS

def LR_obj_maker_batch1(X,Y,lambd):
	def LR(w):
		theta = w[1:]
		theta_0 = w[0]
		ind_1 = (Y+1)/2
		ind_0 = (1-Y)/2
		log_s = np.log(sigmoid(np.dot(theta,X.T)+theta_0))
		reg_term = lambd*np.linalg.norm(theta)**2
		return -1*np.sum(ind_1*log_s+ind_0*(1-log_s))+reg_term

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

def trainBatchGradientDescent(X,Y,step_size, convergence_criterion,lmbda,max_iter):
	guess = np.random.random(3)*10
	LR_obj = LR_obj_maker_batch1(X,Y,lmbda)
	LR_grad = LR_grad_maker_batch1(X,Y,lmbda)
	w = batch_gradient_descent(LR_obj,LR_grad,guess,step_size,convergence_criterion,max_iter)
	return w

def trainLRL1norm(X,Y,C,s):
	model = LogisticRegression(penalty='l1',C=C,intercept_scaling=s)
	model.fit(X,Y)
	w = np.concatenate((model.intercept_,model.coef_[0]))
	return w, model # FIX THIS
 
def trainLRL2norm(X,Y,C,s):
	model = LogisticRegression(C=C,intercept_scaling=s)
	model.fit(X,Y)
	w = np.concatenate((model.intercept_,model.coef_[0]))
	return w, model # FIX THIS

def constructPredictLR(w):
	def predictLR(x):
		weight_vector = w[1:]
		w_0 = w[0]
		exp_term = exp(-(dot(weight_vector,x)+w_0))
		s = 1/(1+exp_term)
		return s - 0.5
	return predictLR


if __name__ == "__main__":
	name = '1'

	# print '======Training======'
	# load data from csv files
	train = loadtxt('data/data'+name+'_train.csv')
	X = train[:,0:2] # (400,2)
	Y = train[:,2:3] #(400,1)

	Y = Y.ravel()

	C = 10**8
	# Cs = [.0001,.001,.01,.1,1,10,100,1000,10000]
	step_size = 10**-2
	convergence_criterion = 10**-3


	# w,I,S = trainBatchGradientDescent(X, Y, step_size, convergence_criterion,0,275000)
	w1,I1,S1 = trainBatchGradientDescent(X, Y, step_size, convergence_criterion,1,500000)
	# diff = len(I)-len(I1)
	# conv_val = S1[len(I1)-1]
	# rest = conv_val*np.ones(diff)
	# S1 = np.concatenate((S1,rest),axis=0)

	# for i in [-3,-2,-1,0,1,2,3]:
	# 	C = 10**i
	# 	s = 1
	# 	if i < 0 : s = 10**np.abs(i)
	# 	w, model = trainLRL2norm(X,Y,C,s)
	# 	CER = 1- model.score(X,Y)
	# 	print 'lambda =', 1.0/C, 'Scaling:',s, 'CER:', CER, "w:", w, "norm:", np.linalg.norm(w)
	# 	predictLR = constructPredictLR(w)
	# 	title = "lambda = "+str(1.0/C)+" LR2"
	# 	plotDecisionBoundary(X, Y, predictLR, [0], title = title)
	# 	pl.show()

	# print " "
	# for i in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
	# 	C = 10**i
	# 	w, model = trainLRL2norm(X,Y,C,1)
	# 	CER = 1- model.score(X,Y)
	# 	print 'lambda =', 1.0/C, 'CER:', CER, "w:", w, "norm:", np.linalg.norm(w)

	# w2, model = trainLRL2norm(X,Y,C,1)
	# print w2, np.linalg.norm(w2)

	# pl.plot(I,S,'b',label= "lambda = 0")
	pl.plot(I1,S1,'r', label= "lambda = 1")

	pl.legend(loc='best')
	# predictLR = constructPredictLR(w)

	# plot training results
	# plotDecisionBoundary(X, Y, predictLR, [0], title = 'LR Train')
	pl.show()


	# print '======Validation======'
	# # load data from csv files
	# validate = loadtxt('data/data'+name+'_validate.csv')
	# X = validate[:,0:2]
	# Y = validate[:,2:3]

	# # plot validation results
	# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
	# pl.show()
