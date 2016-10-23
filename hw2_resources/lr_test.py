# from numpy import *
import numpy as np
from plotBoundary import *
import pylab as pl
from sklearn.linear_model import LogisticRegression
from scipy.optimize import fmin
# import your LR training code

# parameters
name = '3'
# print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2] # (400,2)
Y = train[:,2:3] #(400,1)

Y = Y.ravel()

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

def stochastic_gradient_descent(obj_func,gradient_func,init_guess,step_size,convergence_criterion,X,y):
	num_samples = X.shape[0]
	w_old = init_guess
	w_new = np.zeros(len(w_old))
	alpha = step_size
	old_cost = obj_func(w_old)
	new_cost = np.inf
	iterations=1
	converged = False
	while (not converged):
		indices = np.arange(num_samples)
		np.random.shuffle(indices)
		# print iterations
		for i in indices:

			w_new = w_old - alpha*gradient_func(w_old,X[i],y[i])
			
		new_cost = obj_func(w_new)
		
		if np.absolute(new_cost - old_cost) < convergence_criterion:
			converged = True

		print new_cost
		old_cost = new_cost
		w_old = w_new

		
		iterations+=1
		

	print 'minimum occurs at: ', w_new
	print 'minimum value', new_cost
	return w_new

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
		sig_arg = np.dot(theta,X.T)
		R = (sigmoid(sig_arg+theta_0)-new_y) #temp variable
		big_array = (R*X.T).T
		summ = np.sum(big_array,axis=0) + 2*lambd*theta
		return np.insert(summ,0,0)

	return LR_grad

def LR_obj_maker_batch(X,Y,lambd):
	def LR(w):
		theta = w[1:]
		theta_0 = w[0]
		exp_arg = -Y*(np.dot(X,theta) + theta_0)
		log_arg = 1+np.exp(exp_arg)
		log = np.log(log_arg)
		return np.sum(log) + lambd*np.linalg.norm(theta)**2
	return LR

def LR_grad_maker_batch(X,Y,lambd):
	product = 0
	def LR_grad(w):
		theta = w[1:]
		theta_0 = w[0]
		exp_arg = Y*(np.dot(X,theta) + theta_0)
		print product.shape
		print exp_arg.shape
		return -product/(np.exp(exp_arg)+1) + 2*lambd*theta

	return LR_grad

def grad_approx(x,h,obj_func):
	n = len(x)
	gradient = []
	for i in range(n):
		unit_vec = np.zeros(n)
		unit_vec[i] = 1
		df_dxi = (obj_func(x+h*unit_vec)-obj_func(x-h*unit_vec))/(2*h)
		gradient.append(df_dxi)

	return np.array(gradient)

LR_obj = LR_obj_maker_batch1(X,Y,100)
LR_grad = LR_grad_maker_batch1(X,Y,100)

# for i in xrange(10):
# 	w = np.random.random(3)
# 	print 'real:', LR_grad(w)
	# print 'approx:', grad_approx(w,.01,LR_obj)




guess = np.random.random(3)
# print fmin(LR_obj,guess)


w = batch_gradient_descent(LR_obj,LR_grad,guess,.001,.000001)

Cs = [.0001,.001,.01,.1,1,10,100,1000,10000]
# lambda = 1000,100,10,1,0.1,0.01,0.001,0.0001
# for val in Cs:

# l1_model = LogisticRegression(penalty='l1',C=val)
l2_model = LogisticRegression(C=1)

# l1_model.fit(X,Y)
l2_model.fit(X,Y)

model = l2_model


# Carry out training.

# Define the predictLR(x) function, which uses trained parameters
# def predictLR(x):
# 	weight_vector = model.coef_
# 	w_0 = model.intercept_
# 	exp_term = exp(-(dot(weight_vector,x)+w_0))
# 	return 1/(1+exp_term)

def predictLR(x):
	weight_vector = w[1:]
	w_0 = w[0]
	exp_term = exp(-(dot(weight_vector,x)+w_0))
	return 1/(1+exp_term)

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')
pl.show()


# print '======Validation======'
# # load data from csv files
# validate = loadtxt('data/data'+name+'_validate.csv')
# X = validate[:,0:2]
# Y = validate[:,2:3]

# # plot validation results
# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
# pl.show()
