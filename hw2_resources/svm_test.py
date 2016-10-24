from plotBoundary import *
import pylab as pl

from cvxopt import matrix, solvers
import numpy as np
# import your SVM training code

# parameters
name = '1'
C = 10**0
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# Carry out training, primal and/or dual
n,d = X.shape
print n,d

# Construct P
P = np.zeros((n,n))
for i in xrange(n):
    for j in xrange(n):
        P[i][j] = np.dot(Y[i]*X[i], Y[j]*X[j])
# P = P.T
# Construct q
q = -1 * np.ones(n)
# Construct G
G1 = np.identity(n)
G2 = -1*np.identity(n)
G = np.concatenate((G1, G2))
# Construct h
h1 = C * np.ones(n)
h2 = np.zeros(n)
h = np.concatenate((h1,h2))
# Construct A
A = np.array(Y.T)
# Construct b
b = np.zeros(1)

P = matrix(P, tc='d')
q = matrix(q, tc='d')
G = matrix(G, tc='d')
h = matrix(h, tc='d')
A = matrix(A, tc='d')
b = matrix(b, tc='d')

sol = solvers.qp(P,q,G,h,A,b)

alpha = sol['x']
# print alpha
# print sol['primal objective']
# print A
# print np.dot(A, alpha)

# Define the predictSVM(x) function, which uses trained parameters
def constructPredictSVM(X, Y, alpha):
    n,d = X.shape
    SV = []
    for i in xrange(n):
        if alpha[i] >= 10**-4:
            SV.append(i)

    # Calculate theta_0
    theta_0s = []
    for i in SV:
        x_i = X[i]
        y_i = Y[i]
        s = 0
        for t in SV:
            x_t = X[t]
            a_t = alpha[t]
            s += a_t*np.dot(x_t,x_i)
        t = (1 - y_i*s)/y_i
        print "t: ", t
        theta_0s.append(t)
    theta_0 = np.median(theta_0s)
    print 'theta_0', theta_0

    def predictSVM(x):
        s = 0
        for t in SV:
            x_t = X[t]
            y_t = Y[t]
            a_t = alpha[t]
            s += a_t*y_t*np.dot(x_t,x)
        return s + theta_0
    return predictSVM

predictSVM = constructPredictSVM(X, Y, alpha)

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
