from plotBoundary import *
import pylab as pl

from cvxopt import matrix, solvers
import numpy as np
# import your SVM training code

def kernel(x1,x2):
    return np.dot(x1,x2)
    # return 1+np.dot(x1.T, x2)+np.dot(x1.T, x2)**2
    # g = 1
    # return np.exp(-g * np.linalg.norm(x1-x2)**2)

def trainSVM(X,Y,C):
    # Carry out training, primal and/or dual
    n,d = X.shape

    # Construct P
    P = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            P[i][j] = Y[i]*Y[j]*kernel(X[i], X[j])
    q = -1 * np.ones(n)
    G1 = np.identity(n)
    G2 = -1*np.identity(n)
    G = np.concatenate((G1, G2))
    h1 = C * np.ones(n)
    h2 = np.zeros(n)
    h = np.concatenate((h1,h2))
    A = np.array(Y.T)
    b = np.zeros(1)

    P = matrix(P, tc='d')
    q = matrix(q, tc='d')
    G = matrix(G, tc='d')
    h = matrix(h, tc='d')
    A = matrix(A, tc='d')
    b = matrix(b, tc='d')

    sol = solvers.qp(P,q,G,h,A,b)

    alpha = sol['x']
    return alpha

# Define the predictSVM(x) function, which uses trained parameters
def constructPredictSVM(X, Y, alpha):
    n,d = X.shape
    SV = []
    for i in xrange(n):
        if alpha[i] >= 10**-6:
            SV.append(i)

    # Calculate theta_0
    theta_0s = []
    for i in SV:
        x_i = X[i]
        y_i = Y[i]
        s = 0
        for t in SV:
            x_t = X[t]
            y_t = Y[t]
            a_t = alpha[t]
            s += a_t*y_t*kernel(x_t,x_i)
        t = (1 - y_i*s)/y_i
        theta_0s.append(t)
    theta_0 = np.median(theta_0s)

    def predictSVM(x):
        s = 0
        for t in SV:
            x_t = X[t]
            y_t = Y[t]
            a_t = alpha[t]
            s += a_t*y_t*kernel(x_t,x)
        return s + theta_0
    return predictSVM



def evaluatePredict(X, Y, predict):
    n = len(X)
    misclass = 0
    misclassifications = []
    for i in xrange(n):
        x = X[i]
        y = Y[i]
        y_pred = np.sign(predict(x))
        if y != y_pred:
            misclass += 1
            misclassifications.append(i)

    misclassrate = misclass*1.0/n
    print "# of misclass: ", misclass
    print "Misclass rate: ", misclassrate
    print "Misclassifications: ", misclassifications
    return misclassrate, misclassifications

if __name__ == "__main__":
    # parameters
    name = '1'
    C = 10**2
    print '======Training======'
    # load data from csv files
    train = loadtxt('data/data'+name+'_train.csv')
    # use deep copy here to make cvxopt happy
    # X = train[:, 0:2].copy()
    # Y = train[:, 2:3].copy()
    X = np.array([[2,2], [2,3], [0,-1], [-3,-2]])
    Y = np.array([[1,1,-1,-1]]).T

    alpha = trainSVM(X,Y,C)
    print alpha

    predictSVM = constructPredictSVM(X, Y, alpha)

    # plot training results
    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')

    # evaluatePredict(X,Y, predictSVM)


    # print '======Validation======'
    # # load data from csv files
    # validate = loadtxt('data/data'+name+'_validate.csv')
    # X = validate[:, 0:2]
    # Y = validate[:, 2:3]
    # # plot validation results
    # plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

    # evaluatePredict(X,Y, predictSVM)

    # print '======Testing======'
    # # load data from csv files
    # test = loadtxt('data/data'+name+'_test.csv')
    # X = test[:, 0:2]
    # Y = test[:, 2:3]
    # # plot validation results
    # plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Test')

    # # evaluatePredict(X,Y, predictSVM)



    pl.show()
