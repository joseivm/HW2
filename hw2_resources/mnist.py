from plotBoundary import *
from lr_test import *
from svm_test import *
from pegasos_linear_test import *
import numpy as np

from matplotlib import pyplot as plt

def buildTrainingSet(pos_digits, neg_digits):
    return buildSet(pos_digits, neg_digits, 200, 0)
def buildValidationSet(pos_digits, neg_digits):
    return buildSet(pos_digits, neg_digits, 150, 200)
def buildTestSet(pos_digits, neg_digits):
    return buildSet(pos_digits, neg_digits, 150, 350)

def buildSet(pos_digits, neg_digits, num, used):
    X = np.array([]).reshape(0,784)
    for digit in pos_digits:
        X = np.concatenate((X,loadDigit(digit, num, used)))
    for digit in neg_digits:
        X = np.concatenate((X,loadDigit(digit, num, used)))
    Y1 = np.ones((num*len(pos_digits), 1))
    Y2 = -1*np.ones((num*len(neg_digits), 1))
    Y = np.concatenate((Y1,Y2))
    return X,Y
    
def loadDigit(digit, num, used):
    data = loadtxt('data/mnist_digit_' + str(digit) + '.csv')
    X = data[used:used+num]
    for x in X:
        for i in xrange(len(x)):
            x[i] = 2.0*x[i] / 255.0 - 1
    return X

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

    print "# of misclass: ", misclass
    print "Misclass rate: ", misclass*1.0/n
    print "Misclassifications: ", misclassifications
    return misclassifications

def buildLRPredict(X,Y,C):
    w = trainLRL1norm(X,Y,C)
    predict = constructPredictLR(w)
    return predict

def buildSVMPredict(X,Y,C):
    alpha = trainSVM(X,Y,C)
    predict = constructPredictSVM(X, Y, alpha)
    return predict

def buildPegasosPredict(X,Y,lmbda,max_iter):
    w = pegasos_alg(X,Y,lmbda,max_iter)
    def predict_linearSVM(x):
        theta_0 = w[0]
        theta = w[1:]
        return np.dot(theta,x)+theta_0
    return predict_linearSVM

def openMisclassifications(X,Y,misclassifications):
    for i in misclassifications:
        data = X[i].reshape((28,28))
        plt.imshow(data, cmap = plt.get_cmap('gray'))
        plt.show()


C_SVM = [10**0, 10**1, 10**2]
pos_digits = [1,3,5,7,9]
neg_digits = [0,2,4,6,8]
print "Postive Digits:  ", pos_digits
print "Negative Digits: ", neg_digits
print '======Setting Up======'
trainX,trainY = buildTrainingSet(pos_digits, neg_digits)
valX, valY = buildValidationSet(pos_digits,neg_digits)
testX, testY = buildTestSet(pos_digits,neg_digits)

print '======Training======'
# predict = buildLRPredict(trainX, trainY, C)
for C in C_SVM:
    predict = buildSVMPredict(trainX, trainY, C)

# predict = buildPegasosPredict(trainX, trainY)
misclassifications = evaluatePredict(trainX, trainY, predict)

openMisclassifications(trainX, trainY, misclassifications)

print '======Validation======'
misclassifications = evaluatePredict(valX, valY, predict)
openMisclassifications(valX, valY, misclassifications)

print '======Testing======'
misclassifications = evaluatePredict(testX, testY, predict)
openMisclassifications(testX, testY, misclassifications)

