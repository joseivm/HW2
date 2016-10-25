from plotBoundary import *
from lr_test import *
from svm_test import *
import numpy as np


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

def buildLRPredict(X,Y,C):
    w = trainLRL1norm(X,Y,C)
    predict = constructPredictLR(w)
    return predict

def buildSVMPredict(X,Y,C):
    alpha = trainSVM(X,Y,C)
    predict = constructPredictSVM(X, Y, alpha)
    return predict

def buildPegasosPredict(X,Y):
    return predict


C = 10**0
pos_digits = [7]
neg_digits = [1]

print '======Setting Up======'
trainX,trainY = buildTrainingSet(pos_digits, neg_digits)
valX, valY = buildValidationSet(pos_digits,neg_digits)
testX, testY = buildTestSet(pos_digits,neg_digits)

print '======Training======'
# predict = buildLRPredict(trainX, trainY, C)
predict = buildSVMPredict(trainX, trainY, C)
# predict = buildPegasosPredict(trainX, trainY)
evaluatePredict(trainX, trainY, predict)

print '======Validation======'
evaluatePredict(valX, valY, predict)


