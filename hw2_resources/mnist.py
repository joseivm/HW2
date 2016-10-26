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
            # x[i] = 2.0*x[i] / 255.0 - 1
            x[i] = x[i]
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

    misclassrate = misclass*1.0/n
    print "# of misclass: ", misclass
    print "Misclass rate: ", misclassrate
    print "Misclassifications: ", misclassifications
    return misclassrate, misclassifications

def buildLRPredict(X,Y,C):
    w = trainLRL1norm(X,Y,C)
    predict = constructPredictLR(w)
    return predict

def buildSVMPredict(X,Y,C):
    alpha = trainSVM(X,Y,C)
    predict = constructPredictSVM(X, Y, alpha)
    return predict

def buildPegasosPredict(X,Y,lambd,max_iter):
    w = pegasos_alg(X,Y,lambd,max_iter)
    predict = constructPredictPegasosLinearSVM(w)
    return predict

def openMisclassifications(X,Y,misclassifications):
    for i in misclassifications:
        data = X[i].reshape((28,28))
        plt.imshow(data, cmap = plt.get_cmap('gray'))
        plt.show()


C_SVM = [10**0, 10**1, 10**2]
C_LR = [10**0, 10**1, 10**2]
lambds = [2**-10]
max_iter = 500

pos_digits = [1]
neg_digits = [7]
print "Postive Digits:  ", pos_digits
print "Negative Digits: ", neg_digits
print '======Setting Up======'
trainX,trainY = buildTrainingSet(pos_digits, neg_digits)
valX, valY = buildValidationSet(pos_digits,neg_digits)
testX, testY = buildTestSet(pos_digits,neg_digits)

print '======Training======'
predicts = []
# for C in C_LR:
#     predict = buildLRPredict(trainX, trainY, C)
#     predicts.append((C, predict))
for C in C_SVM:
    predict = buildSVMPredict(trainX, trainY, C)
    predicts.append((C,predict))
# for lambd in lambds:
#     predict = buildPegasosPredict(trainX, trainY, lambd, max_iter)
#     predicts.append((lambd,predict))

# misclassifications = evaluatePredict(trainX, trainY, predict)
# openMisclassifications(trainX, trainY, misclassifications)

print '======Validation======'
bestParam = 0
bestpredict = 0
bestmisclassifications = []
bestmisclassrate = 1
for param,predict in predicts:
    misclassrate, misclassifications = evaluatePredict(valX, valY, predict)
    if misclassrate < bestmisclassrate:
        bestParam = param
        bestpredict = predict
        bestmisclassifications = misclassifications

print '======Best Validation======'
print "Param: ", bestParam
print "Misclassification rate: ", bestmisclassrate
print "Misclassifications: ", bestmisclassifications
openMisclassifications(valX, valY, bestmisclassifications)

print '======Testing======'
misclassrate,misclassifications = evaluatePredict(testX, testY, bestpredict)
print "Misclassification rate: ", misclassrate
openMisclassifications(testX, testY, misclassifications)

