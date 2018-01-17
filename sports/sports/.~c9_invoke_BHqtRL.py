from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import requests
import pandas as pd
from BeautifulSoup import BeautifulSoup
import datetime
import time
import urllib2
import numpy as np
import json
import sys
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import neural_network
from sklearn.naive_bayes import GaussianNB

def ReadDataSet(filename):
    return pd.read_csv(filename, sep = ',')

def Scorer(predictions, actuals, ouA, threshold = 0.5):
    # predOver = ((predictions / ouA) > (1 + threshold))
    # predUnder = ((predictions / ouA) < (1 - threshold))
    # betsMade = (float(predOver.sum() + predUnder.sum()))
    # print betsMade
    # actualOver = (actuals - ouA) > 0
    # actualUnder = (actuals - ouA) < 0
    # overRight = (actualOver * predOver)
    # underRight = (actualUnder * predUnder)
    # acc = (float(overRight.sum() + underRight.sum())) / (float(predOver.sum() + predUnder.sum()))
    
    acc = float((np.sign(predictions) == np.sign(actuals)).sum()) / float(len(predictions))
    #print acc
    return -acc
      

def score(params):
    print "Training with params : "
    print params
    
    if 'scale' in params:
        if params['scale'] == 1:
            preprocessing.scale(X_train),preprocessing.scale(X_test),preprocessing.scale(y_train),preprocessing.scale(y_test),preprocessing.scale(ou_test)
    del params['scale']
    
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    clf = svm.svc(**params)
    a = clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    print preds
    score = Scorer(preds, y_test, ou_test)
    print score
    return {'loss': score, 'status': STATUS_OK}

def optimize(trials):
    space = {
            #'C': hp.uniform('C', 7000,11000),
            #'epsilon' : hp.uniform('epsilon', 0, 20),
            #'kernel': hp.choice('kernel', ['rbf']),
            #'gamma': hp.uniform('gamma', 0.011,0.018),
            'scale': hp.choice('scale', [0,0]),
            #'degree' : hp.choice('degree', [1,2])
            #'probability' : hp.choice('probability', [True,True])
            #'n_estimators' : hp.choice('n_estimators', [500])
            
    }
    
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=200)
    print best
    return best

filename = "dataSet.csv"
df = ReadDataSet(filename)

X, y, ou = df.ix[:,3:].values, (df.ix[:,1].values-df.ix[:,2].values), df.ix[:,3].values 
print "Splitting data into train and valid ...\n\n"

trainSize = 0.8
Size = len(X)
X_train,y_train, ou_train = X[:Size*trainSize], np.sign(y[:Size*trainSize]), ou[:Size*trainSize]
X_test,y_test, ou_test = X[Size*trainSize:], np.sign(y[Size*trainSize:]), ou[Size*trainSize:]

trials = Trials()

a = optimize(trials)
