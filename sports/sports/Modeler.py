import requests
import pandas as pd
from BeautifulSoup import BeautifulSoup
import datetime
import time
import urllib2
import numpy as np
import json
from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
from sklearn import svm
from sklearn import neighbors

def ReadDataSet(filename):
    return pd.read_csv(filename, sep = ',')
    
def Score(rule,predictions, actuals, ou):
    predOver = predictions > ou
    actualOver = actuals > ou
    results = (predOver == actualOver)
    return float(results.sum()) / float(len(predictions))
    

def hyperopt_train_test(params):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
            del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
            del params['scale']
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space4rf = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'n_estimators': hp.choice('n_estimators', range(1,20)),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
} 

  
    
if __name__ == "__main__":
    filename = "dataSet.csv"
    df = ReadDataSet(filename)
    
    trainFrac = 0.8
    testFrac = 1- trainFrac
    n = len(df)
    
    trainInputs = df.ix[0:n*trainFrac,2:]
    trainTargets = df.ix[0:n*trainFrac,0]
    trainOU = df.ix[0:n*trainFrac,1]
    
    testInputs = df.ix[n*trainFrac+1:,2:]
    testTargets = df.ix[n*trainFrac+1:,0] 
    testOU = df.ix[n*trainFrac+1:,1]
    

    clf = ensemble.RandomForestRegressor(n_estimators = 20)
    a = clf.fit(trainInputs,trainTargets)
    
    trainPred = clf.predict(trainInputs)
    testPred = clf.predict(testInputs)
    
    trainScore = Score(1,trainPred, trainTargets, trainOU)
    testScore = Score(1,testPred, testTargets, testOU)
    print (trainScore, testScore)