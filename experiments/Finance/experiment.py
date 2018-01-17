import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def cross_validate(X, y, model, rets, window):
    '''
    Cross validates time series data using a shifting window where train data is
    always before test data
    '''
    in_sample_score = []
    out_sample_score = []
    meta_features = np.array([])
    for i in range(1, len(y)//window):
        train_index = np.arange(0, i*window)
        test_index = np.arange(i*window, (i+1)*window)
        y_train = y.take(train_index)
        in_rets = rets.take(train_index)
        y_test = y.take(test_index)
        out_rets = rets.take(test_index)
        X_train = X.take(train_index)
        X_test = X.take(test_index)
        model.fit(X_train, y_train)
        in_sample_act = model.predict_proba(X_train)
        out_sample_act = model.predict_proba(X_test)
        meta_features = np.append(meta_features, out_sample_act)
        in_sample_score.append(Prof(in_sample_act, in_rets))
        out_sample_score.append(Prof(out_sample_act, out_rets))
        print ('Window', i)
        print ('in-sample score', in_sample_score[-1])
        print ('out-sample score:', out_sample_score[-1])
        print ('---')
    print (np.mean(in_sample_score), np.mean(out_sample_score))
    return meta_features,model, in_sample_score, out_sample_score, out_sample_act

df = pd.read_csv('base.csv')