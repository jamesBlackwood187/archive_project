from sklearn.manifold import TSNE
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import xgboost
from sklearn.linear_model import Ridge, Lasso, OrthogonalMatchingPursuit
from sklearn.externals import joblib
import pandas as pd

def accuracy(predictions, actuals):
    return float((np.sign(predictions) == np.sign(actuals)).sum()) / float(len(predictions))

def load_data():
    my_data = np.genfromtxt('dataSet.csv', delimiter=',', skip_header = 1, max_rows=None)
    y = my_data[:,0]
    X = my_data[:,1:]
    return X,y

def cross_validate(X, y, model, window):
    '''
    Cross validates time series data using a shifting window where train data is
    always before test data
    '''
    in_sample_score = []
    out_sample_score = []
    print(model)
    for i in range(1, len(y)/window):
        train_index = np.arange(0, i*window)
        test_index = np.arange(i*window, (i+1)*window)
        y_train = 5*y.take(train_index)
        y_test = 5*y.take(test_index)
        X_train = X.take(train_index, axis=0)
        X_test = X.take(test_index, axis=0)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        in_sample_score.append(accuracy(y_train_pred,y_train))
        out_sample_score.append(accuracy(y_test_pred,y_test))
        print 'Window', i
        print 'Train Size:', len(X_train)
        print 'Test Size:', len(y_test)
        print 'in-sample score:', in_sample_score[-1]
        print 'out-sample score:', out_sample_score[-1]
        print '---'
    print 'Average in-sample cross-val score:', np.mean(in_sample_score)
    print 'Average out-of-sample cross-val score:', np.mean(out_sample_score)
    return model, np.mean(in_sample_score), np.mean(out_sample_score)



if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set
    X, y = load_data()

    X_1 = X[:3500]
    y_1 = y[:3500]

    X_blend = X[3500:4200]
    y_blend = y[3500:4200]

    X_holdout = X[4200:]
    y_holdout = y[4200:]


    

    clfs = [
              LinearRegression()
             ,SVR(kernel='linear',C=100, max_iter=1000000)
             ,xgboost.XGBRegressor(max_depth=5, learning_rate=0.001,n_estimators=800)
             ,xgboost.XGBRegressor(max_depth=2, learning_rate=0.001,n_estimators=1500)
             ,xgboost.XGBRegressor(max_depth=2, learning_rate=0.01,n_estimators=3000)
             ,Ridge(alpha=0.5)
             ,Lasso(alpha=0.0001)
             ,OrthogonalMatchingPursuit()
            ]
    
    predsDF = pd.DataFrame()
    holdoutDF = pd.DataFrame()
    holdoutScores = np.array([])


    for clf in clfs:
        mod, inScore, outScore = cross_validate(X_1,y_1, clf,len(y)/7)
        y_holdout_predict = mod.predict(X_holdout)
        y_blend_predict = mod.predict(X_blend)
        holdout_accuracy= accuracy(y_holdout_predict,y_holdout)
        print 'Holdout accuracy:', holdout_accuracy
        print '--------'
        print '--------'
        print '--------'
        predsDF = pd.concat([predsDF,pd.DataFrame(y_blend_predict)],axis = 1)
        holdoutDF = pd.concat([holdoutDF,pd.DataFrame(y_holdout_predict)],axis = 1)
        holdoutScores = np.append(holdoutScores,[holdout_accuracy])

    y_blend = np.sign(y_blend)

    clfBlender = LogisticRegression()
    clfBlender.fit(predsDF, y_blend)
    y_holdout_blendpreds = clfBlender.predict(holdoutDF)
 
    print 'Blended Holdout Score:', accuracy(y_holdout_blendpreds, y_holdout)
   
