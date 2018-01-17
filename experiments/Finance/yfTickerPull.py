import pandas_datareader.data as web
import os
import pandas as pd
import numpy as np
from datetime import datetime
import talib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.svm import SVC
import xgboost


def cross_validate(X, y, model, rets, window):
    '''
    Cross validates time series data using a shifting window where train data is
    always before test data
    '''
    in_sample_score = []
    out_sample_score = []
    oob_metaFeatures = np.array([])
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
        in_sample_act = model.predict(X_train)
        out_sample_act = model.predict(X_test)
        print(out_sample_act, y_test)
        in_sample_score.append(Prof(in_sample_act, in_rets))
        out_sample_score.append(Prof(out_sample_act, out_rets))
        print ('Window', i)
        print ('in-sample score', in_sample_score[-1])
        print ('out-sample score:', out_sample_score[-1])
        print ('---')
    return model, np.mean(in_sample_score), np.mean(out_sample_score), out_sample_act

if __name__ == "__main__":
    startDate = datetime(1990,1,1)
    endDate   = datetime(2017,3,30)
    GDX = web.DataReader('USO',  'yahoo', startDate, endDate)
    
    gdx_inputs = {
        'open'  : np.reshape(GDX.as_matrix(columns = ['Open']),-1),
        'high'  : np.reshape(GDX.as_matrix(columns = ['High']),-1),
        'low'   : np.reshape(GDX.as_matrix(columns = ['Low']),-1),
        'close' : np.reshape(GDX.as_matrix(columns = ['Close']),-1),
        'volume': np.reshape(GDX.as_matrix(columns = ['Volume']),-1),
    }
    ema = talib.EMA(gdx_inputs['close'], 15)
    dayRange = (gdx_inputs['high'] - gdx_inputs['close']) / gdx_inputs['close']
    df = pd.DataFrame(data={'Close': gdx_inputs['close'],
                            'Open': gdx_inputs['open'],
                            'Low': gdx_inputs['close'],
                            'HML': gdx_inputs['high'] / gdx_inputs['open'] - gdx_inputs['low'] / gdx_inputs['open'] ,
                            'DayRange': dayRange,
                            'ema_15'  :ema},
                      index = GDX.index)

    df.to_csv('USO_ts.csv', sep = ',')
