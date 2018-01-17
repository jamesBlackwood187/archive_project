import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def ReadTS(filename):
    return pd.read_csv(filename,index_col = 0)

def MapProbs(ls):
    ret_arr = np.zeros(len(ls))
    for i,l in enumerate(ls):
        v,ind = max( (v, i) for i, v in enumerate(l) )
        if v > 0.4:
            if ind == 0:
                ret_arr[i] = -5
            elif ind == 1:
                ret_arr[i] = -1
            elif ind == 2:
                ret_arr[i] = 1
            elif ind == 3:
                ret_arr[i] = 5
            else:
                ret_arr[i] = 0
        else:
            ret_arr[i] = 0
    return ret_arr

def Prof(action, ret):
    action = MapProbs(action)
    ret_series = np.multiply(action,ret)  / 10
    return np.sum(ret_series) , np.mean(ret_series) / np.std(ret_series) * np.sqrt(30)
  


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
        in_sample_act = model.predict(X_train)
        out_sample_act = model.predict(X_test)
        #out_sample_probs = model.predict_proba(X_test)
        meta_features = np.append(meta_features, out_sample_act)
        in_sample_score.append(model.score(X_train, y_train))
        out_sample_score.append(model.score(X_test, y_test))
        print ('Window', i)
        print ('in-sample score', in_sample_score[-1])
        print ('out-sample score:', out_sample_score[-1])
        print ('---')
    print (np.mean(in_sample_score), np.mean(out_sample_score))
    return meta_features,model, in_sample_score, out_sample_score, out_sample_act

if __name__ == "__main__":
    df = ReadTS('SPY_ts.csv')
    print(len(df))
    df['Ret'] = df['Close'].shift(-1)/ df['Close'] - 1
    df['Vol'] = pd.rolling_std(df['Ret'], window = 5) * np.sqrt(250)
    

    df['dVol'] = df['Vol'].shift(-6) - df['Vol']
    num_classes = 1
    df['y'] = df['dVol']

    df['vol1']  = df['Vol'].shift(1)
    df['vol2']  = df['Vol'].shift(2)
    df['vol3']  = df['Vol'].shift(3)
    df['vol4']  = df['Vol'].shift(4)
    df['vol5']  = df['Vol'].shift(5)
    df['r0']  = df['Close']/df['Close'].shift(1) - 1
    df['r1']  = df['Close']/df['Close'].shift(2) - 1
    df['r2']  = df['Close']/df['Close'].shift(3) - 1
    df['r3']  = df['Close']/df['Close'].shift(4) - 1

   
    df['fet0'] =  df['ema_15'] / df['Close'] - 1
    df['fet1'] = df['fet0'].shift(1)
    df['fet2'] = df['fet0'].shift(2)
    df['fet3'] = df['fet0'].shift(3)
    df['fet4'] = df['fet0'].shift(4)
    df['fet5'] = df['fet0'].shift(5)
    df['fet6'] = df['fet0'].shift(6)
    df['fet7'] = df['fet0'].shift(7)
    df['fet8'] = df['fet0'] - df['fet1']
    df['fet9'] = df['fet1'] - df['fet2']
    df['fet10'] = df['fet2'] - df['fet3']
    df['fet11'] = df['fet3'] - df['fet4']
    df['fet12'] = np.sign(df['fet0'])
    df['fet13'] = np.sign(df['fet0']) - np.sign(df['fet1'])
    df['fet14'] = df['fet8'] - df['fet9']
    df['fet15'] = df['fet9'] - df['fet10']

    df['im1'] = np.sign(df['fet1']) 
    df['im2'] = np.sign(df['fet2']) 
    df['im3'] = np.sign(df['fet3'])
    df['im4'] = np.sign(df['fet4']) 
    df['im5'] = np.sign(df['fet5']) 
    df['im6'] = np.sign(df['fet6'])
    df['im7'] = np.sign(df['fet7'])

    col_list = ['im1','im2','im3','im4','im5','im6','im7']
    df['fet16'] = df[col_list].sum(axis=1)

    df = df.round(2)
    df_full = df
    windoo = 10
    df = df.dropna()

    holdoutSize = 10
    holdout = df[len(df)-holdoutSize:len(df)]
    df_a = df.head(len(df)-holdoutSize)
    met = df_a.tail(len(df) - windoo)
    y_meta = met['y']

    rets = df_a['Ret']
    X = df_a[['fet0','fet1', 'fet2', 'r0', 'r1','r2' , 'r3'
            , 'fet3','fet4', 'fet5'
            , 'fet6', 'fet7', 'fet8', 'fet9'
            , 'fet10', 'fet11', 'fet12', 'fet13','fet14', 'fet15'
            , 'fet16', 'vol1', 'vol2', 'vol3', 'vol4','vol5']]
    y = df_a['y']

    clfa = RandomForestClassifier(n_estimators=500, max_depth = 4, n_jobs = -1, random_state = 100, criterion = 'gini')
    clfb = DecisionTreeClassifier(max_depth = 20)
    clfc = XGBClassifier(max_depth=6, gamma = 3, learning_rate=0.1, n_estimators=100,  objective='multiclass:softmax')

    #pick uncorrelated ishh
    clfs = [
            Ridge(alpha = 3)
        ]

    l1_features = np.array([])
    hold_preds = np.array([])

    X_h = holdout[['fet0','fet1', 'fet2', 'r0', 'r1','r2' , 'r3'
            , 'fet3','fet4', 'fet5'
            , 'fet6', 'fet7', 'fet8', 'fet9'
            , 'fet10', 'fet11', 'fet12', 'fet13','fet14', 'fet15'
            , 'fet16', 'vol1', 'vol2', 'vol3', 'vol4','vol5']]

    y_h = holdout['y']

    for i,mod in enumerate(clfs):
        print(mod)
        m, mod, b, c, d = cross_validate(X, y, mod, rets, windoo)
        preds = mod.predict(X_h)
        #probs = mod.predict_proba(X_h)
        print('Holdout', mod.score(X_h, y_h))
        l1_features = np.append(l1_features,m)
        #hold_preds  = np.append(hold_preds, probs)
        hold_predsDF = pd.DataFrame(preds, index=holdout.index)
        file = str(mod)[:3] + ".csv"
        hold_predsDF.to_csv(file)
    l1_features = np.resize(l1_features, (len(df) - windoo, (i+1 )*num_classes))
    hold_meta_features = np.resize(hold_preds, (holdoutSize, (i+1) *num_classes ))

    #l1DF = pd.DataFrame(l1_features, index = met.index)
#    #l1DF.to_csv("level1.csv")
#
#    #level2a = LogisticRegression() 
#    #level2a.fit(l1_features, y_meta)   
#
#    #level2b = LogisticRegression()
#    #level2b.fit(l1_features, y_meta)
#
#    #level2c = SVC(probability = True)
#    #level2c.fit(l1_features, y_meta)
#
##---#--------------------------------------------------------------
#    #
#    #hold2a = level2a.predict_proba(hold_meta_features)
#    #hold2b = level2b.predict_proba(hold_meta_features)
#    #hold2c = level2c.predict_proba(hold_meta_features)
#
#    #hold = (0.3*hold2a + 0.7*hold2b + 0*hold2c )
#
#    #hold2aClass = level2a.predict(hold_meta_features)
#    #hold2bClass = level2b.predict(hold_meta_features)
#    #hold2cClass = level2c.predict(hold_meta_features)
#    #
#
#
#    ##holdout['ayy'] = pd.Series(hold_meta_features, index = X_h.index)
#    ##holdout['oobs'] = pd.Series(hold, index = X_h.index)
#
#    #h_preds = pd.DataFrame(hold,index = X_h.index)
#    #h_preds.to_csv('gen_preds.csv')
#
#    #print('Meta-hold-a', accuracy_score(hold2aClass, y_h))
#    #print('Meta-hold-b', accuracy_score(hold2bClass, y_h))
#    #print('Meta-hold-c', accuracy_score(hold2cClass, y_h))
#    ##print('Meta-hold-Av', accuracy_score(hold, y_h))
#    #holdout.to_csv('oobs.csv')
#
#
    #np.savetxt("foo.csv", hold_meta_features, delimiter=",")

    X_curr= df_full[['fet0','fet1', 'fet2', 'r0', 'r1','r2' , 'r3'
            , 'fet3','fet4', 'fet5'
            , 'fet6', 'fet7', 'fet8', 'fet9'
            , 'fet10', 'fet11', 'fet12', 'fet13','fet14', 'fet15'
            , 'fet16','vol1', 'vol2', 'vol3', 'vol4','vol5']].tail(5)

    X_full_layer_1 = pd.concat([X, X_h])
    y_full_layer_1 = pd.concat([y,y_h])

    currPreds = np.array([])  
    for i,model in enumerate(clfs):
        model.fit(X_full_layer_1, y_full_layer_1)
        predsa = model.predict(X_curr)
        currPreds  = np.append(currPreds, predsa)
    
    curr_features = np.resize(currPreds, (5, (i+1 ) * num_classes))

    curra = model.predict(X_curr)



    #holdout['ayy'] = pd.Series(hold_meta_features, index = X_h.index)
    #holdout['oobs'] = pd.Series(hold, index = X_h.index)

    curr_preds = pd.DataFrame(curra, index = X_curr.index)

