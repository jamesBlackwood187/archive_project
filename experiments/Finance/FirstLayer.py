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

def ReadTS(filename):
    return pd.read_csv(filename,index_col = 0)

def MapProbs(ls):
    ret_arr = np.zeros(len(ls))
    for i,l in enumerate(ls):
        v,ind = max( (v, i) for i, v in enumerate(l) )
        if v > 0:
            if ind == 0:
                ret_arr[i] = -10
            elif ind == 1:
                ret_arr[i] = 0
            elif ind == 2:
                ret_arr[i] = 0
            elif ind == 3:
                ret_arr[i] = 10
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

if __name__ == "__main__":
    df = ReadTS('USO_ts.csv')
    print(len(df))
    df['Ret'] = df['Close'].shift(-20)/ df['Close'] - 1
    df['Vol'] = pd.rolling_std(df['Ret'], window = 5)
    

    df['y'] = pd.qcut(df['Ret'], 4, labels = [-2,-1,1,2])

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
    df['fet16'] = np.sign(df['fet0'] - df['fet1'])

    df['im1'] = np.sign(df['fet1']) 
    df['im2'] = np.sign(df['fet2']) 
    df['im3'] = np.sign(df['fet3'])
    df['im4'] = np.sign(df['fet4']) 
    df['im5'] = np.sign(df['fet5']) 
    df['im6'] = np.sign(df['fet6'])
    df['im7'] = np.sign(df['fet7'])

    col_list = ['im1','im2','im3','im4','im5','im6','im7']
    df['fet16'] = df[col_list].sum(axis=1)

    df_full = df
    windoo = 100
    df = df.dropna()

    holdoutSize = 100
    holdout = df[len(df)-holdoutSize:len(df)]
    df_a = df.head(len(df)-holdoutSize)
    met = df_a.tail(len(df) - windoo)
    y_meta = met['y']

    rets = df_a['Ret']
    X = df_a[['fet12', 'fet13', 'fet16' ]]
    y = df_a['y']

    clfa = RandomForestClassifier(n_estimators=500, max_depth = 4, n_jobs = -1, random_state = 100, criterion = 'gini')
    clfb = DecisionTreeClassifier(max_depth = 20)

    #pick uncorrelated ishh
    clfs = [
           XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=100) 
         # , ExtraTreesClassifier(n_estimators=500, max_depth = 4, n_jobs = -1, random_state = 69, criterion = 'gini', max_features = 0.2)
         # , SVC( C= 10000
         #        , kernel = 'rbf'
         #        , gamma = 0.001
         #        , probability = True)
         # , LogisticRegression()
        ]

    l1_features = np.array([])
    hold_preds = np.array([])


    X_h = holdout[[ 'fet12', 'fet13', 'fet16' ]]

    y_h = holdout['y']

    for i,mod in enumerate(clfs):
        print(mod)
        m, mod, b, c, d = cross_validate(X, y, mod, rets, windoo)
        preds = mod.predict_proba(X_h)
        print('Holdout', Prof(preds,holdout['Ret']))
        l1_features = np.append(l1_features,m)
        hold_preds  = np.append(hold_preds, preds)
        file = str(mod)[:3] + ".csv"
        hold_predsDF = pd.DataFrame(preds, index=holdout.index)
        hold_predsDF.to_csv(file)
    l1_features = np.resize(l1_features, (len(df) - windoo, (i+1 )* 4))
    hold_meta_features = np.resize(hold_preds, (holdoutSize, (i+1 )* 4))



    level2a = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100) 
    level2a.fit(l1_features, y_meta)   

    level2b = LogisticRegression()
    level2b.fit(l1_features, y_meta)

    level2c = ExtraTreesClassifier(n_estimators=500, max_depth = 8, n_jobs = -1, random_state = 69, criterion = 'gini')
    level2c.fit(l1_features, y_meta)

#-----------------------------------------------------------------
    
    hold2a = level2a.predict_proba(hold_meta_features)
    hold2b = level2b.predict_proba(hold_meta_features)
    hold2c = level2c.predict_proba(hold_meta_features)

    hold = (0.45*hold2a + 0.1*hold2b + 0.45*hold2c )

    #holdout['ayy'] = pd.Series(hold_meta_features, index = X_h.index)
    #holdout['oobs'] = pd.Series(hold, index = X_h.index)

    h_preds = pd.DataFrame(hold,index = X_h.index)
    h_preds.to_csv('gen_preds.csv')

    print('Meta-hold-a', Prof(hold2a, holdout['Ret']))
    print('Meta-hold-b', Prof(hold2b, holdout['Ret']))
    print('Meta-hold-c', Prof(hold2c, holdout['Ret']))
    print('Meta-hold-Av', Prof(hold, holdout['Ret']))
    holdout.to_csv('oobs.csv')


    np.savetxt("foo.csv", hold_meta_features, delimiter=",")



    #retrain "production model"
    X_curr= df_full[['fet12', 'fet13', 'fet16' ]].tail(5)

    X_full_layer_1 = pd.concat([X, X_h])
    y_full_layer_1 = pd.concat([y,y_h])

    currPreds = np.array([])  
    for i,model in enumerate(clfs):
        model.fit(X_full_layer_1, y_full_layer_1)
        predsa = model.predict_proba(X_curr)
        currPreds  = np.append(currPreds, predsa)
    
    curr_features = np.resize(currPreds, (5, (i+1 ) * 4))

    curra = level2a.predict_proba(curr_features)
    currb = level2b.predict_proba(curr_features)
    currc = level2c.predict_proba(curr_features)


    currFPreds = (0.45*curra + 0.10*currb + 0.45*currc )

    #holdout['ayy'] = pd.Series(hold_meta_features, index = X_h.index)
    #holdout['oobs'] = pd.Series(hold, index = X_h.index)

    curr_preds = pd.DataFrame(currFPreds,index = X_curr.index)

    aaa = mod.predict_proba(X_curr)