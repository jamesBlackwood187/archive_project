import os
import pandas as pd
import numpy as np
from datetime import datetime

def ReadTS(filename):
    return pd.read_csv(filename,index_col = 0)

def SetReturnTarget(df, daysInFuture = 15):
    df['RetTarget'] = df['Close'].shift(-daysInFuture)/ df['Close'] - 1
    return df

def SetQuantileTarget(df, nquantiles = 4):
    df['y'] = pd.qcut(df['RetTarget'],4, labels = range(nquantiles))
    return df

def TrailingReturns(lag=0):
    colName = "TrailCumReturn"+str(lag)
    df[colName]  = df['Close']/df['Close'].shift(lag) - 1
    return df


def LagReturns(lag=0):
    colName = "return"+str(lag)
    retSeries  = df['Close']/df['Close'].shift(1) - 1
    df[colName] = retSeries.shift(lag)
    return df

def EMAPriceRelative(lag = 0):
    colName = "emaRelLag"+str(lag)
    lagSeries =  df['ema_15'] / df['Close'] - 1
    df[colName] = lagSeries.shift(lag)
    return df



if __name__ == "__main__":
    df = ReadTS('GDX_ts.csv')

    df = SetReturnTarget(df)
    df = SetQuantileTarget(df)

    for i in range(5):
        df = TrailingReturns(i)
        df = LagReturns(i)
        df = EMAPriceRelative(i)
