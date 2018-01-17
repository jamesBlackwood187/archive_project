import os
import pandas as pd
import numpy as np
from datetime import datetime

def ReadTS(filename):
    return pd.read_csv(filename,index_col = 0)

if __name__ == "__main__":
    spy = ReadTS('spy_ts.csv')
    vix = ReadTS('vix_ts.csv')

    spyReturns = np.log(spy['Close'] / spy['Close'].shift(1))

    spyTrailingSD = pd.rolling_std(spyReturns, window = 5) * np.sqrt(252)

    