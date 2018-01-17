import re
import pandas as pd
import datetime
import requests
import numpy as np
from matplotlib.finance import candlestick2_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import shutil
import os

def get_data_for_symbol(symbol, t1, t2):
	base_url = 'https://query1.finance.yahoo.com/v8/finance/chart/'+symbol+'?symbol=SPY&period1='+str(t1)+'&period2='+str(t2)+'&interval=5m&includePrePost=false&events=div|split|earn&corsDomain=finance.yahoo.com'
	
	page = requests.get(base_url)
	data = page.json()
	ohlcv_key = ['open', 'high', 'low', 'close', 'volume']
	p_data = data['chart']['result'][0]['indicators']['quote'][0]
	ohlcv_data = [p_data[key] for key in ohlcv_key]

	timestamps = data['chart']['result'][0]['timestamp']
	fin_data = zip(timestamps, ohlcv_data[0], ohlcv_data[1], ohlcv_data[2], ohlcv_data[3], ohlcv_data[4])
	fin_data_df = pd.DataFrame(fin_data, columns = ['timestamp','open', 'high', 'low', 'close', 'volume'])
	return fin_data_df.set_index('timestamp')

if __name__ == '__main__':
	