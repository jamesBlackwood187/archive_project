
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
	fin_data = list(zip(timestamps, ohlcv_data[0], ohlcv_data[1], ohlcv_data[2], ohlcv_data[3], ohlcv_data[4]))
	fin_data_df = pd.DataFrame(fin_data, columns = ['timestamp','open', 'high', 'low', 'close', 'volume'])
	return fin_data_df.set_index('timestamp')

def get_price_subset(curr_time, ohlcv_data, lookback = 120):
	curr_time_ind = ohlcv_data.index.get_loc(curr_time)
	if curr_time_ind-lookback < 0:
		return 
	else:	
		lookback_time = int(ohlcv_data.iloc[curr_time_ind-lookback].name)
		df_subset = ohlcv_data.loc[lookback_time:curr_time]
	return df_subset	

def plot_curr_time(curr_time, quotes, slices=40):
	if quotes is None:
		return
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	candlestick2_ohlc(ax1,quotes['open'],quotes['high'],quotes['low'],quotes['close'],width=0.6)
	
	vol_prof = construct_vol_profile(curr_time, quotes, slices = 40)
	prices = [x[0] for x in vol_prof]
	vols = [x[1] for x in vol_prof]

	oldMin = min(vols)
	oldMax = 5000000
	newMin = 0
	newMax = 90
	vol_scale = [(((x - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin for x in vols]
	t = np.arange(slices)
	ax2.barh(t, vol_scale, alpha = 0.5)
	ax1.get_yaxis().set_visible(False)
	ax2.get_yaxis().set_visible(False)
	ax1.get_xaxis().set_visible(False)
	ax2.get_xaxis().set_visible(False)

	fig.savefig('./images/'+str(curr_time)+'.png', bbox_inches = 'tight')
	plt.close(fig)
	return

def construct_vol_profile(timestamp,df_subset, slices = 60):
	columns = ['v'+str(i) for i in range(slices)]
	volume_vec = np.zeros(slices+1)
	if df_subset is None:
		volume_vec = [np.nan for x in volume_vec]
		return dict(zip(columns, volume_vec))
	df_subset = df_subset.round(2)
	g_high = max(df_subset['high'])
	g_low  = min(df_subset['low'])
	partition = np.linspace(g_low, g_high, slices)

	def candle_vol_apply(row):
		GRAN = 0.01
		bin_inds = np.digitize([row['low'], row['high']], bins=partition)
		steps = (row['high'] - row['low']) / GRAN
		if steps == 0:
			return (bin_inds, row['volume'])
		else:
			return (bin_inds, row['volume'] / steps)

	vol_sets = df_subset.apply(candle_vol_apply, axis = 1)

	for elem in vol_sets:
		inds, vol = elem
		for x in range(inds[0]-1, inds[1]+1):
			volume_vec[x] += vol

	return list(zip(partition,volume_vec))


def add_mkt_profile_info(full_df,timestamp):
	df_sub = get_price_subset(timestamp,full_df)
	volume_vec = construct_vol_profile(timestamp, df_sub)
	print(volume_vec)
	df_row = pd.DataFrame(index = [timestamp], data = volume_vec)
	return df_row
	

if __name__ == '__main__':
	r = get_data_for_symbol('SPY',1506816000, 1513446685 )
	r = r.dropna()
	ticks_in_future = 8
	r['RetTarget'] = r['close'].shift(-ticks_in_future)/ r['close'] - 1 

	r.dropna()
	if os.path.exists('/images'):	
		shutil.rmtree('./images')
	
	try:
		os.makedirs('./images')
	except OSError:
		pass

	target_vec = np.zeros(len(r))
	label = np.zeros(len(r))
	ts = np.empty(len(r), dtype = "U32")
	train_test = np.empty(len(r), dtype = "U8")
	i = 0
	print(len(r))
	for ind, row in r.iterrows():
		print(ind)
		target_vec[i] = row['RetTarget']
		label[i] = (row['RetTarget']) > 0
		img_string = str(ind) + '.png'
		ts[i] = img_string
		#rint(add_mkt_profile_info(r, ind))
		if i < 2700:
			train_test[i] = 'train'
		else:
			train_test[i] = 'test'
		i += 1
		p_subset = get_price_subset(ind,r)
		plot_curr_time(ind, p_subset)

	image_info = pd.DataFrame({"file" : ts, "label" : label, "value" : target_vec, "train_test": train_test})
	image_info = image_info[150:]
	image_info.to_csv("img_info.csv", index = False)

#	for i,timestamp in enumerate(r.index):
#		row = add_mkt_profile_info(r, timestamp)
#		if i == 0:
#			profile_df = row
#		else:
#			profile_df = profile_df.append(row)
#			print(len(profile_df))
#	h = pd.merge(r, profile_df, left_index = True, right_index = True)
#	h['omc'] = h['open'] / h['close'] - 1
#	h['hml'] = h['high'] / h['low'] - 1
#	ticks_in_future = 12
#	h['RetTarget'] = h['close'].shift(-ticks_in_future)/ h['close'] - 1 
#	h.dropna()
#	h.to_csv('base.csv')