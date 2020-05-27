import csv
import json
import talib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# declare function fs()
def fs(stock_data): # stock_data = Chart Data

	minmax = MinMaxScaler()

	a = stock_data.close
	b = stock_data.open
	c = stock_data.date
	outcome = []

	# make outcome data. if open price < close price, outcome = 1 else, outcome = 0
	for index in range(len(stock_data.close)):
		if int(a[index]) - int(b[index]) > 0:
			outcome.append(1)
		else :
			outcome.append(0)

	dict = {'outcome':outcome, 'open':stock_data.open, 'high':stock_data.high, 'low':stock_data.low, 'close':stock_data.close, 'volume':stock_data.volume}

	# make DataFrame that includes Training Data
	df = pd.DataFrame(dict)

	# 5 predict set : 5,10,15,30,60 days
	predict_number = [5,10,15,30,60]

	# not in time
	df['bop'] = talib.BOP(df.open, df.high, df.low, df.close)
	#df['ht'] = talib.HT_TRENDLINE(df.close) # all data is NaN
	df['trsrange'] = talib.TRANGE(df.high,df.low,df.close)
	df['ad'] = talib.AD(df.high,df.low,df.close,df.volume)
	df['obv'] = talib.OBV(df.close,df.volume)
	df['sar'] = talib.SAR(df.high, df.low)

	# need predict time
	for predict_len in predict_number:
		# volilatily indicator
		df['atr' + str(predict_len)] = talib.ATR(df.high,df.low,df.close,timeperiod=predict_len)
		df['natr' + str(predict_len)] = talib.NATR(df.high,df.low,df.close,timeperiod=predict_len)
		df['adosc' + str(predict_len)] = talib.ADOSC(df.high,df.low,df.close,df.volume,fastperiod=predict_len,slowperiod=predict_len+5)

		# technical indicator
		b_upper, b_middle, b_lower = talib.BBANDS(df.close,timeperiod=predict_len)
		df['b_upper' + str(predict_len)] = b_upper
		df['b_middle' + str(predict_len)] = b_middle
		df['b_lower' + str(predict_len)] = b_lower

		df['ema' + str(predict_len)] = talib.EMA(df.close,timeperiod=predict_len)
		df['kama' + str(predict_len)] = talib.KAMA(df.close,timeperiod=predict_len)
		df['ma' + str(predict_len)] = talib.MA(df.close,timeperiod=predict_len) 
		df['dmi' + str(predict_len)] = talib.DX(df.high,df.low,df.close,timeperiod=predict_len)  
		df['sma' + str(predict_len)] = talib.SMA(df.close, timeperiod=predict_len) 
		df['wma' + str(predict_len)] = talib.WMA(df.close,timeperiod=predict_len)

		# momentum indicator
		df['adx' + str(predict_len)] = talib.ADX(df.high,df.low,df.close,timeperiod=predict_len) 
		df['adxr' + str(predict_len)] = talib.ADXR(df.high,df.low,df.close,timeperiod=predict_len) 
		df['apo' + str(predict_len)] = talib.APO(df.close, fastperiod=predict_len,slowperiod=predict_len+5)

		aroondown, aroonup = talib.AROON(df.high,df.low, timeperiod=predict_len)
		df['aroondown' + str(predict_len)] = aroondown
		df['aroonup' + str(predict_len)] = aroonup

		df['aroonosc' + str(predict_len)] = talib.AROONOSC(df.high,df.low, timeperiod=predict_len)
		df['rsi' + str(predict_len)] = talib.RSI(df.close,timeperiod=predict_len)                             #Relative Strength Index

		slowk, slowd = talib.STOCH(df.high,df.low,df.close, fastk_period=predict_len, slowk_period=predict_len+5, slowk_matype=0, slowd_period=predict_len+5, slowd_matype=0)             #Stochastic
		df['slowk' + str(predict_len)] = slowk
		df['slowd' + str(predict_len)] = slowd

		df['willi' + str(predict_len)] = talib.WILLR(df.high,df.low,df.close,timeperiod=predict_len)                  #Williams                                   
		df['mom' + str(predict_len)] = talib.MOM(df.close,timeperiod=predict_len)                               #momentum

		macd, macdsignal, macdhist = talib.MACD(df.close,fastperiod=predict_len, slowperiod=predict_len+5,signalperiod=4)                                    #moving average convergence/divergence
		df['macd' + str(predict_len)] = macd
		df['macdsignal' + str(predict_len)] = macdsignal
		df['macdhist' + str(predict_len)] = macdhist

		df['cci' + str(predict_len)] = talib.CCI(df.high,df.low,df.close,timeperiod=predict_len)
		df['dx' + str(predict_len)] = talib.DX(df.high,df.low,df.close,timeperiod=predict_len)
		df['cmo' + str(predict_len)] = talib.CMO(df.close,timeperiod=predict_len)
		df['ppo' + str(predict_len)] = talib.PPO(df.close, fastperiod=predict_len, slowperiod=predict_len+5)
		df['roc' + str(predict_len)] = talib.ROC(df.close, timeperiod=predict_len)
		df['rocp' + str(predict_len)] = talib.ROCP(df.close, timeperiod=predict_len)
		df['rocr' + str(predict_len)] = talib.ROCR(df.close, timeperiod=predict_len)
		df['ultosc' + str(predict_len)] = talib.ULTOSC(df.high,df.low,df.close, timeperiod1=predict_len, timeperiod2=predict_len+5, timeperiod3=predict_len+10)

	# clear 'NaN' row
	df = df.dropna()

	# normalize
	x = df.values.astype(float)
	x_scaled = minmax.fit_transform(x)
	df = pd.DataFrame(x_scaled, columns=df.columns)

	# Data => ratio
	
	# Feature Selection
	array = df.values
	Y = array[:,0:1]
	X = array[:,1:181]

	model = LogisticRegression()
	rfe = RFE(model, 3)
	fit = rfe.fit(X, Y)

	# extract feature
	feat_data = pd.DataFrame() 	# feat_data : Selected Feature DataFrame
	prefer_rank = 1			# prefer_rank : Use to choose Top 15 rank Data
	num_feat = 0			# num_feat : number of chosen features

	while True: # infinity loop
		for i in range(0, 179): # number of features = 180 (0~179)
			if fit.ranking_[i] == prefer_rank: # ranking = prefer_rank, choose that
				col_name = list(df)[i] # col_name : column name
				feat_data[col_name] = df[col_name] # input to feat_data
				num_feat += 1

			if num_feat == 15: # have chosen 15 features
				break

		if(num_feat < 15): # if not 15 features,
			prefer_rank += 1 # choose next rank
		else: # if have 15 features,
			break

	return feat_data
