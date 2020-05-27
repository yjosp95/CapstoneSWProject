import csv
import json
import talib
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#//////////////////////////////////////////////////////#
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#find and remove correlated features
#in order to reduce the feature space a bit
#so that the algorithm takes shorter
def separation(stock_data,num,num2):
	#test_data = stock_data.columns
	train_data = DataFrame(columns=stock_data.columns) #create dataframe
	test_data = DataFrame(columns=stock_data.columns)


	test_data.loc[200]=stock_data.values[5]

	data_size=len(stock_data)

	while num2>0:
		num2-=1
		data_size-=1
		test_data.loc[num2] = stock_data.values[data_size]

	while num>0:
		num-=1
		data_size-=1
		train_data.loc[num] = stock_data.values[data_size]

	test_data.sort_index(inplace=True)
	train_data.sort_index(inplace=True)

	return train_data, test_data

def correlation(dataset, threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

def forward(stock_data):
	data=stock_data
	numerics = ['int16', 'int32','int64','float16','float32','float64']
	numerical_vars=list(data.select_dtypes(include=numerics).columns)
	data=data[numerical_vars]

	
	# separate train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
		data.drop(labels=['outcome'], axis=1),
		data['outcome'],
		test_size=0.3,
		random_state=0)

	corr_features = correlation(X_train, 0.8) #use or not use
	
	#step backward feature selectio
	#Using 15 features with ROC_AUC scoring
	sfs1 = SFS(RandomForestClassifier(n_jobs=4),
			k_features=10,
			forward=True,
			floating=False,
			verbose=2,
			scoring='roc_auc',
			cv=3)
	sfs1 = sfs1.fit(np.array(X_train.fillna(0)), y_train)

	selected_feat=X_train.columns[list(sfs1.k_feature_idx_)]

	return stock_data[selected_feat]

def backward(stock_data):
	data=stock_data
	numerics = ['int16', 'int32','int64','float16','float32','float64']
	numerical_vars=list(data.select_dtypes(include=numerics).columns)
	data=data[numerical_vars]

	# separate train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
		data.drop(labels=['outcome'], axis=1),
		data['outcome'],
		test_size=0.3,
		random_state=0)
		
	corr_features = correlation(X_train, 0.8) #use or not use
	
	#step backward feature selectio
	#Using 15 features with ROC_AUC scoring
	sfs1 = SFS(RandomForestClassifier(n_jobs=4),
			k_features=10,
			forward=False,
			floating=False,
			verbose=2,
			scoring='roc_auc',
			cv=3)
	sfs1 = sfs1.fit(np.array(X_train.fillna(0)), y_train)
	
	selected_feat=X_train.columns[list(sfs1.k_feature_idx_)]

	return stock_data[selected_feat]
	
def fs(df):
	# Feature Selectiondesx-
	array = df.values
	Y = array[:,0]
	X = array[:,1:108]

	model = LogisticRegression(solver='lbfgs', max_iter=10000)
	#solver --> 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga' etc... // algorithm
	rfe = RFE(model, 15)
	fit = rfe.fit(X, Y)
	
	print("Num Features: %s" % (fit.n_features_))
	print("Selected Features: %s" % (fit.support_))
	print("Features Ranking: %s" % (fit.ranking_))

	# extract feature
	feat_data = pd.DataFrame() 	# feat_data : Selected Feature DataFrame.bcntx
	prefer_rank = 1			# prefer_rank : Use to choose Top 15 rank Data3x-
	num_feat = 0			# num_feat : number of chosen features

#//////////////////////////////////////////////////////////////////////////////////////////#
	while True: # infinity loop
		for i in range(0, 100): # number of features = 180 (0~179)
			if fit.ranking_[i] == prefer_rank: # ranking = prefer_rank, choose that
				col_name = list(df)[i+1] # col_name : column name
				feat_data[col_name] = df[col_name] # input to feat_data
				num_feat += 1

			if num_feat == 15: # have chosen 15 features
				break

		if(num_feat < 15): # if not 15 features,/
			prefer_rank += 1 # choose next rank
		else: # if have 15 features,
			break

	return feat_data

def preprocess(chart_data):
    prep_data = chart_data
    windows = [5,10,15,30,60]
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()
        prep_data['volume_ma{}'.format(window)] = (
            prep_data['volume'].rolling(window).mean())
    return prep_data

def build_training_data(prep_data):
    training_data = prep_data

    training_data['open_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'open_lastclose_ratio'] = \
        (training_data['open'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    training_data['high_close_ratio'] = \
        (training_data['high'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data['low_close_ratio'] = \
        (training_data['low'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'close_lastclose_ratio'] = \
        (training_data['close'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'volume_lastvolume_ratio'] = \
        (training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
        training_data['volume'][:-1]\
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    windows = [5,10,15,30,60]
    for window in windows:
        training_data['close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / \
            training_data['close_ma%d' % window]
        training_data['volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / \
            training_data['volume_ma%d' % window]	
    return training_data


def build_training_data2(final_data):
	training_data = final_data
	training_data['trsrange_ratio'] = np.zeros(len(training_data))
	training_data['trsrange_ratio'].iloc[1:] = \
		(training_data['trsrange'][1:].values - training_data['trsrange'][:-1].values) / \
			training_data['trsrange'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
	
	training_data['ad_ratio'] = np.zeros(len(training_data))
	training_data['ad_ratio'].iloc[1:] = \
		(training_data['ad'][1:].values - training_data['ad'][:-1].values) / \
			training_data['ad'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['obv_ratio'] = np.zeros(len(training_data))
	training_data['obv_ratio'].iloc[1:] = \
		(training_data['obv'][1:].values - training_data['obv'][:-1].values) / \
			training_data['obv'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['sar_ratio'] = np.zeros(len(training_data))
	training_data['sar_ratio'].iloc[1:] = \
		(training_data['sar'][1:].values - training_data['sar'][:-1].values) / \
			training_data['sar'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
	
	training_data['atr5_ratio'] = np.zeros(len(training_data))
	training_data['atr5_ratio'].iloc[1:] = \
		(training_data['atr5'][1:].values - training_data['atr5'][:-1].values) / \
			training_data['atr5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['natr5_ratio'] = np.zeros(len(training_data))
	training_data['natr5_ratio'].iloc[1:] = \
		(training_data['natr5'][1:].values - training_data['natr5'][:-1].values) / \
			training_data['natr5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
	
	training_data['adosc5_ratio'] = np.zeros(len(training_data))
	training_data['adosc5_ratio'].iloc[1:] = \
		(training_data['adosc5'][1:].values - training_data['adosc5'][:-1].values) / \
			training_data['adosc5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
	
	training_data['b_upper5_ratio'] = np.zeros(len(training_data))
	training_data['b_upper5_ratio'].iloc[1:] = \
		(training_data['b_upper5'][1:].values - training_data['b_upper5'][:-1].values) / \
			training_data['b_upper5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
	
	training_data['b_middle5_ratio'] = np.zeros(len(training_data))
	training_data['b_middle5_ratio'].iloc[1:] = \
		(training_data['b_middle5'][1:].values - training_data['b_middle5'][:-1].values) / \
			training_data['b_middle5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
		
	training_data['b_lower5_ratio'] = np.zeros(len(training_data))
	training_data['b_lower5_ratio'].iloc[1:] = \
		(training_data['b_lower5'][1:].values - training_data['b_lower5'][:-1].values) / \
			training_data['b_lower5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['ema5_ratio'] = np.zeros(len(training_data))
	training_data['ema5_ratio'].iloc[1:] = \
		(training_data['ema5'][1:].values - training_data['ema5'][:-1].values) / \
			training_data['ema5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
		
	training_data['kama5_ratio'] = np.zeros(len(training_data))
	training_data['kama5_ratio'].iloc[1:] = \
		(training_data['kama5'][1:].values - training_data['kama5'][:-1].values) / \
			training_data['kama5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
		
	training_data['ma5_ratio'] = np.zeros(len(training_data))
	training_data['ma5_ratio'].iloc[1:] = \
		(training_data['ma5'][1:].values - training_data['ma5'][:-1].values) / \
			training_data['ma5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['dmi5_ratio'] = np.zeros(len(training_data))
	training_data['dmi5_ratio'].iloc[1:] = \
		(training_data['dmi5'][1:].values - training_data['dmi5'][:-1].values) / \
			training_data['dmi5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['sma5_ratio'] = np.zeros(len(training_data))
	training_data['sma5_ratio'].iloc[1:] = \
		(training_data['sma5'][1:].values - training_data['sma5'][:-1].values) / \
			training_data['sma5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['wma5_ratio'] = np.zeros(len(training_data))
	training_data['wma5_ratio'].iloc[1:] = \
		(training_data['wma5'][1:].values - training_data['wma5'][:-1].values) / \
			training_data['wma5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['adx5_ratio'] = np.zeros(len(training_data))
	training_data['adx5_ratio'].iloc[1:] = \
		(training_data['adx5'][1:].values - training_data['adx5'][:-1].values) / \
			training_data['adx5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
	
	training_data['adxr5_ratio'] = np.zeros(len(training_data))
	training_data['adxr5_ratio'].iloc[1:] = \
		(training_data['adxr5'][1:].values - training_data['adxr5'][:-1].values) / \
			training_data['adxr5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['apo5_ratio'] = np.zeros(len(training_data))
	training_data['apo5_ratio'].iloc[1:] = \
		(training_data['apo5'][1:].values - training_data['apo5'][:-1].values) / \
			training_data['apo5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
		
	training_data['aroondown5_ratio'] = np.zeros(len(training_data))
	training_data['aroondown5_ratio'].iloc[1:] = \
		(training_data['aroondown5'][1:].values - training_data['aroondown5'][:-1].values) / \
			training_data['aroondown5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['aroonup5_ratio'] = np.zeros(len(training_data))
	training_data['aroonup5_ratio'].iloc[1:] = \
		(training_data['aroonup5'][1:].values - training_data['aroonup5'][:-1].values) / \
			training_data['aroonup5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
		
	training_data['aroonosc5_ratio'] = np.zeros(len(training_data))
	training_data['aroonosc5_ratio'].iloc[1:] = \
		(training_data['aroonosc5'][1:].values - training_data['aroonosc5'][:-1].values) / \
			training_data['aroonosc5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['rsi5_ratio'] = np.zeros(len(training_data))
	training_data['rsi5_ratio'].iloc[1:] = \
		(training_data['rsi5'][1:].values - training_data['rsi5'][:-1].values) / \
			training_data['rsi5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['slowk5_ratio'] = np.zeros(len(training_data))
	training_data['slowk5_ratio'].iloc[1:] = \
		(training_data['slowk5'][1:].values - training_data['slowk5'][:-1].values) / \
			training_data['slowk5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['slowd5_ratio'] = np.zeros(len(training_data))
	training_data['slowd5_ratio'].iloc[1:] = \
		(training_data['slowd5'][1:].values - training_data['slowd5'][:-1].values) / \
			training_data['slowd5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['willi5_ratio'] = np.zeros(len(training_data))
	training_data['willi5_ratio'].iloc[1:] = \
		(training_data['willi5'][1:].values - training_data['willi5'][:-1].values) / \
			training_data['willi5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['mom5_ratio'] = np.zeros(len(training_data))
	training_data['mom5_ratio'].iloc[1:] = \
		(training_data['mom5'][1:].values - training_data['mom5'][:-1].values) / \
			training_data['mom5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['macd5_ratio'] = np.zeros(len(training_data))
	training_data['macd5_ratio'].iloc[1:] = \
		(training_data['macd5'][1:].values - training_data['macd5'][:-1].values) / \
			training_data['macd5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['macdsignal5_ratio'] = np.zeros(len(training_data))
	training_data['macdsignal5_ratio'].iloc[1:] = \
		(training_data['macdsignal5'][1:].values - training_data['macdsignal5'][:-1].values) / \
			training_data['macdsignal5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['macdhist5_ratio'] = np.zeros(len(training_data))
	training_data['macdhist5_ratio'].iloc[1:] = \
		(training_data['macdhist5'][1:].values - training_data['macdhist5'][:-1].values) / \
			training_data['macdhist5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['cci5_ratio'] = np.zeros(len(training_data))
	training_data['cci5_ratio'].iloc[1:] = \
		(training_data['cci5'][1:].values - training_data['cci5'][:-1].values) / \
			training_data['cci5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['dx5_ratio'] = np.zeros(len(training_data))
	training_data['dx5_ratio'].iloc[1:] = \
		(training_data['dx5'][1:].values - training_data['dx5'][:-1].values) / \
			training_data['dx5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
	
	training_data['cmo5_ratio'] = np.zeros(len(training_data))
	training_data['cmo5_ratio'].iloc[1:] = \
		(training_data['cmo5'][1:].values - training_data['cmo5'][:-1].values) / \
			training_data['cmo5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['ppo5_ratio'] = np.zeros(len(training_data))
	training_data['ppo5_ratio'].iloc[1:] = \
		(training_data['ppo5'][1:].values - training_data['ppo5'][:-1].values) / \
			training_data['ppo5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['roc5_ratio'] = np.zeros(len(training_data))
	training_data['roc5_ratio'].iloc[1:] = \
		(training_data['roc5'][1:].values - training_data['roc5'][:-1].values) / \
			training_data['roc5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['rocp5_ratio'] = np.zeros(len(training_data))
	training_data['rocp5_ratio'].iloc[1:] = \
		(training_data['rocp5'][1:].values - training_data['rocp5'][:-1].values) / \
			training_data['rocp5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values
	
	training_data['rocr5_ratio'] = np.zeros(len(training_data))
	training_data['rocr5_ratio'].iloc[1:] = \
		(training_data['rocr5'][1:].values - training_data['rocr5'][:-1].values) / \
			training_data['rocr5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	training_data['ultosc5_ratio'] = np.zeros(len(training_data))
	training_data['ultosc5_ratio'].iloc[1:] = \
		(training_data['ultosc5'][1:].values - training_data['ultosc5'][:-1].values) / \
			training_data['ultosc5'][:-1]\
				.replace(to_replace=0, method='ffill')\
				.replace(to_replace=0, method='bfill').values

	return training_data

# declare function fs()m-
def load_talib(stock_data): # stock_data = Chart Data
	a = stock_data.close
	b = stock_data.open
	c = stock_data.date
	outcome = []

	# make outcome data. if open price < close price, outcome = 1 else, outcome = 0
	for index in range(a.index[0],len(stock_data.close)+a.index[0]):
		if int(a[index]) - int(b[index]) > 0:
			outcome.append(1)
		else :
			outcome.append(0)
	#outcome.append(np.nan)
	
	#dict = {'outcome':outcome, 'open':stock_data.open, 'high':stock_data.high, 'low':stock_data.low, 'close':stock_data.close, 'volume':stock_data.volume}
	dict = {'outcome':outcome,'open':stock_data.open, 'high':stock_data.high, 'low':stock_data.low, 'close':stock_data.close, 'volume':stock_data.volume,
	'close_ma5':stock_data.close_ma5, 'volume_ma5':stock_data.volume_ma5, 'close_ma10':stock_data.close_ma10, 'volume_ma10':stock_data.volume_ma10,
	'close_ma15':stock_data.close_ma15, 'volume_ma15':stock_data.volume_ma15, 'close_ma30':stock_data.close_ma30, 'volume_ma30':stock_data.volume_ma30,
	'close_ma60':stock_data.close_ma60, 'volume_ma60':stock_data.volume_ma60, 'open_lastclose_ratio':stock_data.open_lastclose_ratio, 'high_close_ratio':stock_data.high_close_ratio,
	'low_close_ratio':stock_data.low_close_ratio, 'close_lastclose_ratio':stock_data.close_lastclose_ratio, 'volume_lastvolume_ratio':stock_data.volume_lastvolume_ratio, 'close_ma5_ratio':stock_data.close_ma5_ratio,
	'volume_ma5_ratio':stock_data.volume_ma5_ratio, 'close_ma10_ratio':stock_data.close_ma10_ratio, 'volume_ma10_ratio':stock_data.volume_ma10_ratio, 'close_ma15_ratio':stock_data.close_ma15_ratio,
	'volume_ma15_ratio':stock_data.volume_ma15_ratio, 'close_ma30_ratio':stock_data.close_ma30_ratio, 'volume_ma30_ratio':stock_data.volume_ma30_ratio,'close_ma60_ratio':stock_data.close_ma60_ratio, 'volume_ma60_ratio':stock_data.volume_ma60_ratio}

	# make DataFrame that includes Training Data
	df = pd.DataFrame(dict)
	df=df.dropna(axis=0)

	# 5 predict set : 5,10,15,30,60 days
	predict_number = [5]

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
	return df

# normalization
def normalize(stock_data):
	minmax = MaxAbsScaler()  # -1 ~ 1
	#StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

	#normailze
	#////////////////////////////////////////////////////////
	x=stock_data.values.astype(float)
	x_scaled = minmax.fit_transform(x)
	df=pd.DataFrame(x_scaled, columns=stock_data.columns)
	return df

def ratio_data(stock_data):
	features_stock_data = ['outcome','open', 'high', 'low', 'close', 
		'volume', 'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio', 'close_ma10_ratio', 'volume_ma10_ratio','close_ma15_ratio', 'volume_ma15_ratio',
        'close_ma30_ratio', 'volume_ma30_ratio', 'close_ma60_ratio', 'volume_ma60_ratio','trsrange_ratio', 'ad_ratio',	
		'obv_ratio', 'sar_ratio', 'atr5_ratio', 'natr5_ratio', 'adosc5_ratio', 'b_upper5_ratio', 'b_middle5_ratio', 
		'b_lower5_ratio', 'ema5_ratio', 'kama5_ratio', 'ma5_ratio', 'dmi5_ratio', 'sma5_ratio', 'wma5_ratio', 'adx5_ratio',
		'adxr5_ratio', 'apo5_ratio', 'aroondown5_ratio', 'aroonup5_ratio', 'aroonosc5_ratio', 'rsi5_ratio', 'slowk5_ratio',	
		'slowd5_ratio',	'willi5_ratio',	'mom5_ratio', 'macd5_ratio', 'macdsignal5_ratio', 'macdhist5_ratio', 'cci5_ratio', 
		'dx5_ratio', 'cmo5_ratio', 'ppo5_ratio', 'roc5_ratio', 'rocp5_ratio', 'rocr5_ratio', 'ultosc5_ratio']

	df = stock_data[features_stock_data]
	return df