import os
import sys
import pandas as pd
import numpy as np
import feature_select   #ver1
import feature_select2  #ver2

stock_code = '002240'

import feature_select

if __name__ == '__main__':
    # Read Chart Data CSV file
    chart_data = pd.read_csv('data/chart_data/{}.csv'.format(stock_code))

    #///////////////////////////////////////////////////////////////////////// ver2 //////////////////////////////////////////
    #chart_data, training_data = feature_select2.load_data('data/chart_data/{}.csv'.format(stock_code),'2009-01-14','2019-12-31')

    #number_data=chart_data[(chart_data['date']>='2008-01-01')&  #ver2
    #(chart_data['date']<='2019-12-31')]
    #number_data2=chart_data[(chart_data['date']>='2019-01-01')&
    #(chart_data['date']<='2020-12-31')]

    #print(number_data)
    #exit()

    #number = (number_data.index[len(number_data)-1]-number_data.index[0])+1
    #number2 = (number_data2.index[len(number_data2)-1]-number_data2.index[0])+1 #data number --> parameter value

    #train_data, test_data = feature_select.separation(training_data,number,number2)
    #train_data.to_csv('068270_train.csv', mode='w',index=False)
    #test_data.to_csv('068270_test.csv', mode='w',index=False)
    #exit()
    #//////////////////////////////////////////////////////////////////////// ver2 //////////////////////////////////////////

    chart_data = chart_data[(chart_data['date']>='2004-01-01')&  #for calculating
    (chart_data['date']<='2019-12-31')]

    number_data=chart_data[(chart_data['date']>='2005-01-01')&  #ver1 
    (chart_data['date']<='2018-12-31')]
    number_data2=chart_data[(chart_data['date']>='2019-01-01')&  
    (chart_data['date']<='2019-12-31')]

    number = (number_data.index[len(number_data)-1]-number_data.index[0])+1
    number2 = (number_data2.index[len(number_data2)-1]-number_data2.index[0])+1 #data number --> parameter value
 
    prep_data = feature_select.preprocess(chart_data)
 
    training_data = feature_select.build_training_data(prep_data)
    training_data = training_data.dropna() #cut dates
    
    #///////////////////////////////////////////////////#
    training_data = feature_select.load_talib(training_data)               #Data of ta-lib
    training_data = feature_select.build_training_data2(training_data)      #The data based on ratio from ta-lib
    training_data = feature_select.ratio_data(training_data)                #Ratio data
    training_data = feature_select.normalize(training_data)                 #Normalize
    training_data=feature_select.fs(training_data)

    train_data, test_data = feature_select.separation(training_data,number,number2)

    train_data.to_csv('002240_train.csv', mode='w',index=False)
    test_data.to_csv('002240_test.csv', mode='w',index=False)
    
    #///////////////////////////////////////////////////#

    #a_a=feature_select.forward(training_data)                             #other method
    #a_a.to_csv('chart_068270_2.csv', mode='w',index=False)
    #b_b=feature_select.backward(training_data)                            #other method
    #b_b.to_csv('chart_068270_3.csv', mode='w',index=False)