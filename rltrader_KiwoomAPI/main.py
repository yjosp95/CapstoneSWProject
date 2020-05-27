import crawling

import pandas as pd
import numpy as np
import os

from actor import Actor

#////////////////////////
import feature_select
#////////////////////////

if __name__ == '__main__':

    stock_code = '035250'

    min_trading_unit = 1
    max_trading_unit = 10

    chart_data = crawling.crawling(stock_code)
    chart_data.to_csv("035250.csv",mode='w',index=False) #for index
    # pre processing
    my_sample = None

    # YH /////////////////////////////////////////////////////////////
    prep_data=feature_select.preprocess(chart_data)
    prep_data.to_csv("prep_data.csv",mode='w',index=False) #for index
    prep_data = pd.read_csv("prep_data.csv")
    training_data = feature_select.build_training_data(prep_data)
    training_data = feature_select.load_talib(training_data)
    training_data = feature_select.build_training_data2(training_data)      #The data based on ratio from ta-lib
    training_data = feature_select.ratio_data(training_data)                #Ratio data
    training_data = feature_select.normalize(training_data)                 #Normalize
    my_sample = feature_select.fs(training_data)
    my_sample = my_sample.values[len(my_sample)-1]
    
    val = np.array([0.2, 1.0])
    my_sample = np.append(my_sample, val)
    #/////////////////////////////////////////////////////////////////

    # make actor
    actor_path = os.path.join(os.getcwd(), 'actor_model/my_actor_{}.h5'.format(stock_code))
    my_actor = Actor(17, 2, actor_path)

    # predict by actor
    pred = None
    pred = my_actor.predict(my_sample) # prob : [buy, sell]

    # deciding action, trading_unit
    my_action = np.argmax(pred) # 0:buy, 1:sell
    my_confidence = np.max(pred) # max prob = confidence

    # calculate trading unit
    # confidence
    added_traiding = max(min(int(my_confidence * (max_trading_unit - min_trading_unit)), max_trading_unit-min_trading_unit), 0)

    my_trading_unit = min_trading_unit + added_traiding

    gap = 0.0

    if(float(pred[0][1]) > float(pred[0][0])):
        gap = float(pred[0][1]) - float(pred[0][0])
    else:
        gap = float(pred[0][0]) - float(pred[0][1])

    print(my_action, gap)
    # make text
    if int(my_action) == 0 and gap > 0.1:
        my_text = '매수;{};시장가;{};0;매수전'.format(stock_code, my_trading_unit)
        f = open('/home/user/kmc/Dropbox/googletest/buy_list.txt', mode='wt',encoding='utf-8')
        f.write(my_text)
        print(my_text)
        f.close()
    elif int(my_action) == 1 and gap > 0.1:
        my_text = '매도;{};시장가;{};0;매도전'.format(stock_code, my_trading_unit)
        f = open('/home/user/kmc/Dropbox/googletest/sell_list.txt', mode='wt',encoding='utf-8')
        f.write(my_text)
        print(my_text)
        f.close()
    print(pred[0])
    

    # JH
   
