import os
import sys
import logging
import settings
import pandas as pd
import numpy as np

# set stock code.
stock_code = '005930'

# making LOG in Dir : "./logs/*"
log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
timestr = settings.get_time_str()
if not os.path.exists('logs/%s' % stock_code):
    os.makedirs('logs/%s' % stock_code)
file_handler = logging.FileHandler(filename=os.path.join(
    log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s",
                    handlers=[file_handler, stream_handler], level=logging.DEBUG)

# import my feature selection
import feature_select
from policy_learner import PolicyLearner

if __name__ == '__main__':
    # Read Chart Data CSV file
    chart_data = pd.read_csv('data/chart_data/{}.csv'.format(stock_code))
    chart_data.columns = map(str.lower, chart_data.columns) # to lower case

    #chart_data.drop(chart_data.columns[5], axis='columns') # drop 'adj close'

    # filtering by date, 17.1.1 ~ 18.12.31
    chart_data = chart_data[(chart_data['date'] >= '2017-03-01') &
                                  (chart_data['date'] <= '2018-12-31')]

    # Reset index
    chart_data.reset_index(drop=True, inplace=True)

    # make training data set by using fs() in feature_select.py
    training_data = feature_select.fs(chart_data)

    # making instance of PolicyLearner.
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=10, delayed_reward_threshold=.2, lr=.001)
    
    # Start Fit
    policy_learner.fit(balance=10000000, num_epoches=3000, discount_factor=0, start_epsilon=.5)

    # save network model in Dir : "./models/*"
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    policy_learner.policy_network.save_model(model_path)
