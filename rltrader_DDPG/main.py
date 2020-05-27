import os
import sys
import logging
import argparse
import json

import settings
import utils
import data_manager

import pandas as pd
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+') #####
    parser.add_argument('--ver', choices=['v1', 'v2'], default='v2')
    parser.add_argument('--rl_method', choices=['ddpg'], default='ddpg')
    parser.add_argument('--net', choices=['dnn', 'lstm', 'actor-critic'], default='lstm')
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=float, default=1.0) #####
    parser.add_argument('--balance', type=int, default=10000000)
    parser.add_argument('--num_epoches', type=int, default=5001) #####
    parser.add_argument('--delayed_reward_threshold', type=float, default=0.05)
    parser.add_argument('--backend', choices=['tensorflow', 'plaidml'], default='tensorflow')
    parser.add_argument('--output_name', default=utils.get_time_str())
    parser.add_argument('--reuse_models', action='store_true', default=False) #####
    parser.add_argument('--learning', action='store_true', default=True) #####
    args = parser.parse_args()

    # Keras Backend 설정
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 설정
    output_path = os.path.join(settings.BASE_DIR, 
        'output/{}_{}_{}'.format(args.output_name, args.rl_method, args.net))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    reuse_path = os.path.join(settings.BASE_DIR, 'models/reuse')

    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    
    # 로그 기록 설정
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, "{}.log".format(args.output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)
        
    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from agent import Agent
    from learners import ReinforcementLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    for stock_code in args.stock_code:
        # set network
        value_network_path = os.path.join(settings.BASE_DIR, 'models/value/{}_{}_value_{}'.format(args.rl_method, args.net, stock_code))
        policy_network_path = os.path.join(settings.BASE_DIR, 'models/policy/{}_{}_policy_{}'.format(args.rl_method, args.net, stock_code))

        # 차트 데이터, 학습 데이터 준비
        # Read Chart Data
        chart_data = pd.read_csv('data/chart/{}.csv'.format(stock_code), usecols=["date", "open", "high", "low", "close", "volume"])

        # Filtering by Date
        chart_data = chart_data[(chart_data['date'] >= '2019-01-01') & (chart_data['date'] <= '2019-12-31')]

        # Load Training Data
        training_data = pd.read_csv('data/train/{}_train.csv'.format(stock_code)) #, usecols=range(1,16))

        #chart_data.to_csv('chart_{}.csv'.format(stock_code), mode='w', index=False)
        #training_data.to_csv('train_{}.csv'.format(stock_code), mode='w', index=False)

        # 최소/최대 투자 단위 설정
        min_trading_unit = 1    #max(int(100000 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = 10   #max(int(1000000 / chart_data.iloc[-1]['close']), 1)

        # 공통 파라미터 설정
        common_params = { 'rl_method': args.rl_method, 
                        'delayed_reward_threshold': args.delayed_reward_threshold,
                        'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
                        'output_path': output_path, 'reuse_models': args.reuse_models,
                        'stock_code': stock_code,
                        'chart_data': chart_data, 
                        'training_data': training_data,
                        'min_trading_unit': min_trading_unit, 
                        'max_trading_unit': max_trading_unit,
                        'value_network_path' : value_network_path,
                        'policy_network_path' : policy_network_path,
                        'reuse_path' : reuse_path}

        # 강화학습 시작
        learner = None
        learner = ReinforcementLearner(**{**common_params})

        learner.run(balance=args.balance, 
                    num_epoches=args.num_epoches, 
                    discount_factor=args.discount_factor, 
                    start_epsilon=args.start_epsilon,
                    learning=args.learning)

        #learner.sample_predict(my_sample)