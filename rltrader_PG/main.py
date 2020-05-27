#-*-coding: utf-8-*-
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
import timeit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+')#

    parser.add_argument('--rl_method', 
        choices=['dqn', 'pg', 'ac', 'a2c', 'a3c'], default='pg')#

    parser.add_argument('--net', 
        choices=['dnn', 'lstm'], default='dnn')

    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=float, default=0.5)
    parser.add_argument('--balance', type=int, default=10000000)
    parser.add_argument('--num_epoches', type=int, default=10)
    parser.add_argument('--delayed_reward_threshold', 
        type=float, default=0.05)
    parser.add_argument('--backend', 
        choices=['tensorflow', 'plaidml'], default='tensorflow')
    parser.add_argument('--output_name', default=utils.get_time_str())
    parser.add_argument('--reuse_models', action='store_true', default='false')
    parser.add_argument('--learning', action='store_true', default='true')
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
    from learners import DQNLearner, PolicyGradientLearner, \
        ActorCriticLearner, A2CLearner, A3CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    for stock_code in args.stock_code:
        ft = open("./time.txt", 'a+')
        start = timeit.default_timer() ############
        # Read Chart Data
        chart_data = pd.read_csv('data/chart/{}.csv'.format(stock_code), usecols=["date", "open", "high", "low", "close", "volume"])
        
        # TEST
        if(args.reuse_models == 'true'):
            value_network_path = os.path.join(settings.BASE_DIR, 'models/reuse/value/{}_{}_value_{}.h5'.format(args.rl_method, args.net, stock_code))
            policy_network_path = os.path.join(settings.BASE_DIR, 'models/reuse/policy/{}_{}_policy_{}.h5'.format(args.rl_method, args.net, stock_code))

            chart_data = chart_data[(chart_data['date'] >= '2019-01-01') & (chart_data['date'] <= '2019-12-31')]
            training_data = pd.read_csv('data/test/{}.csv'.format(stock_code))
        
        # TRAIN
        else:
            value_network_path = os.path.join(settings.BASE_DIR, 'models/value/{}_{}_value_{}'.format(args.rl_method, args.net, stock_code))
            policy_network_path = os.path.join(settings.BASE_DIR, 'models/policy/{}_{}_policy_{}'.format(args.rl_method, args.net, stock_code))

            chart_data = chart_data[(chart_data['date'] >= '2010-01-01') & (chart_data['date'] <= '2018-12-31')]
            training_data = pd.read_csv('data/train/{}.csv'.format(stock_code))

        # 최소/최대 투자 단위 설정
        min_trading_unit = 1    #max(int(100000 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = 10   #max(int(1000000 / chart_data.iloc[-1]['close']), 1)

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method, 
            'delayed_reward_threshold': args.delayed_reward_threshold,
            'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
            'output_path': output_path, 'reuse_models': args.reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_unit': min_trading_unit, 
                'max_trading_unit': max_trading_unit})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            if learner is not None:
                learner.run(balance=args.balance, 
                    num_epoches=args.num_epoches, 
                    discount_factor=args.discount_factor, 
                    start_epsilon=args.start_epsilon,
                    learning=args.learning)
                # learner.save_models()
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_unit.append(min_trading_unit)
            list_max_trading_unit.append(max_trading_unit)

        stop = timeit.default_timer()
        ft.write(str(Stock_code) + " " + str(stop-start) + "\n")
        ft.close()
    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_unit': list_min_trading_unit, 
            'list_max_trading_unit': list_max_trading_unit,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})

        learner.run(balance=args.balance, num_epoches=args.num_epoches, 
                    discount_factor=args.discount_factor, 
                    start_epsilon=args.start_epsilon,
                    learning=args.learning)
        # learner.save_models()
        
        # python3 main.py --stock_code 000720 --net dnn --num_epoches 10000
        # python3 main.py --stock_code 005380 --net dnn --num_epoches 10000
        # python3 main.py --stock_code 015760 --net dnn --num_epoches 10000
        # python3 main.py --stock_code 033780 --net dnn --num_epoches 10000