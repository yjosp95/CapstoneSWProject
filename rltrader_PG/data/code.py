for stock_code in args.stock_code:
        
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