import os
import locale
import logging
import numpy as np
import settings
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer

import Replay_Memory

locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05, lr=0.01):

        # Data Setting
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.training_data = training_data

        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit

        # Environment.
        self.environment = Environment(chart_data)

        # Agent.
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)

        self.sample = None
        self.training_data_idx = -1 # cur idx of training data (used to check environment.price too)

        # Replay Memory for 'Experience Replay' Method.
        self.replay_memory = Replay_Memory.Replay_Memory(1000)

        # number of input to Network = number of ( training_data(15) + agent state(2) )
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM

        # Making Policy Network.
        self.policy_network = PolicyNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        
        # set Visualizer.
        self.visualizer = Visualizer()

    def reset(self):
        self.sample = None
        self.training_data_idx = -1

    def fit(
        self, num_epoches=1000, max_memory=60, balance=10000000,
        discount_factor=0, start_epsilon=.5, learning=True):
        logging.info("LR: {lr}, DF: {discount_factor}, "
                    "TU: [{min_trading_unit}, {max_trading_unit}], "
                    "DRT: {delayed_reward_threshold}, max_trading_unit: {a}, min_trading_unit: {b}".format(
            lr=self.policy_network.lr,
            discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold,
            a=self.max_trading_unit, b=self.min_trading_unit
        ))

        # Ready to Visualize

        # visuallize Chart Data
        self.visualizer.prepare(self.environment.chart_data)

        # dir for save Visualized Image
        epoch_summary_dir = os.path.join(
            settings.BASE_DIR, 'epoch_summary/%s/epoch_summary_%s' % (
                self.stock_code, settings.timestr))
        if not os.path.isdir(epoch_summary_dir):
            os.makedirs(epoch_summary_dir)

        # set agent's initial balance.
        self.agent.set_balance(balance)

        # reset : whole system data.
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # Repeat Epoch.
        for epoch in range(num_epoches):    
            # reset : epoch data
            loss = 0.
            itr_cnt = 0
            win_cnt = 0
            exploration_cnt = 0
            batch_size = 200
            pos_learning_cnt = 0
            neg_learning_cnt = 0

            # reset : memory.
            memory_sample = []
            memory_action = []
            memory_reward = []
            memory_prob = []
            memory_pv = []
            memory_num_stocks = []
            memory_exp_idx = []
            memory_learning_idx = []

            # reset : env(), agent(), network(reset prob), learner(index of training data).
            self.environment.reset()
            self.agent.reset()
            self.policy_network.reset()
            self.reset()

            # reset : visualizer
            self.visualizer.clear([0, len(self.chart_data)])

            # decreasing epsilon.
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches/2 - 1))
            else:
                epsilon = 0


            while True: # one epoch
                # make sample of current state
                cur_sample = self._build_sample()
                if cur_sample is None:
                    break

                # deciding action by network or random.
                action, confidence, exploration = self.agent.decide_action(self.policy_network, cur_sample, epsilon)

                # get reward by action
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                # make sample of next state
                next_sample = self._build_next_sample()

                # replay memory : use only.
                self.replay_memory.push(cur_sample,action,immediate_reward,next_sample)

                # old memory : not use.
                memory_sample.append(cur_sample)
                memory_action.append(action)
                memory_reward.append(immediate_reward)
                memory_pv.append(self.agent.portfolio_value)
                memory_num_stocks.append(self.agent.num_stocks)
                memory = [(
                    memory_sample[i],
                    memory_action[i],
                    memory_reward[i])
                    for i in list(range(len(memory_action)))[-max_memory:]
                ]

                if exploration:
                    memory_exp_idx.append(itr_cnt)
                    memory_prob.append([np.nan] * Agent.NUM_ACTIONS)
                else:
                    memory_prob.append(self.policy_network.prob)

                # Update of epoch data
                itr_cnt += 1
                exploration_cnt += 1 if exploration else 0
                win_cnt += 1 if immediate_reward > 0 else 0

                # Learning Mode & replay_meory.size > 200, Let's Learn !
                if learning and self.replay_memory.get_len() > 200:
                    batch_size = 100

                    # Make Learning Data { x, y }
                    x, y = self._get_batch(self.replay_memory.random_data(batch_size), batch_size, discount_factor, delayed_reward)

                    if len(x) > 0:
                        if immediate_reward > 0:
                            pos_learning_cnt += 1
                        else:
                            neg_learning_cnt += 1

                        # Network Learn of Random Experience.
                        loss += self.policy_network.train_on_batch(x, y)
                        memory_learning_idx.append([itr_cnt, delayed_reward])

            # Visualize Action.
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')

            self.visualizer.plot(
                epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
                action_list=Agent.ACTIONS, actions=memory_action,
                num_stocks=memory_num_stocks, outvals=memory_prob,
                exps=memory_exp_idx, learning=memory_learning_idx,
                initial_balance=self.agent.initial_balance, pvs=memory_pv
            )
            self.visualizer.save(os.path.join(
                epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                    settings.timestr, epoch_str)))

            # Logging Action.
            if pos_learning_cnt + neg_learning_cnt > 0:
                loss /= pos_learning_cnt + neg_learning_cnt
            logging.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"
                        "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
                        "#Stocks:%d\tPV:%s\t"
                        "Loss:%10.6f" % (
                            epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_stocks,
                            locale.currency(self.agent.portfolio_value, grouping=True),
                            loss))

            # Upate data of Learn.
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # whole system data Logging.
        logging.info("Max PV: %s, \t # Win: %d" % (
            locale.currency(max_portfolio_value, grouping=True), epoch_win_cnt))

    # get Learning Data. { x, y }. function.
    def _get_batch(self, replay_memory, batch_size, discount_factor, delayed_reward):
        x = np.zeros((batch_size, 1, self.num_features))
        y = np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)
        cnt = 0

        for i, (sample, action, reward, next_sample) in enumerate(replay_memory):
            x[i] = np.array(sample).reshape((-1, 1, self.num_features))
            if(next_sample != None):
                y[i-cnt, action] = reward + (1-discount_factor) * np.amax(self.policy_network.predict(next_sample))
            else:
            	cnt = cnt + 1;

        return x, y

    # making current sample function.
    def _build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            sample = self.training_data.iloc[self.training_data_idx].tolist()
            sample.extend(self.agent.get_states())

            return sample
        return None

    # making next sample function.
    def _build_next_sample(self):
        next_training_data_idx = self.training_data_idx # copy idx

        if len(self.training_data) > next_training_data_idx + 1:
            next_training_data_idx += 1
            next_sample = self.training_data.iloc[next_training_data_idx].tolist()
            next_sample.extend(self.agent.get_states())

            return next_sample
        return None

    # What is it ?
    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)
        self.fit(balance=balance, num_epoches=1, learning=False)
