import os
import logging
import abc
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
#from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer

from ddpg import DDPG
from Replay_Memory import Replay_Memory
#from ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

class ReinforcementLearner():
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', stock_code=None, 
                chart_data=None, training_data=None,
                min_trading_unit=1, max_trading_unit=2, 
                delayed_reward_threshold=.05,
                net='dnn', num_steps=1, lr=0.001,
                value_network=None, policy_network=None,
                output_path='', reuse_models=True,
                value_network_path=None, policy_network_path=None, reuse_path=None):

        # 인자 확인
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0
        # 강화학습 기법 설정
        self.rl_method = rl_method
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment,
                    min_trading_unit=min_trading_unit,
                    max_trading_unit=max_trading_unit,
                    delayed_reward_threshold=delayed_reward_threshold)
        # 학습 데이터
        self.training_data = training_data
        
        self.cur_sample = None
        self.next_sample = None

        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path

        self.net = net
        self.num_steps = num_steps
        self.lr = lr

        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        # Replay Memory for 'Experience Replay' Method.
        self.replay_memory = Replay_Memory(300)

        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        # 로그 등 출력 경로
        self.output_path = output_path

        # DDPG setting
        self.ddpg = DDPG(self.agent.NUM_ACTIONS, self.num_features, 0.99, lr, 0.1)

        reuse_value_path = reuse_path + '/value/{}_{}_value_{}.h5'.format(rl_method, net, stock_code)
        reuse_policy_path = reuse_path + '/policy/{}_{}_policy_{}.h5'.format(rl_method, net, stock_code)

        if reuse_models is True:
            self.ddpg.load_weights(reuse_value_path, reuse_policy_path)

    def reset(self):
        self.cur_sample = None
        self.next_sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.replay_memory.reset()

        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.0
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_cur_sample(self):
        self.environment.observe()
        making_sample = None
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            making_sample = self.training_data.iloc[
                self.training_data_idx].tolist()
            making_sample.extend(self.agent.get_states())
            return making_sample
        return None

    def build_next_sample(self):
        #self.environment.observe()
        next_training_data_idx = self.training_data_idx # copy idx
        making_sample = None

        if len(self.training_data) > next_training_data_idx + 1:
            next_training_data_idx += 1

            making_sample = self.training_data.iloc[
                self.training_data_idx].tolist()
            making_sample.extend(self.agent.get_states())
            return making_sample
        return None

    def update_networks(self, 
            batch_size, discount_factor):
        
        memory = self.replay_memory.random_data(batch_size)

        samples = []
        actions = []
        rewards = []
        next_samples = []

        #(samples, actions, rewards, next_samples) = memory 

        for i, (sample, action, reward, next_sample) in enumerate(memory):
            samples.append(sample)
            actions.append(action)
            rewards.append(reward)
            next_samples.append(next_sample)
        
        loss = self.ddpg.train(samples,actions,rewards,next_samples)

        return loss
        # # 배치 학습 데이터 생성
        # x, y_value, y_policy = self.get_batch(batch_size, discount_factor)

        # if len(x) > 0:
        #     loss = 0
        #     if y_value is not None:
        #         # 가치 신경망 갱신
        #         loss += self.value_network.train_on_batch(x, y_value)
        #     if y_policy is not None:
        #         # 정책 신경망 갱신
        #         loss += self.policy_network.train_on_batch(x, y_policy)
        #     return loss
        # return None

    def fit(self, discount_factor):
        batch_size = 10

        # 배치 학습 데이터 생성 및 신경망 갱신
        if batch_size > 0:
            _loss = self.update_networks(batch_size, discount_factor)
            
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0

    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] \
            * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) \
            + self.memory_num_stocks

        self.memory_value = [np.array([np.nan] \
            * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                + self.memory_value

        self.memory_policy = [np.array([np.nan] \
            * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                + self.memory_policy

        self.memory_pv = [self.agent.initial_balance] \
            * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, 
            epsilon=epsilon, action_list=Agent.ACTIONS, 
            actions=self.memory_action, 
            num_stocks=self.memory_num_stocks, 
            outvals_value=self.memory_value, 
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx, 
            learning_idxes=self.memory_learning_idx,
            initial_balance=self.agent.initial_balance, 
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(
            self.epoch_summary_dir, 
            'epoch_summary_{}.png'.format(epoch_str))
        )

    def sample_predict(self, my_sample):

        pred_value, pred_policy = self.ddpg.get_predict(my_sample)
        my_action, my_unit = self.agent.decide_my_action(pred_policy)

        # print text file.
        my_text = None
        if my_action is 0:
            my_text = '매수;{};시장가;{};0;매수전'.format(self.stock_code, my_unit)
        else:
            my_text = '매도;{};시장가;{};0;매도전'.format(self.stock_code, my_unit)

    def run(
        self, num_epoches=100, balance=10000000,
        discount_factor=0.9, start_epsilon=0.5, learning=True):
        info = "[{code}] RL:{rl} Net:{net} LR:{lr} " \
            "DF:{discount_factor} TU:[{min_trading_unit}," \
            "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
            code=self.stock_code, rl=self.rl_method, net=self.net,
            lr=self.lr, discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit, 
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        )
        with self.lock:
            logging.info(info)

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(
            self.output_path, 'epoch_summary_{}'.format(
                self.stock_code))
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon \
                    * (1. - float(epoch) / (num_epoches - 1))
                self.agent.reset_exploration()
            else:
                epsilon = start_epsilon

            while True:
                # cur 샘플 생성
                cur_sample = self.build_cur_sample()
                if cur_sample is None:
                    break

                # next 샘플 생성
                next_sample = self.build_next_sample()
                if next_sample is None:
                    break

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                
                pred_value, pred_policy = self.ddpg.get_predict(cur_sample)

                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = \
                    self.agent.decide_action(pred_policy, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = \
                    self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                # replay memory
                if immediate_reward > 0.01 or immediate_reward < -0.01:
                    self.replay_memory.push(cur_sample, action, immediate_reward, next_sample)
                    
                self.memory_sample.append(cur_sample)
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                if pred_value is not None:
                    self.memory_value.append(pred_value)
                if pred_policy is not None:
                    self.memory_policy.append(pred_policy)
                
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

                # 지연 보상 발생된 경우 미니 배치 학습
                if learning and self.replay_memory.get_len() > 10:
                    self.fit(discount_factor)

            # End of Epoch
            if (epoch == 1000 or epoch == 5000 or epoch == 9999 or epoch == 19999):
                vp = self.value_network_path + '_{}.h5'.format(epoch)
                pp = self.policy_network_path + '_{}.h5'.format(epoch)

                self.ddpg.save_weights(vp, pp)

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt
            logging.info("[{}][Epoch {}/{}] Epsilon:{:.4f} "
                "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                "#Stocks:{} PV:{:,.0f} "
                "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                    self.stock_code, epoch_str, num_epoches, epsilon, 
                    self.exploration_cnt, self.itr_cnt,
                    self.agent.num_buy, self.agent.num_sell, 
                    self.agent.num_hold, self.agent.num_stocks, 
                    self.agent.portfolio_value, self.learning_cnt, 
                    self.loss, elapsed_time_epoch))

            # 에포크 관련 정보 가시화
            #self.visualize(epoch_str, num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logging.info("[{code}] Elapsed Time:{elapsed_time:.4f} "
                "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                code=self.stock_code, elapsed_time=elapsed_time, 
                max_pv=max_portfolio_value, cnt_win=epoch_win_cnt))