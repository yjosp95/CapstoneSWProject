import random
import numpy as np
from collections import deque

class Replay_Memory():
    
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.memory = deque()
        self.cnt = 0

    def push(self, sample, action, reward, next_sample):
        experience = (sample, action, reward, next_sample)

        if self.cnt < self.maxlen:
            self.memory.append(experience)
            self.cnt += 1
        else:
            self.memory.popleft()
            self.memory.append(experience)

    def random_data(self, batch_size):
        if(self.cnt > batch_size):
            return random.sample(self.memory, batch_size)
        else:
            return random.sample(self.memory, self.cnt)

    def all_data(self):
        return self.memory

    def get_len(self):
        return len(self.memory)

    def reset(self):
        self.memory.clear()
        self.cnt = 0

    # def __init__(self, maxlen):
    #     self.maxlen = maxlen
    #     self.memory = []
    #     self.idx = 0

    # def push(self, sample, action, reward, next_sample):
    #     if len(self.memory) < self.maxlen:
    #        self.memory.append(None)
    #     self.memory[self.idx] = (sample,action,reward,next_sample)
    #     self.idx = (self.idx+1) % self.maxlen

    # def random_data(self, batch_size):
    #     return random.sample(self.memory, batch_size)

    # def get_len(self):
    #     return len(self.memory)
