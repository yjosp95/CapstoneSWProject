import random

class Replay_Memory():
    
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.memory = []
        self.idx = 0

    def push(self, sample, action, reward, next_sample):
        if len(self.memory) < self.maxlen:
           self.memory.append(None)
        self.memory[self.idx] = (sample,action,reward,next_sample)
        self.idx = (self.idx+1) % self.maxlen

    def random_data(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_len(self):
        return len(self.memory)
