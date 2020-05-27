import sys
import numpy as np

#from tqdm import tqdm
from actor import Actor
from critic import Critic
from utility.stats import gather_stats
from utility.networks import tfSummary

class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, feature_dim, gamma = 0.99, lr = 0.00005, tau = 0.01):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.actor = Actor(feature_dim, act_dim, 0.1 * lr, tau)
        self.critic = Critic(feature_dim, act_dim, lr, tau)
        self.buffer = None#MemoryBuffer(buffer_size)

    def get_predict(self, s):
        """ Use the actor to predict value
        """
        value = self.critic.target_predict(s, self.actor.target_predict(s))
        policy = self.actor.predict(s)
        return value, policy

    def bellman(self, rewards, q_values):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        # actions = self.actor.predict(states) ##
        #print(actions)
        grads = self.critic.gradients(states, actions)
        # Train actor
        self.actor.train(states, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self, samples, actions, rewards, next_samples):
        # Predict target q-values using target networks
        q_values = self.critic.target_predict(next_samples, self.actor.target_predict(next_samples))
        # Compute critic target
        critic_target = self.bellman(rewards, q_values)
        # Train both networks on sampled batch, update target networks
        self.update_models(samples, actions, critic_target)

        return 0

    def save_weights(self, value_path, policy_path):
        self.actor.save(policy_path)
        self.critic.save(value_path)

    def load_weights(self, value_path, policy_path):
        self.actor.load_weights(policy_path)
        self.critic.load_weights(value_path)