"""This module defines the reversal learning environments.
"""

import numpy as np

from utils import normalized_gaussian


class TrainingEnvironment:
    """Training Environment"""
    def __init__(self, len_t, r_width, switch_prob):
        self.len_t = len_t
        self.t_vec = np.round(np.arange(-2.1, 2.1, 0.1), 3)
        self.r_width = r_width
        self.switch_prob = switch_prob
        self.r_time = np.round(np.arange(0.2, 1.3, 0.1), 3)
        self.time = 0
        self.timestep = 0
        self.rewarded = 0
        self.choice = -1
        self.bandit = [0.7, 0.1] if np.random.uniform() <= 0.5 else [0.1, 0.7]
        self.done = False

        self.r_vec = normalized_gaussian(self.t_vec,
                                         np.random.choice(self.r_time),
                                         self.r_width)
        self.r_vec[self.t_vec <= 0.0] = 0.0
        self.rand_trial = np.random.randint(100)
        self.reset()

    def reset(self):
        self.time = 0
        self.timestep = 0
        self.rewarded = 0
        self.choice = -1
        self.bandit = [0.7, 0.1] if np.random.uniform() <= 0.5 else [0.1, 0.7]
        self.done = False

        self.r_vec = normalized_gaussian(self.t_vec, np.random.choice(self.r_time),
                                   self.r_width)
        self.r_vec[self.t_vec <= 0.0] = 0.0
        self.rand_trial = np.random.randint(100)

    def step(self, action):

        self.timestep += 1

        reward = self.r_vec[self.time] * self.rewarded

        if self.time == 0:

            self.r_vec = normalized_gaussian(self.t_vec,
                                       np.random.choice(self.r_time),
                                       self.r_width)
            self.r_vec[self.t_vec <= 0.0] = 0.0
            self.rand_trial = np.random.randint(100)
            self.rewarded = 0
            self.choice = action - 1

            if np.random.uniform() < self.switch_prob:
                self.bandit = list(reversed(self.bandit))

            if action == 0:
                reward = -1

            else:

                bandit = self.bandit[self.choice]

                result = np.random.uniform()

                if result < bandit:
                    self.rewarded = 1

        else:

            if action != 0:
                reward = -1

        self.time = (self.time + 1) % self.len_t

        return self.time, reward, self.done, self.timestep


# class TestingEnvironment:

# class FixedBlockTestingEnvironment: