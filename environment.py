"""This module defines the reversal learning environments."""

import numpy as np

from utils import normalized_gaussian


class TrainingEnvironment:
    """Training Environment"""

    def __init__(self, reward_width, block_switch_prob):

        self.time_array = np.round(np.arange(-2.1, 2.1, 0.1), 3)
        self.len_time_array = len(self.time_array)
        self.current_trial_time = 0
        self.current_episode_time = 0

        self.mean_reward_time_array = np.round(np.arange(0.2, 1.3, 0.1), 3)
        self.reward_width = reward_width
        self.reward_array = normalized_gaussian(self.time_array,
                                                np.random.choice(
                                                    self.mean_reward_time_array),
                                                self.reward_width)
        self.reward_array[self.time_array <= 0.0] = 0.0
        self.is_rewarded = 0
        self.reward_prob = [0.7, 0.1] if np.random.uniform() <= 0.5 else [0.1,
                                                                          0.7]

        self.block_switch_prob = block_switch_prob

        self.choice = -1

        self.rand_activity_trial_num = np.random.randint(100)

    def reset(self):

        self.current_trial_time = 0
        self.current_episode_time = 0

        self.reward_array = normalized_gaussian(self.time_array,
                                                np.random.choice(
                                                    self.mean_reward_time_array),
                                                self.reward_width)
        self.reward_array[self.time_array <= 0.0] = 0.0
        self.is_rewarded = 0

        self.reward_prob = [0.7, 0.1] if np.random.uniform() <= 0.5 else [0.1,
                                                                          0.7]

        self.choice = -1

        self.rand_activity_trial_num = np.random.randint(100)

    def step(self, action):

        self.current_episode_time += 1

        reward = self.reward_array[self.current_trial_time] * self.is_rewarded

        # Start of a trial
        if self.current_trial_time == 0:
            self.reward_array = normalized_gaussian(self.time_array,
                                                    np.random.choice(
                                                        self.mean_reward_time_array),
                                                    self.reward_width)
            self.reward_array[self.time_array <= 0.0] = 0.0
            self.is_rewarded = 0

            self.choice = action - 1

            self.rand_activity_trial_num = np.random.randint(100)

            if np.random.uniform() < self.block_switch_prob:
                self.reward_prob = list(reversed(self.reward_prob))

            if action == 0:
                reward = -1

            else:
                current_choice_reward_prob = self.reward_prob[self.choice]
                if np.random.uniform() < current_choice_reward_prob:
                    self.is_rewarded = 1

        else:
            if action != 0:
                reward = -1

        self.current_trial_time = (self.current_trial_time + 1) % \
                                  self.len_time_array

        return self.current_trial_time, reward, self.is_rewarded, \
               self.current_episode_time


class TestingEnvironment:
    """Testing Environment"""

    def __init__(self, reward_width):

        self.time_array = np.round(np.arange(-2.1, 2.1, 0.1), 3)
        self.len_time_array = len(self.time_array)
        self.current_trial_time = 0
        self.current_episode_time = 0
        self.trial_count_in_block = 0

        self.mean_reward_time_array = np.round(np.arange(0.2, 1.3, 0.1), 3)
        self.reward_width = reward_width
        self.reward_array = normalized_gaussian(self.time_array,
                                                np.random.choice(
                                                    self.mean_reward_time_array),
                                                self.reward_width)
        self.reward_array[self.time_array <= 0.0] = 0.0
        self.is_rewarded = 0
        self.reward_count = 0

        self.reward_prob = [0.7, 0.1] if np.random.uniform() <= 0.5 else [0.1,
                                                                          0.7]

        self.choice = -1

        self.rand_activity_trial_num = np.random.randint(100)

    def reset(self):

        self.current_trial_time = 0
        self.current_episode_time = 0
        self.trial_count_in_block = 0

        self.reward_array = normalized_gaussian(self.time_array,
                                                np.random.choice(
                                                    self.mean_reward_time_array),
                                                self.reward_width)
        self.reward_array[self.time_array <= 0.0] = 0.0
        self.is_rewarded = 0
        self.reward_count = 0

        self.reward_prob = [0.7, 0.1] if np.random.uniform() <= 0.5 else [0.1,
                                                                          0.7]

        self.choice = -1

        self.rand_activity_trial_num = np.random.randint(100)

    def step(self, action):

        self.current_episode_time += 1

        reward = self.reward_array[self.current_trial_time] * self.is_rewarded

        # Start of a trial
        if self.current_trial_time == 0:
            self.reward_array = normalized_gaussian(self.time_array,
                                                    np.random.choice(
                                                        self.mean_reward_time_array),
                                                    self.reward_width)
            self.reward_array[self.time_array <= 0.0] = 0.0
            self.is_rewarded = 0

            self.choice = action - 1

            self.rand_activity_trial_num = np.random.randint(100)

            if self.reward_count > 9:
                if np.random.uniform() > 0.4:
                    self.reward_prob = list(reversed(self.reward_prob))
                    self.reward_count = 0
                    self.trial_count_in_block = 0

            self.trial_count_in_block += 1

            if action == 0:
                reward = -1

            else:
                current_choice_reward_prob = self.reward_prob[self.choice]
                if np.random.uniform() < current_choice_reward_prob:
                    self.is_rewarded = 1
                    self.reward_count += 1

        else:
            if action != 0:
                reward = -1

        self.current_trial_time = (self.current_trial_time + 1) % \
                                  self.len_time_array

        return self.current_trial_time, reward, self.is_rewarded, \
               self.trial_count_in_block


class OptoTestingEnvironment:
    """Testing Environment for the optogenetic activation case."""

    def __init__(self, reward_width):

        self.time_array = np.round(np.arange(-2.1, 2.1, 0.1), 3)
        self.len_time_array = len(self.time_array)
        self.current_trial_time = 0
        self.current_episode_time = 0
        self.trial_count_in_block = 0

        self.mean_reward_time_array = np.round(np.arange(0.2, 1.3, 0.1), 3)
        self.reward_width = reward_width
        self.reward_array = normalized_gaussian(self.time_array,
                                                np.random.choice(
                                                    self.mean_reward_time_array),
                                                self.reward_width)
        self.reward_array[self.time_array <= 0.0] = 0.0
        self.is_rewarded = 0
        self.reward_count = 0

        self.reward_prob = [0.7, 0.1] if np.random.uniform() <= 0.5 else [0.1,
                                                                          0.7]

        self.choice = -1

        self.rand_activity_trial_num = np.random.randint(100)

        self.is_stimulated = 0

    def reset(self):

        self.current_trial_time = 0
        self.current_episode_time = 0
        self.trial_count_in_block = 0

        self.reward_array = normalized_gaussian(self.time_array,
                                                np.random.choice(
                                                    self.mean_reward_time_array),
                                                self.reward_width)
        self.reward_array[self.time_array <= 0.0] = 0.0
        self.is_rewarded = 0
        self.reward_count = 0

        self.reward_prob = [0.7, 0.1] if np.random.uniform() <= 0.5 else [0.1,
                                                                          0.7]

        self.choice = -1

        self.rand_activity_trial_num = np.random.randint(100)

        self.is_stimulated = 0

    def step(self, action):

        self.current_episode_time += 1

        reward = self.reward_array[self.current_trial_time] * self.is_rewarded

        # Start of a trial
        if self.current_trial_time == 0:
            self.reward_array = normalized_gaussian(self.time_array,
                                                    np.random.choice(
                                                        self.mean_reward_time_array),
                                                    self.reward_width)
            self.reward_array[self.time_array <= 0.0] = 0.0
            self.is_rewarded = 0

            self.choice = action - 1

            self.rand_activity_trial_num = np.random.randint(100)

            if np.random.uniform() < 0.1:
                self.is_stimulated = 1
            else:
                self.is_stimulated = 0

            if self.reward_count > 9:
                if np.random.uniform() > 0.4:
                    self.reward_prob = list(reversed(self.reward_prob))
                    self.reward_count = 0
                    self.trial_count_in_block = 0

            self.trial_count_in_block += 1

            if action == 0:
                reward = -1

            else:
                current_choice_reward_prob = self.reward_prob[self.choice]
                if np.random.uniform() < current_choice_reward_prob:
                    self.is_rewarded = 1
                    self.reward_count += 1

        else:
            if action != 0:
                reward = -1

        if self.current_trial_time == self.len_time_array - 1:
            self.is_stimulated = 0

        self.current_trial_time = (self.current_trial_time + 1) % \
                                  self.len_time_array

        return self.current_trial_time, reward, self.is_rewarded, \
               self.trial_count_in_block


class FixedBlockTestingEnvironment:
    """Fixed Block Testing Environment"""

    def __init__(self, reward_width):

        self.time_array = np.round(np.arange(-2.1, 2.1, 0.1), 3)
        self.len_time_array = len(self.time_array)
        self.current_trial_time = 0
        self.current_episode_time = 0
        self.trial_count_in_block = 0

        self.mean_reward_time_array = np.round(np.arange(0.2, 1.3, 0.1), 3)
        self.reward_width = reward_width
        self.reward_array = normalized_gaussian(self.time_array,
                                                np.random.choice(
                                                    self.mean_reward_time_array),
                                                self.reward_width)
        self.reward_array[self.time_array <= 0.0] = 0.0
        self.is_rewarded = 0

        self.reward_prob = [0.7, 0.1] if np.random.uniform() <= 0.5 else [0.1,
                                                                          0.7]

        self.choice = -1

        self.rand_activity_trial_num = np.random.randint(100)

    def reset(self):

        self.current_trial_time = 0
        self.current_episode_time = 0
        self.trial_count_in_block = 0

        self.reward_array = normalized_gaussian(self.time_array,
                                                np.random.choice(
                                                    self.mean_reward_time_array),
                                                self.reward_width)
        self.reward_array[self.time_array <= 0.0] = 0.0
        self.is_rewarded = 0

        self.reward_prob = [0.7, 0.1] if np.random.uniform() <= 0.5 else [0.1,
                                                                          0.7]

        self.choice = -1

        self.rand_activity_trial_num = np.random.randint(100)

    def step(self, action):

        self.current_episode_time += 1

        reward = self.reward_array[self.current_trial_time] * self.is_rewarded

        # Start of a trial
        if self.current_trial_time == 0:
            self.reward_array = normalized_gaussian(self.time_array,
                                                    np.random.choice(
                                                        self.mean_reward_time_array),
                                                    self.reward_width)
            self.reward_array[self.time_array <= 0.0] = 0.0
            self.is_rewarded = 0

            self.choice = action - 1

            self.rand_activity_trial_num = np.random.randint(100)

            if self.trial_count_in_block > 29:
                self.reward_prob = list(reversed(self.reward_prob))
                self.trial_count_in_block = 0

            self.trial_count_in_block += 1

            if action == 0:
                reward = -1

            else:
                current_choice_reward_prob = self.reward_prob[self.choice]
                if np.random.uniform() < current_choice_reward_prob:
                    self.is_rewarded = 1

        else:
            if action != 0:
                reward = -1

        self.current_trial_time = (self.current_trial_time + 1) % \
                                  self.len_time_array

        return self.current_trial_time, reward, self.is_rewarded, \
               self.trial_count_in_block
