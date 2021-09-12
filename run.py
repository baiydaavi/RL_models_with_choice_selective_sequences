import numpy as np

from environment import TrainingEnvironment
from model_arch import ActorCriticLSTM
from MetaRLModel import MetaRLModel

num_actions = 3
num_states = 42
num_hidden_units = 128
gamma = 0.96
max_episodes = 80001
max_steps_per_episode = 630
r_width = 0.3
learning_rate = 0.001
switch_prob = 0.1

choice_selective_activity = np.load('activity.npz')

model_arch = ActorCriticLSTM(num_actions, num_hidden_units, gamma)
env = TrainingEnvironment(num_states, r_width, switch_prob)
MetaRLModel = MetaRLModel(model_arch,
                          env,
                          choice_selective_activity,
                          num_actions=num_actions,
                          num_states=num_states,
                          num_hidden_units=num_hidden_units,
                          gamma=gamma,
                          max_episodes=max_episodes,
                          max_steps_per_episode=max_steps_per_episode,
                          learning_rate=learning_rate,
                          )
MetaRLModel.learn()
