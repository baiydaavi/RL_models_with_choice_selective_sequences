import argparse
import numpy as np

from MetaRLModel import MetaRLModel
from environment import TrainingEnvironment
from model_arch import ActorCriticLSTM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--num_actions', type=int, default=3,
                        help='number of allowed actions')
    parser.add_argument('--num_states', type=int, default=42,
                        help='number of states')
    parser.add_argument('--num_hidden_units', type=int, default=128,
                        help='number of hidden units for both actor and '
                             'critic LSTM')
    parser.add_argument('--gamma', type=int, default=0.96,
                        help='discount factor per time step')
    parser.add_argument('--learning_rate', type=int, default=0.001,
                        help='learning rate')
    parser.add_argument('--block_switch_prob', type=int, default=0.1,
                        help='probability of a block switch on any given trial')
    parser.add_argument('--max_episodes', type=int, default=70001,
                        help='maximum number of episodes to train for')
    parser.add_argument('--max_steps_per_episode', type=int, default=15 * 42,
                        help='number of steps per each episode')
    parser.add_argument('--reward_width', type=int, default=0.3,
                        help='width of the Gaussian reward input')
    parser.add_argument('--input_type', type=str, default='sequential',
                        help='choice-selective input type to critic - ['
                             'sequential, persistent]')
    args = parser.parse_args()

    # Network
    model_architecture = ActorCriticLSTM(args.num_actions,
                                         args.num_hidden_units, args.gamma)

    # RL environment
    environment = TrainingEnvironment(args.reward_width, args.block_switch_prob)

    # Choice-selective input activity
    choice_selective_activity = np.load(
        f"helper_files/{args.input_type}_activity.npz")

    # Meta RL Model
    MetaRLModel = MetaRLModel(model_architecture,
                              environment,
                              choice_selective_activity,
                              input_type=args.input_type,
                              num_actions=args.num_actions,
                              num_states=args.num_states,
                              num_hidden_units=args.num_hidden_units,
                              gamma=args.gamma,
                              max_episodes=args.max_episodes,
                              max_steps_per_episode=args.max_steps_per_episode,
                              learning_rate=args.learning_rate,
                              )

    # Train the Meta RL model
    MetaRLModel.train()
