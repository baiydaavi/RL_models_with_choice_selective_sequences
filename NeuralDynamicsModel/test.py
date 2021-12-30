import argparse

import numpy as np

from MetaRLModel import MetaRLModel
from environment import TestingEnvironment
from model_arch import ActorCriticLSTM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing')
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
    parser.add_argument('--reward_width', type=int, default=0.3,
                        help='width of the Gaussian reward input')
    parser.add_argument('--input_type', type=str, default='sequential',
                        help='choice-selective input type to critic - ['
                             'sequential, persistent]')
    parser.add_argument('--block_type', default='default', type=str,
                        help="block type - ['default', 'fixed']")
    parser.add_argument('--mode', default='default', type=str,
                        help="mode - ['default', 'optogenetic']")
    parser.add_argument('--max_steps_per_episode', type=int, default=1500 * 42,
                        help='number of steps per each episode')
    parser.add_argument('--fraction_stimulated', type=float, default=0.7,
                        help='fraction of neurons that are stimulated')
    parser.add_argument('--stimulation_level', type=float, default=0.15,
                        help='amount of stimulation')
    parser.add_argument('--run_id', type=int, default=0,
                        help='id of the current test run')
    parser.add_argument('--saved_model_path', type=str,
                        default='saved_model/model-62000',
                        help='path of the saved model')
    args = parser.parse_args()

    # Network
    model_architecture = ActorCriticLSTM(args.num_actions,
                                         args.num_hidden_units, args.gamma)

    # RL environment
    environment = TestingEnvironment(args.reward_width,
                                     block_type=args.block_type,
                                     mode=args.mode)

    # Choice-selective input activity
    choice_selective_activity = np.load(
        f"recorded_data/{args.input_type}_activity.npz"
    )

    # Meta RL Model
    MetaRLModel = MetaRLModel(model_architecture,
                              environment,
                              choice_selective_activity,
                              input_type=args.input_type,
                              num_actions=args.num_actions,
                              num_states=args.num_states,
                              num_hidden_units=args.num_hidden_units,
                              gamma=args.gamma,
                              max_steps_per_episode=args.max_steps_per_episode,
                              learning_rate=args.learning_rate,
                              )

    # folder to save testing data
    test_folder = f"testing_data/block_type-{args.block_type}/" \
                  f"mode-{args.mode}/"

    # Test the Meta RL model
    MetaRLModel.test(
        save_destination_folder=test_folder,
        run_id=args.run_id,
        load_model=args.saved_model_path,
        fraction_stimulated=args.fraction_stimulated,
        stimulation_level=args.stimulation_level,
    )
