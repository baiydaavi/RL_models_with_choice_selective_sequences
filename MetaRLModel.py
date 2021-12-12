"""This module defines the meta reinforcement learning model."""

import tensorflow as tf
import numpy as np
import os
import tqdm

from training_utils import get_expected_return, compute_loss_policy, \
    compute_loss_value
from utils import create_gif, make_gif


class MetaRLModel:
    """Meta-RL model"""

    def __init__(
            self,
            model,
            env,
            choice_selective_activity,
            num_actions=3,
            num_states=42,
            num_hidden_units=128,
            gamma=0.96,
            max_episodes=80001,
            max_steps_per_episode=630,
            learning_rate=0.001,
    ):

        self.model = model
        self.env = env
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_hidden_units = num_hidden_units
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode

        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=self.learning_rate)

        self.choice_selective_activity = choice_selective_activity
        self.left_choice_activity = self.choice_selective_activity['left']
        self.right_choice_activity = self.choice_selective_activity['right']

        self.model_path, self.frame_path = self.training_saver()

    def learn(self, load_model=None):

        if load_model:
            self.model.load_weights(load_model)

        with tqdm.trange(self.max_episodes) as t:
            for episode in t:
                self.env.reset()
                initial_action = np.random.choice([1, 2])
                initial_obs_state, initial_reward, _, _ = self.env.step(
                    initial_action)

                if initial_action == 1:
                    choice_act = tf.convert_to_tensor(
                        self.left_choice_activity[
                        self.env.rand_activity_trial_num, :, initial_obs_state],
                        dtype=tf.float32)
                else:
                    choice_act = tf.convert_to_tensor(
                        self.right_choice_activity[
                        self.env.rand_activity_trial_num, :, initial_obs_state],
                        dtype=tf.float32)

                initial_critic_obs = tf.concat([choice_act, [initial_reward]],
                                               0)
                initial_actor_obs = tf.concat([tf.one_hot(initial_action,
                                                          self.num_actions,
                                                          dtype=tf.float32),
                                               tf.one_hot(initial_obs_state,
                                                          self.num_states,
                                                          dtype=tf.float32),
                                               [0.0], [self.gamma]], 0)

                initial_state = [
                    tf.zeros([1, self.num_hidden_units], tf.float32),
                    tf.zeros([1, self.num_hidden_units], tf.float32)]

                # Run the model for one episode to collect training data
                critic_obs_seq, actor_obs_seq, actions, \
                current_episode_times, rew_probs, rewards, gammas, values, \
                rpes = self.train_episode(initial_critic_obs, initial_actor_obs,
                                          initial_state)

                returns = get_expected_return(
                    tf.concat([rewards, [values[-1]]], 0),
                    tf.concat([gammas, [1.0]], 0))[:-1]

                advantages = returns - values

                episode_loss = self.train_step(critic_obs_seq, actor_obs_seq,
                                               actions, returns, advantages,
                                               initial_state)

                episode_reward = tf.math.reduce_sum(rewards)

                t.set_description(f'Episode {episode}')
                t.set_postfix(
                    episode_reward=episode_reward, episode_loss=episode_loss)

                # Save model weights and create GIF depicting model behavior
                if episode % 2000 == 0:
                    self.model.save_weights(
                        self.model_path + '/model-' + str(
                            episode) + '/model-' + str(episode))

                    episode_frames = [rewards.numpy(), rew_probs.numpy(),
                                      actions.numpy(),
                                      current_episode_times.numpy()]
                    img_list = create_gif(episode_frames)
                    ep_type = "/train_"
                    make_gif(img_list,
                             self.frame_path + ep_type + str(episode) + '.gif',
                             duration=len(img_list) * 0.1, true_image=True)

    def train_episode(
            self,
            in_critic_obs,
            in_actor_obs,
            initial_state,
    ):
        """Runs a single episode to collect training data."""

        critic_obs_seq = tf.TensorArray(dtype=tf.float32, size=0,
                                        dynamic_size=True)
        actor_obs_seq = tf.TensorArray(dtype=tf.float32, size=0,
                                       dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        rew_probs = tf.TensorArray(dtype=tf.float32, size=0,
                                   dynamic_size=True)
        current_episode_times = tf.TensorArray(dtype=tf.int32, size=0,
                                               dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0,
                                 dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rpes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        gammas = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        critic_obs = in_critic_obs
        actor_obs = in_actor_obs
        state_critic = initial_state
        state_actor = initial_state

        num_neurons = self.left_choice_activity.shape[1]

        for step_num in tf.range(self.max_steps_per_episode):

            critic_obs_seq = critic_obs_seq.write(step_num, critic_obs)
            actor_obs_seq = actor_obs_seq.write(step_num, actor_obs)

            # Convert state into a batched tensor (batch size = 1)
            critic_obs = tf.expand_dims(tf.expand_dims(critic_obs, 0), 0)
            actor_obs = tf.expand_dims(tf.expand_dims(actor_obs, 0), 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value, rpe, critic_temp_h, critic_temp_c, \
            actor_temp_h, actor_temp_c = self.model(
                critic_obs, actor_obs, state_critic, state_actor)

            value = tf.squeeze(value)

            # LSTM state for the next time step
            state_critic = [critic_temp_h, critic_temp_c]

            state_actor = [actor_temp_h, actor_temp_c]

            # Sample next action from the action probability distribution
            action = tf.random.categorical(
                tf.expand_dims(tf.squeeze(action_logits_t), 0), 1)[0, 0]

            # Apply action to the environment to get next state and reward
            obs_state, reward, _, current_episode_time = self.env.step(
                action)

            gamma_val = self.gamma

            if obs_state == 0:
                gamma_val = 0.0

            if self.env.choice == 1:
                choice_act = tf.convert_to_tensor(
                    self.left_choice_activity[
                    self.env.rand_activity_trial_num, :,
                    obs_state],
                    dtype=tf.float32)

            elif self.env.choice == 0:
                choice_act = tf.convert_to_tensor(
                    self.right_choice_activity[
                    self.env.rand_activity_trial_num, :,
                    obs_state],
                    dtype=tf.float32)

            else:
                choice_act = tf.zeros(num_neurons, tf.float32)

            gammas = gammas.write(step_num, gamma_val)

            critic_obs = tf.concat([choice_act, [reward]], 0)
            actor_obs = tf.concat(
                [tf.one_hot(action, self.num_actions, dtype=tf.float32),
                 tf.one_hot(obs_state, self.num_states, dtype=tf.float32),
                 [value], [gamma_val]], 0)

            # Store reward
            values = values.write(step_num, value)
            rpes = rpes.write(step_num, tf.squeeze(rpe))
            rewards = rewards.write(step_num, reward)

            # Store action
            actions = actions.write(step_num, tf.cast(action, tf.int32))

            # Store current_episode_time
            current_episode_times = current_episode_times.write(
                step_num,
                current_episode_time
            )

            # Store reward probability
            rew_probs = rew_probs.write(step_num, self.env.reward_prob)

        actor_obs_seq = actor_obs_seq.stack()
        crtic_obs_seq = critic_obs_seq.stack()
        actions = actions.stack()
        current_episode_times = current_episode_times.stack()
        rew_probs = rew_probs.stack()
        rewards = rewards.stack()
        gammas = gammas.stack()
        values = values.stack()
        rpes = rpes.stack()

        return crtic_obs_seq, actor_obs_seq, actions, \
               current_episode_times, rew_probs, rewards, gammas, \
               values, rpes

    def test(self, run_id=0, load_model=None,
             block_type='default', mode='default',
             fraction_stimulated=0.6, stimulation_level=0.15):
        """Run model test."""

        save_destination_folder = f"testing_data/block_type-{block_type}/" \
                                  f"mode-{mode}/"

        os.makedirs(save_destination_folder, exist_ok = True)

        file_save_id = f"test_{run_id}.npz"

        if load_model:
            self.model.load_weights(load_model)

        self.env.reset()
        initial_action = np.random.choice([1, 2])
        initial_obs_state, initial_reward, _, _ = self.env.step(
            initial_action)

        if initial_action == 1:
            choice_act = tf.convert_to_tensor(
                self.left_choice_activity[
                self.env.rand_activity_trial_num, :, initial_obs_state],
                dtype=tf.float32)
        else:
            choice_act = tf.convert_to_tensor(
                self.right_choice_activity[
                self.env.rand_activity_trial_num, :, initial_obs_state],
                dtype=tf.float32)

        initial_critic_obs = tf.concat([choice_act, [initial_reward]],
                                       0)
        initial_actor_obs = tf.concat([tf.one_hot(initial_action,
                                                  self.num_actions,
                                                  dtype=tf.float32),
                                       tf.one_hot(initial_obs_state,
                                                  self.num_states,
                                                  dtype=tf.float32),
                                       [0.0], [self.gamma]], 0)

        initial_state = [
            tf.zeros([1, self.num_hidden_units], tf.float32),
            tf.zeros([1, self.num_hidden_units], tf.float32)]

        # Run the model for one episode to collect testing data
        (
            current_trial_times,
            trial_count_in_blocks,
            actions,
            choices,
            actor_logits,
            reward_probs,
            rewards,
            is_rewardeds,
            values,
            rpes,
            critic_hs,
            critic_cs,
            actor_hs,
            actor_cs,
            is_stimulateds,
        ) = self.test_episode(initial_critic_obs, initial_actor_obs,
                              initial_state,
                              fraction_stimulated=fraction_stimulated,
                              stimulation_level=stimulation_level)

        reward_probs = 2 * np.argmax(reward_probs, axis=1) - 1

        np.savez_compressed(
            save_destination_folder + file_save_id,
            current_trial_times=current_trial_times,
            trial_count_in_blocks=trial_count_in_blocks,
            actions=actions,
            choices=choices,
            actor_logits=actor_logits,
            reward_probs=reward_probs,
            rewards=rewards,
            is_rewardeds=is_rewardeds,
            values=values,
            rpes=rpes,
            critic_hs=critic_hs,
            critic_cs=critic_cs,
            actor_hs=actor_hs,
            actor_cs=actor_cs,
            is_stimulateds=is_stimulateds,
        )

        print(f"saved_file for run={run_id}")

    def test_episode(
            self,
            in_critic_obs,
            in_actor_obs,
            initial_state,
            fraction_stimulated=0.6,
            stimulation_level=0.15,
    ):
        """Runs a single episode to collect training data."""

        current_trial_times = []
        trial_count_in_blocks = []
        actions = []
        choices = []
        actor_logits = []
        reward_probs = []
        rewards = []
        is_rewardeds = []
        values = []
        rpes = []
        critic_hs = []
        critic_cs = []
        actor_hs = []
        actor_cs = []
        is_stimulateds = []

        critic_obs = in_critic_obs
        actor_obs = in_actor_obs
        state_critic = initial_state
        state_actor = initial_state

        [critic_temp_h, critic_temp_c] = state_critic

        [actor_temp_h, actor_temp_c] = state_actor

        num_neurons = self.left_choice_activity.shape[1]

        for _ in tqdm.tqdm(range(self.max_steps_per_episode)):

            # Convert state into a batched tensor (batch size = 1)
            critic_obs = tf.expand_dims(tf.expand_dims(critic_obs, 0), 0)
            actor_obs = tf.expand_dims(tf.expand_dims(actor_obs, 0), 0)

            critic_hs.append(tf.squeeze(critic_temp_h).numpy())
            critic_cs.append(tf.squeeze(critic_temp_c).numpy())

            actor_hs.append(tf.squeeze(actor_temp_h).numpy())
            actor_cs.append(tf.squeeze(actor_temp_c).numpy())

            # Run the model and to get action probabilities and critic value
            action_logits_t, value, rpe, critic_temp_h, critic_temp_c, \
            actor_temp_h, actor_temp_c = self.model(
                critic_obs, actor_obs, state_critic, state_actor)

            actor_logits.append(tf.squeeze(action_logits_t).numpy())
            value = tf.squeeze(value)

            # LSTM state for the next time step
            state_critic = [critic_temp_h, critic_temp_c]

            state_actor = [actor_temp_h, actor_temp_c]

            # Sample next action from the action probability distribution
            action = tf.random.categorical(
                tf.expand_dims(tf.squeeze(action_logits_t), 0), 1
            )[0, 0]

            # Apply action to the environment to get next state and reward
            obs_state, reward, is_rewarded, trial_count_in_block = \
                self.env.step(
                    action)

            gamma_val = self.gamma

            if obs_state == 0:
                gamma_val = 0.0

            if self.env.choice == 1:
                if self.env.is_stimulated == 1:
                    choice_act = self.left_choice_activity[
                                 self.env.rand_activity_trial_num, :,
                                 obs_state]
                    stimulated_neurons = np.random.choice(
                        num_neurons, int(fraction_stimulated * num_neurons),
                        replace=False
                    )
                    choice_act[stimulated_neurons] = stimulation_level
                    choice_act = tf.convert_to_tensor(choice_act,
                                                      dtype=tf.float32)
                else:
                    choice_act = tf.convert_to_tensor(
                        self.left_choice_activity[
                        self.env.rand_activity_trial_num, :,
                        obs_state],
                        dtype=tf.float32)

            elif self.env.choice == 0:
                if self.env.is_stimulated == 1:
                    choice_act = self.right_choice_activity[
                                 self.env.rand_activity_trial_num, :,
                                 obs_state]
                    stimulated_neurons = np.random.choice(
                        num_neurons, int(fraction_stimulated * num_neurons),
                        replace=False
                    )
                    choice_act[stimulated_neurons] = stimulation_level
                    choice_act = tf.convert_to_tensor(choice_act,
                                                      dtype=tf.float32)
                else:
                    choice_act = tf.convert_to_tensor(
                        self.right_choice_activity[
                        self.env.rand_activity_trial_num, :,
                        obs_state],
                        dtype=tf.float32)

            else:
                choice_act = tf.zeros(num_neurons, tf.float32)

            critic_obs = tf.concat([choice_act, [reward]], 0)
            actor_obs = tf.concat(
                [tf.one_hot(action, self.num_actions, dtype=tf.float32),
                 tf.one_hot(obs_state, self.num_states, dtype=tf.float32),
                 [value], [gamma_val]], 0)

            trial_count_in_blocks.append(trial_count_in_block)
            current_trial_times.append(obs_state)
            choices.append(self.env.choice)
            actions.append(tf.cast(action, tf.int32).numpy())
            values.append(value.numpy())
            rpes.append(tf.squeeze(rpe).numpy())
            rewards.append(reward)
            reward_probs.append(self.env.reward_prob)
            is_rewardeds.append(is_rewarded)
            is_stimulateds.append(self.env.is_stimulated)

        trial_count_in_blocks = np.array(trial_count_in_blocks)
        current_trial_times = np.array(current_trial_times)
        actions = np.array(actions)
        choices = np.array(choices)
        actor_logits = np.array(actor_logits)
        reward_probs = np.array(reward_probs)
        rewards = np.array(rewards)
        is_rewardeds = np.array(is_rewardeds)
        values = np.array(values)
        rpes = np.array(rpes)
        critic_hs = np.array(critic_hs)
        critic_cs = np.array(critic_cs)
        actor_hs = np.array(actor_hs)
        actor_cs = np.array(actor_cs)
        is_stimulateds = np.array(is_stimulateds)

        return (
            current_trial_times,
            trial_count_in_blocks,
            actions,
            choices,
            actor_logits,
            reward_probs,
            rewards,
            is_rewardeds,
            values,
            rpes,
            critic_hs,
            critic_cs,
            actor_hs,
            actor_cs,
            is_stimulateds,
        )

    @tf.function
    def train_step(
            self,
            critic_obs: tf.Tensor,
            actor_obs: tf.Tensor,
            actions: tf.Tensor,
            returns: tf.Tensor,
            advantages: tf.Tensor,
            initial_state: tf.Tensor) -> tf.Tensor:
        """Runs a model training step."""

        critic_obs = tf.expand_dims(critic_obs, 0)
        actor_obs = tf.expand_dims(actor_obs, 0)

        with tf.GradientTape() as tape:
            logits, values, _, _, _, _, _ = self.model(critic_obs, actor_obs,
                                                       initial_state,
                                                       initial_state)

            logits = tf.squeeze(logits)
            values = tf.squeeze(values)

            probs = tf.nn.softmax(logits)
            action_probs = tf.TensorArray(dtype=tf.float32, size=0,
                                          dynamic_size=True)
            for item in tf.range(len(actions)):
                action_probs = action_probs.write(item,
                                                  probs[item, actions[item]])
            action_probs = action_probs.stack()

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns, advantages = [
                tf.expand_dims(x, 1) for x in
                [action_probs, values, returns, advantages]]

            # Calculating loss values to update our network
            loss = compute_loss_policy(action_probs, probs,
                                       advantages) + compute_loss_value(values,
                                                                        returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, grad_norms = tf.clip_by_global_norm(grads, 999.0)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss

    def load_weights(self, load_model=None):
        if load_model:
            self.model.load_weights(load_model)

        self.env.reset()
        initial_action = np.random.choice([1, 2])
        initial_obs_state, initial_reward, _, _ = self.env.step(
            initial_action)

        if initial_action == 1:
            choice_act = tf.convert_to_tensor(
                self.left_choice_activity[
                self.env.rand_activity_trial_num, :, initial_obs_state],
                dtype=tf.float32)
        else:
            choice_act = tf.convert_to_tensor(
                self.right_choice_activity[
                self.env.rand_activity_trial_num, :, initial_obs_state],
                dtype=tf.float32)

        initial_critic_obs = tf.concat([choice_act, [initial_reward]],
                                       0)
        initial_actor_obs = tf.concat([tf.one_hot(initial_action,
                                                  self.num_actions,
                                                  dtype=tf.float32),
                                       tf.one_hot(initial_obs_state,
                                                  self.num_states,
                                                  dtype=tf.float32),
                                       [0.0], [self.gamma]], 0)

        initial_state = [
            tf.zeros([1, self.num_hidden_units], tf.float32),
            tf.zeros([1, self.num_hidden_units], tf.float32)]

        # Convert state into a batched tensor (batch size = 1)
        critic_obs = tf.expand_dims(tf.expand_dims(initial_critic_obs, 0), 0)
        actor_obs = tf.expand_dims(tf.expand_dims(initial_actor_obs, 0), 0)

        # Call the model once
        self.model(critic_obs, actor_obs, initial_state, initial_state)

        return {'ActorOutput': self.model.actor.get_weights(),
                'ActorLSTM': self.model.actor_LSTM.get_weights(),
                'CriticOutput': self.model.critic.get_weights(),
                'CriticLSTM': self.model.critic_LSTM.get_weights()}

    def training_saver(self):

        dir_name = f'training_data/learning_rate={self.learning_rate}' \
                   f'_hidden_units={self.num_hidden_units}'

        model_path = dir_name + '/model'
        frame_path = dir_name + '/frames'

        # create the directories
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(frame_path, exist_ok=True)

        return model_path, frame_path
