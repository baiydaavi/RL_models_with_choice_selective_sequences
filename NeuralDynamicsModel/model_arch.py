"""This module defines the actor-critic model architecture."""

import tensorflow as tf
from tensorflow.keras import layers


class ActorCriticLSTM(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
        self,
        num_actions=3,
        num_hidden_units=128,
        gamma=0.96,
    ):
        """Initializes the actor-critic network.

        Args:
            num_actions (int, optional): Number of allowed actions. Defaults to 3.
            num_hidden_units (int, optional): Number of hidden units. Defaults to 128.
            gamma (float, optional): Discount factor. Defaults to 0.96.
        """
        super().__init__()

        self.actor_LSTM = layers.LSTM(
            num_hidden_units, return_state=True, return_sequences=True
        )
        self.critic_LSTM = layers.LSTM(
            num_hidden_units, return_state=True, return_sequences=True
        )
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

        self.gamma = gamma

    def call(self, input_critic, input_actor, critic_lstm_state, actor_lstm_state):
        """Calls the network.

        Args:
            input_critic (tf.Tensor): Input to the critic.
            input_actor (tf.Tensor): Input to the actor.
            critic_lstm_state (tf.Tensor): State of the critic LSTM.
            actor_lstm_state (tf.Tensor): State of the actor LSTM.

        Returns:
            Actor output, critic output, RPE generated, and LSTM states.
        """

        critic_LSTM_output, critic_state_h, critic_state_c = self.critic_LSTM(
            input_critic, initial_state=critic_lstm_state
        )

        current_time_value = self.critic(critic_LSTM_output)

        # calculating the RPE input using the reward input, the current time
        # value and the previous time value
        rpe_inp = (
            tf.expand_dims(input_critic[:, :, -1], 2)
            + tf.expand_dims(input_actor[:, :, -1], 2) * current_time_value
            - tf.expand_dims(input_actor[:, :, -2], 2)
        )

        # appending the RPE input to the actor input
        input_actor = tf.concat([input_actor[:, :, :-2], rpe_inp], 2)

        actor_LSTM_output, actor_state_h, actor_state_c = self.actor_LSTM(
            input_actor, initial_state=actor_lstm_state
        )

        return (
            self.actor(actor_LSTM_output),
            current_time_value,
            rpe_inp,
            critic_state_h,
            critic_state_c,
            actor_state_h,
            actor_state_c,
        )
