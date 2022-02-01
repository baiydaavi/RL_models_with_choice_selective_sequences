"""This module defines utility functions for training."""
import tensorflow as tf


def get_expected_return(rewards, gammas):
    """Compute expected returns per time step.

    Args:
        rewards (tf.Tensor): List of reward received during an episode.
        gammas (tf.Tensor): List of discount factor values over an episode.

    Returns:
        tf.Tensor: List of expected returns for each time step.
    """

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    gammas = tf.cast(gammas[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape

    for i in tf.range(n):
        reward = rewards[i]
        gamma_val = gammas[i]
        discounted_sum = reward + gamma_val * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)

    returns = returns.stack()[::-1]
    return returns


def compute_loss_policy(
    action_probs,
    probs,
    advantages,
    entropy_coef=0.05,
):
    """Computes the actor loss.

    Args:
        action_probs (tf.Tensor): List of choice probabilities for the chosen action.
        probs (tf.Tensor): List of choice probabilities for all actions.
        advantages (tf.Tensor): List of advantages.
        entropy_coef (float, optional): Weight of entropy loss. Defaults to 0.05.

    Returns:
        tf.Tensor: Actor loss.
    """

    policy_loss = -tf.math.reduce_sum(tf.math.log(action_probs) * advantages)
    entropy_loss = -tf.math.reduce_sum(probs * tf.math.log(probs))
    return policy_loss - entropy_coef * entropy_loss


def compute_loss_value(values, returns, value_coef=0.05):
    """Computes the critic loss.

    Args:
        values (tf.Tensor): List of values.
        returns (tf.Tensor): List of expected returns.
        value_coef (float, optional): Weight of value loss. Defaults to 0.05.

    Returns:
        tf.Tensor: Critic loss.
    """

    val_err = returns - values
    value_loss = 0.5 * tf.math.reduce_sum(tf.math.square(val_err))
    return value_coef * value_loss
