"""This module defines the utility functions."""
import numpy as np


def softmax(x):
    """Softmax function

    Args:
        x (np.ndarray): array of x values

    Returns:
        np.ndarray: softmax output
    """

    exps = np.exp(x)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)


def normalized_gaussian(x, mu, sig):
    """Normalized gaussian function.

    Args:
        x (np.ndarray): Input array for the gaussian.
        mu (float): Mean of the gaussian.
        sig (float): Standard deviation of the gaussian.

    Returns:
        np.ndarray: Normalized gaussian output.
    """

    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0))) * (
        1 / np.sqrt(2 * np.pi * np.power(sig, 2.0))
    )
