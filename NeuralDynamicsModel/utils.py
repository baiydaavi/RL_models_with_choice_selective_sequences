"""This module defines the utility functions."""
import numpy as np

def softmax(x):
    """Softmax function"""
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)

def normalized_gaussian(x, mu, sig):
    """Function defining normalized gaussian
    Args:
        x: input array for the gaussian
        mu: mean of the gaussian
        sig: standard deviation of the gaussian

    Returns: normalized gaussian array
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * (
                1 / np.sqrt(2 * np.pi * np.power(sig, 2.)))
