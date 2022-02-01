import numpy as np


def gaussian(x, mu, sig):
    """Gaussian function.

    Args:
        x (np.ndarray): Input array for the gaussian.
        mu (float): Mean of the gaussian.
        sig (float): Standard deviation of the gaussian.

    Returns:
        np.ndarray: Gaussian output.
    """

    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def gaussian_normalised(x, mu, sig):
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
