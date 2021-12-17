import numpy as np

def gaussian(x, mu, sig):
    """defining a gaussian function with 3 inputs - 'mu' is the values
    corresponding to peak firing, sig is the values of standard deviation x is
    the vector of values on which the gaussian is calculated"""
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def gaussian_normalised(x, mu, sig):
    """defining a normalised gaussian function with 3 inputs - 'mu' is the
    values corresponding to peak firing, sig is the values of standard
    deviation x is the vector of values on which the gaussian is calculated"""
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0))) * (
            1 / np.sqrt(2 * np.pi * np.power(sig, 2.0))
    )