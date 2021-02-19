"""
Normalisation for SAXS data used for machine learning applications
"""
import numpy as np


def normalise(Is, subtractor=None, divisor=None):
    """Normalise data as log(x) - <log(x)> / std(log(X)"""
    Is = np.log10(np.array(Is) + 1.0)
    if subtractor is None: subtractor = np.mean(Is, axis=0)
    Is = Is - subtractor
    if divisor is None: divisor = np.std(Is, axis=0)
    Is = np.divide(Is, divisor)
    where_are_NaNs = np.isnan(Is)
    Is[where_are_NaNs] = 0.0
    return Is, subtractor, divisor


def unnormalise(Is, mean, std):
    """
    Used for autoencoders.
    Mean and std arrays must be the same as used for data
    normalisation to properly restore your data!
    """
    # if len(np.shape(Is)) == 2:
    #     for num1, I in enumerate(Is):
    #         for num2, v in enumerate(I):
    #             if num2 <= 128: Is[num1, num2] = v
    # if len(np.shape(Is)) == 1:
    #     for num, I in enumerate(Is):
    #         if num <= 128: Is[num] = I
    Is = np.multiply(Is, std)
    Is += mean
    Is = 10 ** (Is + 1.0)
    return Is
