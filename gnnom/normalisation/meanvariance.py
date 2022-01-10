"""
Normalisation for SAXS data used for machine learning applications
"""
import numpy as np


def normalise(Is, divisor=None, subtractor=None):
    """Normalise data as x - <x> / Var(x)"""
    Is = np.array(Is)
    if subtractor is None: subtractor = np.mean(Is, axis=0)
    Is = Is - subtractor
    if divisor is None: divisor = np.var(Is, axis=0)
    Is = np.divide(Is, divisor)
    where_are_NaNs = np.isnan(Is)
    Is[where_are_NaNs] = 0.0
    where_are_Infs = np.isinf(Is)
    Is[where_are_Infs] = 0.0
    return Is, divisor, subtractor


def unnormalise(Is, divisor, subtractor):
    """
    Used for autoencoders.
    Mean and std arrays must be the same as used for data
    normalisation to properly restore your data!
    """
    Is = np.multiply(Is, divisor)
    Is += subtractor
    return Is
