import numpy as np

from miscellaneous import argmax_lastNaxes


def max_lambda_theta(power_spectrum, lambdas, thetas):
    """
    Assumes power_spectrum is of format (image_dim1, image_dim2, lambda, theta)
    Parameters
    ----------
    power_spectrum
    lambdas
    thetas

    Returns
    -------

    """
    posvect_pspec_max = argmax_lastNaxes(power_spectrum, 2)
    dom_lambda = lambdas[np.moveaxis(posvect_pspec_max, 0, -1)[..., 0]]
    dom_theta = thetas[np.moveaxis(posvect_pspec_max, 0, -1)[..., 1]]

    return dom_lambda, dom_theta
