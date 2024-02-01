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


def find_array_max_idxs(a):
    return np.unravel_index(a.argmax(), a.shape)


def find_2d_peak_width_idxs(a, peak_idxs, height=0.5):

    thresh = a[*peak_idxs] * height

    x_lowererr_idx, x_uppererr_idx = peak_idxs[0], peak_idxs[0]
    y_lowererr_idx, y_uppererr_idx = peak_idxs[1], peak_idxs[1]

    while True:
        try:
            if a[x_uppererr_idx + 1, peak_idxs[1]] > thresh:
                x_uppererr_idx += 1
            else:
                break
        except IndexError:
            break

    while True:
        try:
            if a[x_lowererr_idx - 1, peak_idxs[1]] > thresh:
                x_lowererr_idx -= 1
            else:
                break
        except IndexError:
            break

    while True:
        try:
            if a[peak_idxs[0], y_uppererr_idx + 1] > thresh:
                y_uppererr_idx += 1
            else:
                break
        except IndexError:
            break

    while True:
        try:
            if a[peak_idxs[0], y_lowererr_idx - 1] > thresh:
                y_lowererr_idx -= 1
            else:
                break
        except IndexError:
            break

    return [[x_lowererr_idx, x_uppererr_idx], [y_lowererr_idx, y_uppererr_idx]]


def find_polar_max_and_error(polar_array, lambdas, thetas, height=0.5):
    max_idxs = find_array_max_idxs(polar_array)

    # tile so that errors can wrap around 180-0 degrees
    tiled_array = np.tile(polar_array, 3)
    tiled_max_idxs = [max_idxs[0], max_idxs[1] + polar_array.shape[1]]
    error_idxs = find_2d_peak_width_idxs(tiled_array, tiled_max_idxs, height=height)

    for i in range(2):
        error_idxs[1][i] %= polar_array.shape[1]

    return lambdas[max_idxs[0]], thetas[max_idxs[1]], lambdas[error_idxs[0]], thetas[error_idxs[1]]
