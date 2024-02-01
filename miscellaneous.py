import json
import os
import sys
from types import SimpleNamespace

import numpy as np


# TODO possible split functions between different files if this file gets too long/general

def make_great_circle_points(start, end, n):
    """
    returns an array of n lon/lat pairs on great circle between (lon, lat) of start and end points
    Parameters
    ----------
    start : tuple
        (lon, lat) of start point
    end : tuple
        (lon, lat) of end point
    n : int
        number of points

    Returns
    -------
    ndarray
        of shape (2, n). lon/lat pairs of points on great circle
    ndarray
        of shape (n). corresponding distances of points along the great circle from start
    """
    from pyproj import Geod
    # TODO copy that web source and change n to desired km resolution? https://gis.stackexchange.com/questions/47/what-tools-in-python-are-available-for-doing-great-circle-distance-line-creati
    g = Geod(ellps='WGS84')
    _, _, dist = g.inv(*start, *end)
    distances = np.linspace(0, dist, n)
    great_circle = np.array(g.npts(*start, *end, n, initial_idx=0, terminus_idx=0)).T
    return great_circle, distances


def make_custom_traj(sample_points):
    """
    Returns an iris.analysis.Trajectory instance with sample points given by gc
    Parameters
    ----------
    sample_points : ndarray
        ndarray of shape (m, 2) containing the sample points for the trajectory instance.
        should be in format lon/lat

    Returns
    -------
    iris.analysis.Trajectory
    """
    from iris.analysis import trajectory
    waypoints = [{'grid_longitude': sample_points[0][0], 'grid_latitude': sample_points[0][1]},
                 {'grid_longitude': sample_points[-1][0], 'grid_latitude': sample_points[-1][1]}]
    traj = trajectory.Trajectory(waypoints, sample_count=sample_points.shape[0])
    # replace trajectory points which are equally spaced in lat/lon with great circle points
    for gcpoint, d in zip(sample_points, traj.sampled_points):
        d['grid_longitude'] = gcpoint[0]
        d['grid_latitude'] = gcpoint[1]

    return traj


def convert_to_ukv_coords(x, y, in_crs, out_crs):
    """transforms coordinates given in crs in_crs to coordinates in crs out_crs.
    works at least for UKV rotated pole."""
    out_x, out_y = out_crs.transform_point(x, y, in_crs)
    return out_x + 360, out_y


def convert_list_to_ukv_coords(x_list, y_list, in_crs, out_crs):
    """list version of convert_to_ukv_coords"""
    return np.array([convert_to_ukv_coords(x, y, in_crs, out_crs) for x, y in zip(x_list, y_list)])


def index_selector(desired_value, array):
    """returns the index of the value in array that is closest to desired_value"""
    return (np.abs(array - desired_value)).argmin()


def load_settings(file):
    """
    loads the settings for a TLW case from json file
    Parameters
    ----------
    file : str
        the .json file

    Returns
    -------
    SimpleNamespace
        containing settings
    """
    # TODO get rid of namespace globals().update(settings)
    with open(file) as f:
        settings = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    return settings


def get_datetime_from_settings(settings):
    return f'{settings.year}-{settings.month:02d}-{settings.day:02d}_{settings.h}'


def check_argv_num(argv, num_args, message=None):
    """
    Checks whether the number of arguments given through the command line is the correct number for the file
    Parameters
    ----------
    argv : list
        sys.argv
    num_args : int
        the required number of arguments for the file
    message : str
        message to append to Exception raised

    Returns
    -------

    """
    text = f'Gave {len(sys.argv) - 1} arguments but this file takes exactly {num_args} '
    if message is not None:
        text += message

    if len(argv) - 1 != num_args:
        raise Exception(text)


def make_title_and_save_path(datetime, region, data_source_string, test, smoothed=False, mag_filter=False, k2=False, k3=False, use_sim_sat=False):
    my_title = f'{datetime}_{region}_{data_source_string}'

    save_path = f'plots/{datetime}/{region}/'

    if not os.path.exists('plots/' + datetime):
        os.makedirs('plots/' + datetime)

    if test:
        save_path = f'plots/test/'
        my_title += '_test'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if use_sim_sat:
        save_path += 'radsim_'
        my_title += '_radsim'

    if k2:
        save_path += 'k2_'
        my_title += '_k2'
    elif k3:
        save_path += 'k3_'
        my_title += '_k3'
    if smoothed:
        save_path += 'smoothed_'
        my_title += '_smoothed'
    if mag_filter:
        save_path += 'magfiltered_'
        my_title += '_magfiltered'

    return my_title, save_path


def create_bins(range, bin_width):
    """
    Creates edges of bins and mid-bin values within a range given an approximate bin width.
    Parameters
    ----------
    range : tuple
        lower and upper bounds of the range
    bin_width : float


    Returns
    -------

    """
    bins = np.linspace(range[0], range[1], int(np.ceil((range[1] - range[0]) / bin_width) + 1))
    vals = 0.5 * (bins[1:] + bins[:-1])
    return bins, vals


def create_bins_from_midpoints(midpoints_array):
    """

    Parameters
    ----------
    midpoints_array : np.array
        requires evenly spaced points in the array

    Returns
    -------

    """
    bin_width = midpoints_array[1] - midpoints_array[0]
    bins = np.linspace(midpoints_array[0] - bin_width/2, midpoints_array[-1] + bin_width / 2, len(midpoints_array) + 1)
    return bins


def create_range_and_bin_edges_from_minmax(minmax, n):
    range = np.linspace(minmax[0], minmax[1], n)
    diff = range[1] - range[0]
    bin_edges = np.linspace(minmax[0] - diff/2, minmax[1] + diff/2, n+1)
    return range, bin_edges


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def argmax_lastNaxes(A, N):
    """
    finds the indices of the maxima along the last N axes of array A
    Parameters
    ----------
    A
    N

    Returns
    -------

    """
    s = A.shape
    new_shp = s[:-N] + (np.prod(s[-N:]),)
    max_idx = A.reshape(new_shp).argmax(-1)
    return np.unravel_index(max_idx, s[-N:])


def log_spaced_lambda(minmax, factor):
    n = int(np.round(1 + np.log(minmax[1] / minmax[0]) / np.log(factor)))
    range, bin_edges = create_range_and_bin_edges_from_minmax(np.log(minmax), n)
    return np.exp(range), np.exp(bin_edges)


def k_spaced_lambda(minmax, n):
    range, bin_edges = create_range_and_bin_edges_from_minmax(2 * np.pi / np.array(minmax), n)
    return 2 * np.pi / range, 2 * np.pi / bin_edges