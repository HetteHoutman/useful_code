import numpy as np
import pyproj
from matplotlib import pyplot as plt
from scipy import stats
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.optimize import *
from skimage.morphology.footprints import ellipse
from skimage.transform import rotate


from miscellaneous import create_bins, pol2cart


def recip_space(Lx, Ly, shape):
    """
    Given the shape and physical x and y extent of an image, returns 2d grids of the reciprocal space of the image,
    given in both the x and y components of the wavevector, as well as their polar counterparts abs(k) and theta.
    Parameters
    ----------
    Lx : float
        physical extent of the image in the x direction in km.
    Ly : float
        physical extent of the image in the y direction in km.
    shape : tuple of ints
        shape of the image
    Returns
    -------
    K : ndarray
        2d of shape "shape" with the x components of the wavevector in km
    L : ndarray
        2d of shape "shape" with the y components of the wavevector in km
    dist_array : ndarray
        2d of shape "shape" with magnitudes of the wavevector in km (the wavenumber)
    thetas : ndarray
        2d of shape "shape" with angles of the wavevector in degrees
    """
    xlen = shape[1]
    ylen = shape[0]

    # np uses linear frequency f instead of angular frequency omega=2pi*f, so multiply by 2pi to get angular wavenum k
    k = 2 * np.pi * np.fft.fftfreq(xlen, d=Lx / xlen)
    l = 2 * np.pi * np.fft.fftfreq(ylen, d=Ly / ylen)

    # do fft shift
    K, L = np.meshgrid(np.roll(k, k.shape[0] // 2), np.roll(l, l.shape[0] // 2)[::-1])

    dist_array = np.sqrt(K ** 2 + L ** 2)
    thetas = np.rad2deg(np.arctan2(K, L)) % 180
    return K, L, dist_array, thetas


def ideal_bandpass(ft, Lx, Ly, low, high):
    """
    Passes a 2d fourier transform through an ideal bandpass between low and high values of the wavenumber, given the
    physical extent of the image.
    Parameters
    ----------
    ft : ndarray
        the 2d fourier transformed image
    Lx : float
        the physical extent of the original image in the x direction in km.
    Ly : float
        the physical extent of the original image in the y direction in km.
    low : float
        lower value of the passband in inverse km
    high : float
        higher value of the passband in inverse km.

    Returns
    -------
    masked : np.ma.MaskedArray
        array with values outside the passband masked
    """
    _, _, dist_array, thetas = recip_space(Lx, Ly, ft.shape)

    low_mask = (dist_array < low)
    high_mask = (dist_array > high)
    masked = np.ma.masked_where(low_mask | high_mask, ft)

    return masked


def extract_distances(lats, lons):
    """
    Extracts the approximate physical extent of an image given arrays of latitude and longitude values
    Parameters
    ----------
    lats : list or ndarray or tuple
        in which the first and last elements of the list are the extreme latitude values
    lons : list or ndarray or tuple
        in which the first and last elements of the list are the extreme longitude values

    Returns
    -------

    """
    g = pyproj.Geod(ellps='WGS84')
    _, _, Lx = g.inv(lons[0], lats[lats.shape[0] // 2],
                     lons[-1], lats[lats.shape[0] // 2])
    _, _, Ly = g.inv(lons[lons.shape[0] // 2], lats[0],
                     lons[lons.shape[0] // 2], lats[-1])

    return Lx / 1000, Ly / 1000


def make_stripes(X, Y, wavelength, angle, wiggle=0, wiggle_wavelength=0):
    angle += 90
    angle = np.deg2rad(angle)
    rot_x = X * np.cos(angle) + Y * np.sin(angle)
    rot_y = X * np.sin(-angle) + Y * np.cos(angle)
    wiggle *= np.sin(2 * np.pi * rot_y / wiggle_wavelength)
    return np.sin((2 * np.pi * rot_x + wiggle) / wavelength)


def stripey_test(orig, Lx, Ly, wavelens, angles, wiggle=0, wiggle_wavelength=0):
    """
    Returns an image containing pure sine waves of chosen wavelength and angle with the same shape and value range as
    the given original image.
    Parameters
    ----------
    orig : ndarray
        the original image
    Lx : float
        the physical extent of the image in the x direction in km.
    Ly : float
        the physical extent of the image in the y direction in km.
    wavelens : list of floats
        containing the wavelength of the desired sine waves in km.
    angles : list of float
        the angles of each of the wavelengths given by wavelens in degrees from north.

    Returns
    -------
    total : ndarray


    """
    x = np.linspace(-Lx / 2, Lx / 2, orig.shape[1])
    y = np.linspace(-Ly / 2, Ly / 2, orig.shape[0])
    X, Y = np.meshgrid(x, y)
    total = np.zeros(X.shape)

    for wavelen, angle in zip(wavelens, angles):
        total += make_stripes(X, Y, wavelen, angle, wiggle=wiggle, wiggle_wavelength=wiggle_wavelength)

    # this ensures the stripes are roughly in the same range as the input data
    middle = (orig.max() + orig.min()) / 2
    total *= (orig.max() - orig.min()) / (total.max() - total.min())
    total += middle

    return total


def make_polar_pspec(pspec_2d, wavenumbers, wavenumber_bin_width, thetas, theta_bin_width):
    """
    Expresses a Cartesian power spectrum in polar coordinates by averaging it over bins of wavenumber and theta
    Parameters
    ----------
    pspec_2d : ndarray
        2d power spectrum
    wavenumbers : ndarray
        2d array of corresponding wavenumbers
    wavenumber_bin_width : float
    thetas : ndarray
        2d array of corresponding wavenumbers in degrees
    theta_bin_width : float

    Returns
    -------
    np.array(radial_pspec_array) : ndarray
        the power spectrum expressed in polar coordinates
    wnum_bins : ndarray
        bin edges
    wnum_vals : ndarray
        bin mid-point values
    idem for theta

    """

    wnum_bins, wnum_vals = create_bins((0, wavenumbers.max()), wavenumber_bin_width)
    theta_ranges, theta_vals = create_bins((-theta_bin_width / 2, 180 - theta_bin_width / 2), theta_bin_width)
    thetas_redefined = thetas.copy()
    thetas_redefined[(180 - theta_bin_width / 2 <= thetas_redefined) & (thetas_redefined < 180)] -= 180
    radial_pspec_array = []

    for i in range(len(theta_ranges) - 1):
        low_mask = (thetas_redefined >= theta_ranges[i])
        high_mask = (thetas_redefined < theta_ranges[i + 1])
        mask = (low_mask & high_mask)

        radial_pspec, _, _ = stats.binned_statistic(wavenumbers[mask].flatten(), pspec_2d[mask].flatten(),
                                                    statistic="mean",
                                                    bins=wnum_bins)
        radial_pspec *= np.pi * (wnum_bins[1:] ** 2 - wnum_bins[:-1] ** 2) * np.deg2rad(theta_bin_width)
        radial_pspec_array.append(radial_pspec)

    return np.array(radial_pspec_array), wnum_bins, wnum_vals, theta_ranges, theta_vals


def make_angular_pspec(pspec_2d: np.ma.masked_array, thetas, theta_bin_width, wavelengths, wavelength_ranges):
    """
    not used
    """
    theta_bins, theta_vals = create_bins((-theta_bin_width / 2, 180 - theta_bin_width / 2), theta_bin_width)
    thetas_redefined = thetas.copy()
    thetas_redefined[(180 - theta_bin_width / 2 <= thetas_redefined) & (thetas_redefined < 180)] -= 180
    ang_pspec_array = []
    for i in range(len(wavelength_ranges) - 1):
        low_mask = wavelengths >= wavelength_ranges[i]
        high_mask = wavelengths < wavelength_ranges[i + 1]
        mask = (low_mask & high_mask)

        ang_pspec, _, _ = stats.binned_statistic(thetas_redefined[mask].flatten(), pspec_2d.data[mask].flatten(),
                                                 statistic="mean",
                                                 bins=theta_bins)
        ang_pspec *= np.deg2rad(theta_bin_width) * (
                (2 * np.pi / wavelength_ranges[i]) ** 2 - (2 * np.pi / wavelength_ranges[i + 1]) ** 2
        )
        ang_pspec_array.append(ang_pspec)

    return ang_pspec_array, theta_vals


def interp_to_polar(pspec_2d, wavenumbers, thetas, theta_bins=(0, 180), theta_step=1, wnum_range=(0.2, 2),
                    wnum_step=0.01):
    """Interpolates power spectrum onto polar grid. not used currently"""

    # create values of theta and wavenumber at which to interpolate
    theta_bins_interp, theta_gridp = create_bins(theta_bins, theta_step)
    wnum_bins_interp, wavenumber_gridp = create_bins(wnum_range, wnum_step)
    meshed_polar = np.meshgrid(wavenumber_gridp, theta_gridp)

    points = np.array([[k, l] for k, l in zip(wavenumbers.flatten(), thetas.flatten())])
    xi = np.array([[w, t] for w, t in zip(meshed_polar[0].flatten(), meshed_polar[1].flatten())])
    values = pspec_2d.flatten()

    interp_values = griddata(points, values.data, xi, method='linear')

    grid = xi.reshape(meshed_polar[0].shape[0], meshed_polar[0].shape[1], 2)

    return wnum_bins_interp, theta_bins_interp, grid, interp_values.reshape(meshed_polar[0].shape)


def find_max(polar_pspec, wnum_vals, theta_vals):
    """
    finds the wavenumber and theta corresponding to the maximum value of the polar power spectrum
    Parameters
    ----------
    polar_pspec
    wnum_vals
    theta_vals

    Returns
    -------

    """
    meshed_polar = np.meshgrid(wnum_vals, theta_vals)
    max_idx = np.nanargmax(polar_pspec)
    return meshed_polar[0].flatten()[max_idx], meshed_polar[1].flatten()[max_idx]


def find_corr_max(collapsed_correlation, K, L, wavelengths, thetas):
    """
    Same as find_max but for the 2d correlation array
    Parameters
    ----------
    collapsed_correlation : np.array
        has to be the "collapsed" correlation array
    K
    L
    wavelengths
    thetas

    Returns
    -------

    """
    halfway = K.shape[1] // 2
    test_idx = np.nanargmax(collapsed_correlation)
    dom_K, dom_L = K[:, halfway:].flatten()[test_idx], L[:, halfway:].flatten()[test_idx]
    dominant_wlen, dominant_theta = wavelengths[:, halfway:].flatten()[test_idx], thetas[:, halfway:].flatten()[test_idx]

    return dominant_wlen, dominant_theta, dom_K, dom_L


def find_corr_error(coll_corr, K, L, dom_wnum, dom_theta):
    """
    gives an estimate on the error of the corr method in lambda and theta
    Parameters
    ----------
    coll_corr
    K
    L
    dom_wnum
    dom_theta

    Returns
    -------

    """
    # make coll_corr with values for 180<theta<360 so that edges are properly interpolated
    full_coll_corr = np.empty_like(K)
    full_coll_corr[:, K.shape[1]//2:] = coll_corr.data
    full_coll_corr[:, :K.shape[1]//2 + 1] = rotate(coll_corr.data, 180)

    # reverse y direction as function needs y strictly increasing
    interp = RectBivariateSpline(L[::-1, 0], K[0], full_coll_corr[::-1])

    def interp_at_const_theta(k, theta=90):
        kx, ky = pol2cart(k, np.deg2rad(-(theta-90)))
        return interp(ky, kx).flatten() - np.nanmax(coll_corr)/2

    def interp_at_const_k(theta, k=1):
        kx, ky = pol2cart(k, np.deg2rad(-(theta-90)))
        return interp(ky, kx).flatten() - np.nanmax(coll_corr)/2

    res = root(interp_at_const_k, dom_theta + 5, args=dom_wnum)
    if not res.success:
        print('WARNING: root finding not successful')
    theta_plus = res.x[0] % 360
    res = root(interp_at_const_k, dom_theta - 5, args=dom_wnum)
    if not res.success:
        print('WARNING: root finding not successful')
    theta_min = res.x[0] % 360

    res = root(interp_at_const_theta, dom_wnum + 0.05, args=dom_theta)
    if not res.success:
        print('WARNING: root finding not successful')
    k_plus = res.x[0]
    res = root(interp_at_const_theta, dom_wnum - 0.05, args=dom_theta)
    if not res.success:
        print('WARNING: root finding not successful')
    k_min = res.x[0]

    return (2 * np.pi / k_plus, 2 * np.pi / k_min), (theta_min, theta_plus)


def apply_wnum_bounds(polar_pspec, wnum_vals, wnum_bins, wlen_range):
    """
    Returns the polar power spectrum that is within a desired wavelength range, given its wavenumber bins and midpoint
    values.
    Parameters
    ----------
    polar_pspec
    wnum_vals
    wnum_bins
    wlen_range

    Returns
    -------

    """
    min_mask = (wnum_bins > 2 * np.pi / wlen_range[1])[:-1]
    max_mask = (wnum_bins < 2 * np.pi / wlen_range[0])[1:]
    mask = (min_mask & max_mask)

    return polar_pspec[:, mask], wnum_vals[mask]


def correlate(a1, b1):
    a = a1.copy()
    b = b1.copy()

    assert a.shape == b.shape

    num = np.sum((a - a.mean()) * (b - b.sum()))

    return num


def correlate_ellipse(pspec, angles, shape):
    """shape has to be longer in y direction"""
    correlation_array = np.zeros_like(pspec.data)

    # loop over elements
    for iy, ix in np.ndindex(pspec.shape):
        # only consider points within lambda-range
        if not pspec.mask[iy, ix]:
            # rotate according to theta
            ell = np.pad(ellipse(*shape), ((0, 0), (shape[1] - shape[0], shape[1] - shape[0])))
            ell = rotate(ell, 90 - angles[iy, ix], resize=False)

            # select correct sub-matrix around pixel to correlate with
            half_y_len = ell.shape[0] // 2
            half_x_len = ell.shape[1] // 2
            sub_matrix = pspec.data[iy - half_y_len: iy + half_y_len + 1, ix - half_x_len: ix + half_x_len + 1]

            # correlate and assign to pixel
            correlation_array[iy, ix] = correlate(sub_matrix, ell / ell.sum())

    return np.ma.masked_where(pspec.mask, correlation_array)


def get_ellipse_correlation(pspec_2d, thetas, ell_shape):
    """
    gets the correlation of the 2d power spectrum with an ellipse of shape ell_shape
    Parameters
    ----------
    pspec_2d
    thetas
    ell_shape

    Returns
    -------

    """

    corr = correlate_ellipse(pspec_2d, thetas, ell_shape)
    rot_left_half = rotate(corr[:, :corr.shape[1] // 2 + 1], 180)
    collapsed_corr = corr[:, corr.shape[1] // 2:] + rot_left_half

    # mask bottom half of k_x = 0 line as this is the same as the top half
    collapsed_corr.mask[collapsed_corr.shape[0] // 2:, 0] = True

    return collapsed_corr