import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker, colors

from miscellaneous import create_bins_from_midpoints


def plot_pspec_polar(wnum_bins, theta_bins, radial_pspec_array, scale='linear', xlim=None, vmin=None, vmax=None,
                     min_lambda=3, max_lambda=20, title='title'):
    if vmin is not None and vmax is not None:
        plt.pcolormesh(wnum_bins, theta_bins, radial_pspec_array, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        plt.colorbar(extend='both')
    else:
        plt.pcolormesh(wnum_bins, theta_bins, radial_pspec_array, norm=mpl.colors.LogNorm())
        plt.colorbar()
    plt.xscale(scale)
    if xlim is not None:
        plt.xlim(xlim)
    plt.ylim(theta_bins[0], theta_bins[-1])

    plt.vlines(2 * np.pi / min_lambda, theta_bins[0], theta_bins[-1], 'k', linestyles='-.')
    plt.vlines(2 * np.pi / max_lambda, theta_bins[0], theta_bins[-1], 'k', linestyles='-.')

    plt.xlabel(r"$|\mathbf{k}|$" + ' / ' + r"$\rm{km}^{-1}$")
    plt.ylabel(r'$\theta$')
    plt.title(title)


def plot_interp_pcolormesh(pspec_2d, wnum_bins_interp, theta_bins_interp, interp_values):
    """Plots interpolated values on a pcolormesh plot."""
    plt.pcolormesh(wnum_bins_interp, theta_bins_interp, interp_values,
                   norm=colors.LogNorm(vmin=pspec_2d.min(), vmax=pspec_2d.max()), )
    plt.colorbar(extend='both')
    plt.xlabel(r"$|\mathbf{k}|$" + ' / ' + r"$\rm{km}^{-1}$")
    plt.ylabel(r'$\theta$')


def plot_interp_contour(grid, interp_values):
    """Plots interpolated values on a contour plot. Colourscale is weird though"""
    con = plt.contourf(grid[:, :, 0], grid[:, :, 1], interp_values, locator=ticker.LogLocator())
    plt.colorbar(con, extend='both')
    plt.xlabel(r"$|\mathbf{k}|$" + ' / ' + r"$\rm{km}^{-1}$")
    plt.ylabel(r'$\theta$')


def plot_ang_pspec(pspec_array, vals, wavelength_ranges):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0, 1, len(pspec_array))))
    for i, pspec in enumerate(pspec_array):
        plt.plot(vals, pspec,
                 label=f'{wavelength_ranges[i]} km' + r'$ \leq \lambda < $' + f'{wavelength_ranges[i + 1]} km')

    ax = plt.gca()
    ax.set_yscale('log')
    plt.title('Angular power spectrum')
    plt.ylabel(r"$P(\theta)$")
    plt.xlabel(r'$\theta$ (deg)')
    plt.legend(loc='lower left')
    plt.grid()


def plot_radial_pspec(pspec_array, vals, theta_ranges, dom_wnum, min_lambda=3, max_lambda=20, title='title'):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0, 1, len(pspec_array))))
    for i, pspec in enumerate(pspec_array):
        plt.loglog(vals, pspec, label=f'{theta_ranges[i]}' + r'$ \leq \theta < $' + f'{theta_ranges[i + 1]}')

    ymin = np.nanmin(np.array(pspec_array))
    ymax = np.nanmax(np.array(pspec_array))

    plt.vlines(2 * np.pi / 8, ymin, ymax, 'k', linestyles='--')
    plt.vlines(2 * np.pi / min_lambda, ymin, ymax, 'k', linestyles='-.')
    plt.vlines(2 * np.pi / max_lambda, ymin, ymax, 'k', linestyles='-.')
    plt.vlines(dom_wnum, ymin, ymax, 'k')

    plt.title('1D Power Spectrum')
    plt.xlabel(r"$|\mathbf{k}|$" + ' / ' + r"$\rm{km}^{-1}$")
    plt.ylabel(r"$P(|\mathbf{k}|)$")
    plt.ylim(ymin, ymax)
    plt.legend(loc='lower left')
    plt.title(title)
    plt.tight_layout()


def plot_2D_pspec(bandpassed_pspec, K, L, wavelengths, wavelength_contours=None, title='title'):

    fig2, ax2 = plt.subplots(1, 1)
    kbins = create_bins_from_midpoints(K[0])
    lbins = create_bins_from_midpoints(L[:,0])
    pcol = ax2.pcolormesh(kbins, lbins, bandpassed_pspec.data, norm=mpl.colors.LogNorm(vmin=bandpassed_pspec.min(), vmax=bandpassed_pspec.max()))

    if wavelength_contours:
        con = ax2.contour(K, L, wavelengths, levels=wavelength_contours, colors=['k'], linestyles=['--'])
        ax2.clabel(con)

    ax2.set_title('2D Power Spectrum')
    ax2.set_xlabel(r"$k_x$" + ' / ' + r"$\rm{km}^{-1}$")
    ax2.set_ylabel(r"$k_y$" + ' / ' + r"$\rm{km}^{-1}$")
    fig2.colorbar(pcol, extend='both')
    plt.title(title)
    plt.tight_layout()


def filtered_inv_plot(img, filtered_ft, Lx, Ly, latlon=None, inverse_fft=True, min_lambda=3, max_lambda=20,
                      title='title', radsim=True):
    if inverse_fft:
        fig, (ax1, ax3) = plt.subplots(1, 2, sharey=True)
    else:
        fig, ax1 = plt.subplots(1, 1)

    xlen = img.shape[1]
    ylen = img.shape[0]

    if latlon:
        physical_extent = [latlon[0], latlon[2], latlon[1], latlon[3]]
        xlabel = 'Longitude'
        ylabel = 'Latitude'
    else:
        pixel_x = Lx / xlen
        pixel_y = Ly / ylen
        physical_extent = [-Lx / 2 - pixel_x / 2, Lx / 2 + pixel_x / 2, -Ly / 2 - pixel_y / 2, Ly / 2 + pixel_y / 2]
        xlabel = 'x distance / km'
        ylabel = 'y distance / km'

    plot = ax1.imshow(img,
               extent=physical_extent,
               cmap='gray')
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)

    if inverse_fft:
        inv = np.fft.ifft2(filtered_ft.filled(fill_value=1))
        ax3.set_title(f'{min_lambda} km < lambda < {max_lambda} km')
        ax3.imshow(abs(inv),
                   extent=physical_extent,
                   cmap='gray')

    ax1.set_title(title)
    if radsim:
        label = 'TOA reflectance'
    else:
        label = 'Vertical velocity'
    plt.colorbar(plot, label=label)
    plt.tight_layout()


def plot_corr(coll_corr, K, L):
    fig2, ax2 = plt.subplots(1, 1)
    kbins = create_bins_from_midpoints(K[0])
    lbins = create_bins_from_midpoints(L[:,0])

    im = ax2.pcolormesh(kbins[kbins.size // 2 - 1:], lbins, coll_corr)

    ax2.set_xlabel(r"$k_x$" + ' / ' + r"$\rm{km}^{-1}$")
    ax2.set_ylabel(r"$k_y$" + ' / ' + r"$\rm{km}^{-1}$")
    ax2.set_xlim(0, 2)
    ax2.set_ylim(-2, 2)
    fig2.colorbar(im)
    ax2.set_aspect('equal')
    plt.tight_layout()
