import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter

def calc_extent(image, Lx, Ly):
    pixel_x = Lx / image.shape[1]
    pixel_y = Ly / image.shape[0]
    extent = [-Lx / 2 - pixel_x / 2, Lx / 2 + pixel_x / 2, -Ly / 2 - pixel_y / 2, Ly / 2 + pixel_y / 2]
    return extent


def plot_contour_over_image(orig, plotted_array, Lx, Ly, cbarlabel='colorbar label', **contour_kwargs):
    plt.imshow(orig, cmap='gray', extent=calc_extent(orig, Lx, Ly))
    plt.contourf(plotted_array[::-1], extent=calc_extent(orig, Lx, Ly), **contour_kwargs)
    plt.colorbar(label=cbarlabel)

    plt.xlabel('x distance / km')
    plt.ylabel('y distance / km')


def plot_k_histogram(dom_lambdas, dom_thetas, lambda_bin_edges, theta_bin_edges, **kwargs):
    plt.hist2d(dom_lambdas, dom_thetas, bins=[lambda_bin_edges, theta_bin_edges], **kwargs)

    plt.colorbar(label='Dominant wavelet count')
    plt.ylabel('Orientation (degrees from North)')
    plt.xlabel('Wavelength (km)')


def plot_polar_pcolormesh(hist, lambda_bin_edges, theta_bin_edges, cbarlabel='label', **kwargs):
    T, L = np.meshgrid(np.deg2rad(theta_bin_edges), lambda_bin_edges)

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_thetalim((np.deg2rad(theta_bin_edges[0]), np.deg2rad(theta_bin_edges[-1])))
    ax.set_rscale('log')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_rlim([int(lambda_bin_edges[0]), int(lambda_bin_edges[-1]) + 1])
    ax.set_rgrids([int(lambda_bin_edges[0]) + 1, 10, 20, 30])

    pc = ax.pcolormesh(T, L, hist, **kwargs)
    fig.colorbar(pc, label=cbarlabel)
    plt.tight_layout()


def plot_result_lambda_hist(l1, l2, l_edges, label1=None, label2=None):
    hist, _, _, _ = plt.hist2d(l1, l2, bins=[l_edges, l_edges], cmin=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar(label='Count')
    ax = plt.gca()
    ax.yaxis.set_minor_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    plt.xlabel(label1)
    plt.ylabel(label2)