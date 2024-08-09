import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from matplotlib.colors import CenteredNorm
import cartopy.crs as ccrs
import iris.plot as iplt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cube_processing import cube_from_array_and_cube


def calc_extent(image, Lx, Ly):
    pixel_x = Lx / image.shape[1]
    pixel_y = Ly / image.shape[0]
    extent = [-Lx / 2 - pixel_x / 2, Lx / 2 + pixel_x / 2, -Ly / 2 - pixel_y / 2, Ly / 2 + pixel_y / 2]
    return extent


def plot_contour_over_image(orig, plotted_array, Lx, Ly, cbarlabels=['colorbar label1', 'cbarlabel2'], pspec_thresh=None,
                            **contour_kwargs):
    img = plt.imshow(orig, cmap='gray', extent=calc_extent(orig, Lx, Ly))
    var = plt.contourf(plotted_array[::-1], extent=calc_extent(orig, Lx, Ly), vmin=pspec_thresh,
                       vmax = plotted_array.max(), **contour_kwargs)

    var_cbar = plt.colorbar(var, label=cbarlabels[1])
    if pspec_thresh is not None:
        var_cbar.ax.set_yticks(list(var_cbar.ax.get_yticks()) + [pspec_thresh])
        var_cbar.ax.set_ylim(pspec_thresh, plotted_array.max())
    var_cbar.solids.set(alpha=1)
    img_cbar = plt.colorbar(img, label=cbarlabels[0])

    plt.xlabel('Easting (km)')
    plt.ylabel('Northing (km)')
    plt.tight_layout()


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
    fig.colorbar(pc, label=cbarlabel, pad=-0.075)
    ax.set_ylabel('Wavelength (km)', labelpad=-40)
    plt.tight_layout()


def plot_result_lambda_hist(l1, l2, l_edges, label1=None, label2=None):
    hist, _, _, _ = plt.hist2d(l1, l2, bins=[l_edges, l_edges], cmin=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar(label='Number of cases')
    ax = plt.gca()
    ax.yaxis.set_minor_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    plt.xlabel(label1)
    plt.ylabel(label2)


def plot_wind(w, u, v, step=25):
    fig, ax = plt.subplots(1, 1,
                           subplot_kw={'projection': ccrs.PlateCarree()}
                           )
    con = iplt.pcolormesh(w[0], norm=CenteredNorm(), cmap='brewer_PuOr_11')
    iplt.quiver(u[0, ::step, ::step], v[0, ::step, ::step])
    ax.gridlines(draw_labels=True)
    ax.coastlines()
    plt.colorbar(con, label='Upward air velocity / m/s', location='right',
                 # orientation='vertical'
                 )

def plot_wind_and_or(u, v, theta, sat, step=25):
    fig, ax = plt.subplots(1, 1, subplot_kw = {'projection': ccrs.PlateCarree()})
    orig_cube = cube_from_array_and_cube(sat[::-1][None, ...], u)
    con = iplt.pcolormesh(orig_cube[0], cmap='gray')

    iplt.quiver(u[0, ::step, ::step], v[0, ::step, ::step])

    theta_x, theta_y = (cube_from_array_and_cube(np.cos(np.deg2rad(90 - theta[::-1]))[None, ...], u),
                        cube_from_array_and_cube(np.sin(np.deg2rad(90 - theta[::-1]))[None, ...], u))
    iplt.quiver(theta_x[0, ::step, ::step], theta_y[0, ::step, ::step], color='r')
    ax.gridlines(draw_labels=True)
    ax.coastlines()
    plt.colorbar(con, label='TOA reflectance', location='right')


def plot_method_example_hist(hist, lambda_bin_edges, theta_bin_edges, lambdas_selected, thetas_selected):
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
    ax.set_ylabel('Wavelength (km)', labelpad=-40)

    pc = ax.pcolormesh(T, L, hist)
    fig.colorbar(pc, label=r'Area / $\lambda^2$', location='right', pad=-0.075)
    for l, t in zip(lambdas_selected, thetas_selected):
        plt.scatter(np.deg2rad(t), l, marker='x', color='r', s=75)

    plt.savefig('plots/methods_2dhist.png', bbox_inches='tight', dpi=300)


def plot_method_example_map(orig, max_lambdas, Lx, Ly):
    fig, ax2 = plt.subplots(1, 1)
    img = ax2.imshow(orig, cmap='gray', extent=calc_extent(orig, Lx, Ly))
    var = ax2.contourf(max_lambdas[::-1], extent=calc_extent(orig, Lx, Ly), alpha=0.5, levels=5, cmap='plasma')
    ax2.axis('off')

    divider = make_axes_locatable(ax2)
    cax1 = divider.append_axes("top", size="5%", pad=0.05)
    cax3 = divider.append_axes("bottom", size="5%", pad=0.05)
    img_cbar = plt.colorbar(img, label=r'TOA reflectance', location='top', cax=cax1)
    var_cbar = plt.colorbar(var, label='Dominant wavelength (km)', location='bottom', cax=cax3)
    var_cbar.solids.set(alpha=1)

    plt.savefig('plots/methods_map.png', bbox_inches='tight', dpi=300)

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
def plot_method_example_old(orig, max_lambdas, Lx, Ly, hist, lambda_bin_edges, theta_bin_edges):
    fig = plt.figure()
    gs0 = GridSpec(1, 2, figure=fig)
    gs00 = GridSpecFromSubplotSpec(3, 1, height_ratios=(0.5, 4, 0.5), subplot_spec=gs0[0], wspace=0.01, hspace=0.01)
    gs01 = GridSpecFromSubplotSpec(1, 2, width_ratios=(8, 1), subplot_spec=gs0[1])

    ax1 = fig.add_subplot(gs00[0, 0])
    ax2 = fig.add_subplot(gs00[1, 0])
    ax3 = fig.add_subplot(gs00[2, 0])

    img = ax2.imshow(orig, cmap='gray', extent=calc_extent(orig, Lx, Ly))
    cbar_img = plt.colorbar(img, label=r'Vertical velocity $\mathregular{(ms^{-1})}$', cax=ax1, location='top')

    var = ax2.contourf(max_lambdas[::-1], extent=calc_extent(orig, Lx, Ly), alpha=0.5, cmap='Reds')
    cbar_var = plt.colorbar(var, location='bottom', label='Dominant wavelength (km)', cax=ax3)
    cbar_var.solids.set(alpha=1)

    T, L = np.meshgrid(np.deg2rad(theta_bin_edges), lambda_bin_edges)

    ax4 = fig.add_subplot(gs01[:, 0], projection='polar')
    ax5 = fig.add_subplot(gs01[:, 1])
    ax4.set_theta_direction(-1)
    ax4.set_theta_offset(np.pi / 2)
    ax4.set_thetalim((np.deg2rad(theta_bin_edges[0]), np.deg2rad(theta_bin_edges[-1])))
    ax4.set_rscale('log')
    ax4.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax4.yaxis.set_minor_formatter(NullFormatter())
    ax4.set_rlim([int(lambda_bin_edges[0]), int(lambda_bin_edges[-1]) + 1])
    ax4.set_rgrids([int(lambda_bin_edges[0]) + 1, 10, 20, 30])


    pc = ax4.pcolormesh(T, L, hist)
    cbar_pc = plt.colorbar(pc, location='right', label=r'Area / $\lambda^2$', cax=ax5)

    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(orig, cmap='gray', extent=calc_extent(orig, Lx, Ly))
    cbar_img = plt.colorbar(img, label=r'Vertical velocity $\mathregular{(ms^{-1})}$', location='top')

    var = ax.contourf(max_lambdas[::-1], extent=calc_extent(orig, Lx, Ly), alpha=0.5, cmap='Reds')
    cbar_var = plt.colorbar(var, location='bottom', label='Dominant wavelength (km)')
    cbar_var.solids.set(alpha=1)

    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.set_theta_direction(-1)
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_thetalim((np.deg2rad(theta_bin_edges[0]), np.deg2rad(theta_bin_edges[-1])))
    ax2.set_rscale('log')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_rlim([int(lambda_bin_edges[0]), int(lambda_bin_edges[-1]) + 1])
    ax2.set_rgrids([int(lambda_bin_edges[0]) + 1, 10, 20, 30])

    pc = ax2.pcolormesh(T, L, hist)
    fig.colorbar(pc, label='henk', location='right', pad=-0.075)


