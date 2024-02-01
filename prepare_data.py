import json
import os
import sys

import iris
from iris.analysis import Linear
import netCDF4 as nc

from cube_processing import read_variable, cube_from_array_and_cube, cube_at_single_level, create_km_cube
from fourier import extract_distances
from psd import periodic_smooth_decomp


def get_w_field_img(datetime, region, map_height=2000, leadtime=0, region_root='/home/users/sw825517/Documents/tephiplot/regions/'):
    """
    gets w field from ukv and prepares it for fourier analysis
    Parameters
    ----------
    settings

    Returns
    -------

    """
    file = f'/home/users/sw825517/Documents/ukv_data/ukv_{datetime.strftime("%Y-%m-%d_%H")}_{leadtime:03.0f}.pp'

    w_cube = read_variable(file, 150, datetime.hour)
    u_cube = read_variable(file, 2, datetime.hour).regrid(w_cube, iris.analysis.Linear())
    v_cube = read_variable(file, 3, datetime.hour).regrid(w_cube, iris.analysis.Linear())

    sat_bl, sat_tr, map_bl, map_tr = get_sat_map_bltr(region, region_root=region_root)
    w_single_level, u_single_level, v_single_level = cube_at_single_level(map_height, w_cube, u_cube, v_cube,
                                                                          bottomleft=map_bl, topright=map_tr)
    w_field = w_single_level.regrid(create_km_cube(sat_bl, sat_tr), iris.analysis.Linear())

    # prepare data for fourier analysis
    Lx, Ly = extract_distances(w_field.coords('latitude')[0].points, w_field.coords('longitude')[0].points)
    w_field = w_field[0, ::-1].data
    return w_field, Lx, Ly


def get_radsim_img(datetime, region, region_root='/home/users/sw825517/Documents/tephiplot/regions/'):
    """
    gets the simulated satellite imagery from radsim and prepares it for fourier analysis.
    could probs divvy this up into functions too
    Parameters
    ----------
    settings
    datetime

    Returns
    -------

    """
    # this should be the radsim output netCDF4 file
    nc_file_root = f"/home/users/sw825517/radsim/radsim-3.2/outputs"
    datetime_string = datetime.strftime("%Y-%m-%d_%H")
    nc_filename = f'{datetime_string}.nc'

    # pp files
    packed_pp = f'/home/users/sw825517/Documents/ukv_data/ukv_{datetime_string}_000.pp'
    unpacked_pp = f'/home/users/sw825517/Documents/ukv_data/ukv_{datetime_string}_000.unpacked.pp'

    # in case radsim has already run simulation
    try:
        refl = get_refl(nc_file_root + '/' + nc_filename)

    # if not, then run radsim
    except FileNotFoundError:
        print('Radsim output .nc file not found: running radsim to create')
        radsim_run_file = "/home/users/sw825517/radsim/radsim-3.2/src/scripts/radsim_run.py"

        # run radsim_run.py with radsim_settings, so set radsim_settings accordingly
        radsim_settings = {'config_file': f'/home/users/sw825517/radsim/radsim-3.2/outputs/{datetime_string}.cfg',
                           'radsim_bin_dir': '/home/users/sw825517/radsim/radsim-3.2/bin/',
                           'brdf_atlas_dir': '/home/users/sw825517/rttov13/brdf_data/',
                           'use_brdf_atlas': False,
                           'model_datafile': unpacked_pp,
                           'model_filetype': 0,
                           'rttov_coeffs_dir': '/home/users/sw825517/rttov13/rtcoef_rttov13/rttov13pred54L',
                           'rttov_coeffs_options': '_o3co2',
                           'rttov_sccld_dir': '/home/users/sw825517/rttov13/rtcoef_rttov13/cldaer_visir/',
                           'platform': "msg",
                           'satid': 3,
                           'inst': "seviri",
                           'channels': 12,
                           'output_mode': 1,
                           'addsolar': True,
                           'ir_addclouds': True,
                           'output_dir': nc_file_root,
                           'output_file': nc_filename,
                           'write_latlon': True,
                           # 'run_mfasis': True,
                           # 'rttov_mfasis_nn_dir': '/home/users/sw825517/rttov13/rtcoef_rttov13/mfasis_nn/'
                           }

        # check whether unpacked pp file exists, if not then unpack packed pp file
        if not os.path.isfile(radsim_settings['model_datafile']):
            print('Unpacked .pp file does not exist, will try to create...')

            # check whether packed pp exists
            if os.path.isfile(packed_pp):
                # ensure packed pp has 10m winds on correct grid
                try:
                    _ = read_variable(packed_pp, 3209, datetime.hour)
                    _ = read_variable(packed_pp, 3210, datetime.hour)
                except IndexError:
                    print(f'packed .pp {packed_pp} does not have 10m winds on correct grid, regridding...')
                    regrid_10m_wind_and_append(datetime, packed_pp)

                # unpack
                os.system(f'/home/users/sw825517/Documents/ukv_data/pp_unpack {packed_pp}')
                # rename unpacked pp so that it ends on '.pp'
                os.system(
                    f"cp /home/users/sw825517/Documents/ukv_data/ukv_{datetime_string}_000.pp.unpacked {unpacked_pp}")
                os.system(f"rm /home/users/sw825517/Documents/ukv_data/ukv_{datetime_string}_000.pp.unpacked")

            else:
                print(f'packed .pp {packed_pp} not found')
                sys.exit(1)

        set_str = ''

        for setting in radsim_settings:
            set_str += f'--{setting} {radsim_settings[setting]} '

        os.system(f"python {radsim_run_file} {set_str}")
        refl = get_refl(nc_file_root + '/' + nc_filename)

    # convert radsim reflectivity data from netCDF4 into iris cube, to regrid it onto a regular latlon grid
    surf_t = read_variable(packed_pp, 24, datetime.hour)
    refl_cube = cube_from_array_and_cube(refl[::-1], surf_t, unit=1, std_name='toa_bidirectional_reflectance')
    sat_bl, sat_tr, _, _ = get_sat_map_bltr(region, region_root)
    refl_regrid = refl_cube.regrid(create_km_cube(sat_bl, sat_tr), Linear())

    x_dist, y_dist = extract_distances(refl_regrid.coords('latitude')[0].points,
                                       refl_regrid.coords('longitude')[0].points)
    image = refl_regrid.data[::-1]
    image, smooth = periodic_smooth_decomp(image)

    return image, x_dist, y_dist


def get_refl(nc_file):
    varname = 'refl'
    output = nc.Dataset(nc_file)

    if varname not in output.variables:
        print('Error: varname ' + varname + ' not found in ' + nc_file)
        sys.exit(1)
    if 'lat' not in output.variables or 'lon' not in output.variables:
        print('Error: netCDF file must contain lat and lon values')
        sys.exit(1)

    refl = output.variables['refl'][:].reshape(808, 621)[::-1]
    return refl


def regrid_10m_wind_and_append(datetime, pp_file):
    surft = read_variable(pp_file, 24, datetime.hour)
    u = read_variable(pp_file, 3225, datetime.hour)
    v = read_variable(pp_file, 3226, datetime.hour)

    u_rg = u.regrid(surft, Linear())
    v_rg = v.regrid(surft, Linear())

    u_rg.attributes['STASH'] = u_rg.attributes['STASH']._replace(item=209)
    v_rg.attributes['STASH'] = v_rg.attributes['STASH']._replace(item=210)

    iris.save([u_rg, v_rg], pp_file, append=True)


def regrid_10m_wind_and_save(settings, pp_file, target_file):
    """
    !not used
    assumes places of certain cubes in list of cubes from pp_file
    u10m at -4, v10m -2, surface temperature -6
    Parameters
    ----------
    settings
    pp_file
    target_file

    Returns
    -------

    """

    full_pp = iris.load(pp_file)

    surft = full_pp[-6]
    u = full_pp[-4]
    v = full_pp[-2]

    u_rg = u.regrid(surft, Linear())
    v_rg = v.regrid(surft, Linear())

    u_rg.attributes['STASH'] = u_rg.attributes['STASH']._replace(item=309)
    v_rg.attributes['STASH'] = v_rg.attributes['STASH']._replace(item=310)

    full_pp[-4] = u_rg
    full_pp[-2] = v_rg

    iris.save(full_pp, target_file)


def get_sat_map_bltr(region, region_root='/home/users/sw825517/Documents/tephiplot/regions/'):
    """
    gives the bottom left and top right points of the "map" and "satellite" plots
    Parameters
    ----------
    region
    region_root

    Returns
    -------

    """
    sat_bounds = get_variable_from_region_json("sat_bounds", region, region_root)
    satellite_bottomleft, satellite_topright = sat_bounds[:2], sat_bounds[2:]
    map_bounds = get_variable_from_region_json("map_bounds", region, region_root)
    map_bottomleft, map_topright = map_bounds[:2], map_bounds[2:]

    return satellite_bottomleft, satellite_topright, map_bottomleft, map_topright


def get_variable_from_region_json(var, region, root='/home/users/sw825517/Documents/tephiplot/regions/'):
    """
    returns the bottom left and top right lon/lat coordinates for the satellite image and map
    Parameters
    ----------
    region : str
        the region for which the bounds should be returned

    Returns
    -------

    """
    with open(root + region + '.json') as f:
        bounds_dict = json.loads(f.read())

    return bounds_dict[var]