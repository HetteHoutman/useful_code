import os
import sys

import iris
import netCDF4 as nc
import numpy as np
from iris.analysis import Linear
from iris.analysis.cartography import rotate_winds
from iris.coords import AuxCoord

from cube_processing import read_variable, cube_from_array_and_cube, cube_at_single_level, create_km_cube, add_pressure_to_cube
from fourier import extract_distances
from prepare_metadata import get_sat_map_bltr


def get_w_field_img(datetime, region, map_height=2000, leadtime=0, coord='altitude',
                    region_root='/home/users/sw825517/Documents/tephiplot/regions/'):
    """
    gets w field from ukv and prepares it for fourier analysis
    Parameters
    ----------
    settings

    Returns
    -------

    """

    file = f'/storage/silver/metstudent/phd/sw825517/ukv_data/ukv_{datetime.strftime("%Y-%m-%d_%H")}_{leadtime:03.0f}.pp'
    sat_bl, sat_tr, map_bl, map_tr = get_sat_map_bltr(region, region_root=region_root)
    km_cube = create_km_cube(sat_bl, sat_tr)

    w_cube = read_variable(file, 150, datetime.hour)
    u_cube = read_variable(file, 2, datetime.hour).regrid(w_cube, Linear())
    v_cube = read_variable(file, 3, datetime.hour).regrid(w_cube, Linear())

    u_cube, v_cube = rotate_winds(u_cube, v_cube, km_cube.coord_system())

    if coord=='air_pressure':
        pw_cube = read_variable(file, 407, datetime.hour)
        pw_coord = AuxCoord(points=pw_cube.data, standard_name=pw_cube.standard_name, long_name=pw_cube.long_name,
                            units=pw_cube.units, coord_system=pw_cube.coord_system())
        pu_cube = read_variable(file, 408, datetime.hour)
        pu_coord = AuxCoord(points=pu_cube.data, standard_name=pu_cube.standard_name, long_name=pu_cube.long_name,
                            units=pu_cube.units, coord_system=pu_cube.coord_system())
        add_pressure_to_cube(w_cube, pw_coord)
        add_pressure_to_cube(u_cube, pu_coord)
        add_pressure_to_cube(v_cube, pu_coord)

    w_single_level, u_single_level, v_single_level = cube_at_single_level(map_height, w_cube, u_cube, v_cube,
                                                                          bottomleft=map_bl, topright=map_tr, coord=coord)
    w_field = w_single_level.regrid(km_cube, Linear())
    u_field = u_single_level.regrid(km_cube, Linear())
    v_field = v_single_level.regrid(km_cube, Linear())
    wind_dir_field = cube_from_array_and_cube(90 -  np.rad2deg(np.arctan2(v_field[0].data, u_field[0].data)), km_cube)

    # prepare data for fourier analysis
    Lx, Ly = extract_distances(w_field.coords('latitude')[0].points, w_field.coords('longitude')[0].points)
    return w_field, u_field, v_field, wind_dir_field, Lx, Ly


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

        # remove unpacked pp now that .nc file has been created, to free up space
        os.system(f'rm {unpacked_pp}')

    # convert radsim reflectivity data from netCDF4 into iris cube, to regrid it onto a regular latlon grid
    surf_t = read_variable(packed_pp, 24, datetime.hour)
    refl_cube = cube_from_array_and_cube(refl[::-1], surf_t, unit=1, std_name='toa_bidirectional_reflectance')
    sat_bl, sat_tr, _, _ = get_sat_map_bltr(region, region_root)
    refl_regrid = refl_cube.regrid(create_km_cube(sat_bl, sat_tr), Linear())

    x_dist, y_dist = extract_distances(refl_regrid.coords('latitude')[0].points,
                                       refl_regrid.coords('longitude')[0].points)
    image = refl_regrid.data[::-1]

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
