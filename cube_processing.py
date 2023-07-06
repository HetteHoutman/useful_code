# this file contains some useful functions for processing cubes

import time

import cartopy.crs as ccrs
import iris
from iris.experimental.stratify import relevel
import numpy as np

from miscellaneous import make_custom_traj, convert_to_ukv_coords


def read_variable(pp_file, code, hour_selected):
    '''
    Reads variable defined by stash code from pp_file. From P Clark

    Args:
        pp_file (str)
        code (int)

    Returns:
        cubes (list)
    '''
    stash_code = iris_stash_code(code)
    stash_const = iris.AttributeConstraint(STASH=stash_code)
    cubes = iris.load(pp_file, stash_const)
    print(f"Reading data from stash {code:d} at hour {hour_selected:d}")
    hour_const = iris.Constraint(time=lambda cell:
    cell.point.hour == hour_selected)
    cube = cubes.extract(hour_const)[0]

    return cube


def iris_stash_code(code):
    '''
    Converts stash code to iris format. from P Clark

    Args:
        code : Stash code string of up to 5 digits

    Returns:
        stash code in iris format
    '''
    temp = f"{code:05d}"
    iris_stash_code = 'm01s' + temp[0:2] + 'i' + temp[2:]
    return iris_stash_code


def get_coord_index(cube, name):
    """from P Clarks code"""
    for i, c in enumerate(cube.coords()):
        if name in c.standard_name:
            break
    return i


def add_grid_latlon_to_cube(cube, grid_latlon):
    """from P Clarks code"""
    ilat = get_coord_index(cube, 'grid_latitude')
    ilon = get_coord_index(cube, 'grid_longitude')
    cube.add_aux_coord(grid_latlon['lat_coord'], [ilat, ilon])
    cube.add_aux_coord(grid_latlon['lon_coord'], [ilat, ilon])

def add_pressure_to_cube(cube, pcoord):
    ilev = get_coord_index(cube, 'model_level_number')
    ilat = get_coord_index(cube, 'latitude')
    ilon = get_coord_index(cube, 'longitude')
    cube.add_aux_coord(pcoord, [ilev, ilat, ilon])


def get_grid_latlon_from_rotated(cube):
    """from P Clarks code"""
    # in the form of a cartopy map projection.
    crs_rotated = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

    crs_sphere = ccrs.PlateCarree()

    r_lats = cube.coord('grid_latitude').points.copy()
    r_lons = cube.coord('grid_longitude').points.copy()

    rot_lons, rot_lats = np.meshgrid(r_lons, r_lats)

    true_lons = np.zeros_like(cube.data[0, :, :])
    true_lats = np.zeros_like(cube.data[0, :, :])

    for i, r_lon in enumerate(r_lons):
        for j, r_lat in enumerate(r_lats):
            true_lons[j, i], true_lats[j, i] = crs_sphere.transform_point(r_lon,
                                                                          r_lat, crs_rotated)

    true_lons[true_lons > 180] -= 360

    grid_latlon = {'rot_lons': rot_lons,
                   'rot_lats': rot_lats,
                   'true_lons': true_lons,
                   'true_lats': true_lats,
                   'lon_coord': iris.coords.AuxCoord(points=true_lons,
                                                     standard_name='longitude'),
                   'lat_coord': iris.coords.AuxCoord(points=true_lats,
                                                     standard_name='latitude')
                   }
    return grid_latlon


def cube_at_single_level(cube, level, bottomleft=None, topright=None, coord='altitude'):
    """
    returns the cube at a selected level_height and between bottom left and top right bounds
    Parameters
    ----------
    coord
    cube
    level
    bottomleft : tuple
        (true) lon/lat for the bottom left point of the map
    topright : tuple
        (true) lon/lat for the top right point of the map
    Returns
    -------

    """
    if bottomleft is not None and topright is not None:
        crs_latlon = ccrs.PlateCarree()
        crs_rotated = cube.coord('grid_latitude').coord_system.as_cartopy_crs()
        bl_model = crs_rotated.transform_point(bottomleft[0], bottomleft[1], crs_latlon)
        tr_model = crs_rotated.transform_point(topright[0], topright[1], crs_latlon)
        cube = cube.intersection(grid_latitude=(bl_model[1], tr_model[1]),
                                 grid_longitude=(bl_model[0], tr_model[0]))

    single_level = relevel(cube, cube.coords(coord)[0], [level])
    return single_level


def cube_slice(*cubes, bottom_left=None, top_right=None, height=None, force_latitude=False):
    """
    Returns slice(s) of cube(s) between bottom_left (lon, lat) and top_right corners, and between heights.
    Currently, a tad unnecessarily complicated. Could be possibly be simplified using .intersection?
    Parameters
    ----------
    cubes : Cube
        the cube(s) to be sliced
    bottom_left : tuple
        (true) lon/lat of the bottom left corner
    top_right : tuple
        (true) lon/lat of the top right corner
    height : tuple
        range of heights between which to slice
    force_latitude : bool
        if True will set the top_right latitude index to the bottom_left latitude index

    Returns
    -------

    """

    crs_latlon = ccrs.PlateCarree()
    crs_rotated = cubes[0].coord('grid_latitude').coord_system.as_cartopy_crs()

    bl_model = convert_to_ukv_coords(*bottom_left, crs_latlon, crs_rotated)
    tr_model = convert_to_ukv_coords(*top_right, crs_latlon, crs_rotated)

    new_cubes = []

    for cube in cubes:
        lat_idxs = [cube.coord('grid_latitude').nearest_neighbour_index(bl_model[1]),
                    cube.coord('grid_latitude').nearest_neighbour_index(tr_model[1])]

        lon_idxs = (cube.coord('grid_longitude').nearest_neighbour_index(bl_model[0]),
                    cube.coord('grid_longitude').nearest_neighbour_index(tr_model[0]))

        # only slice the height if it is given and if there is a height coordinate
        if (cube.ndim == 3) and height is not None:
            height_idxs = (cube.coord('level_height').nearest_neighbour_index(height[0]),
                           cube.coord('level_height').nearest_neighbour_index(height[1]))
            cube = cube[height_idxs[0]: height_idxs[1] + 1]

        elif height is not None and cube.ndim != 3:
            raise Exception('you gave heights but the cube is not 3 dimensional')

        if force_latitude:
            # "..." ensures only the last two dimensions are sliced regardless of how many are in front of them
            new_cubes.append(cube[..., lat_idxs[0], lon_idxs[0]: lon_idxs[1] + 1])
        else:
            new_cubes.append(cube[..., lat_idxs[0]: lat_idxs[1] + 1, lon_idxs[0]: lon_idxs[1] + 1])

    return new_cubes


def check_level_heights(q, t):
    """check whether q (or any other cube, like w) and Temperature cubes have same level heights and adjust if necessary."""
    if q.coord('level_height').points[0] == t.coord('level_height').points[0]:
        pass
    elif q.coord('level_height').points[1] == t.coord('level_height').points[0]:
        q = q[1:]
    else:
        raise ValueError('Double check the T and q level_heights - they do not match')
    return q


def cube_from_array_and_cube(array, copy_cube, unit=None, std_name=None):
    """
    Creates a new Cube by coping copy_cube and sticking in array as cube.data
    Parameters
    ----------
    array : ndarray
        data for new array
    copy_cube : Cube
        cube to be copied
    unit : str
        optional. units for new cube. if None will use copy_cube's units
    std_name : str
        optional. standard name for new cube. if None will use copy_cube's standard name

    Returns
    -------
    Cube

    """
    new_cube = copy_cube.copy()
    new_cube.data = array
    # is deleting useful in any way?
    del array
    if unit is not None:
        new_cube.units = unit
    if std_name is not None:
        new_cube.standard_name = std_name

    return new_cube


def cube_custom_line_interpolate(custom_line, *cubes):
    """
    Returns the cubes interpolated along the custom_line given.
    Parameters
    ----------
    custom_line : ndarray
        ndarray of shape (m, 2) in format (grid)lon/lat along which to interpolate the cube(s)
    cubes : Cube
        the cube(s) to interpolate

    Returns
    -------
    """
    new_cubes = []
    traj = make_custom_traj(custom_line)
    for cube in cubes:
        new_cubes.append(traj.interpolate(cube, method='linear'))

    return new_cubes


def great_circle_xsect(cube, great_circle, n=50):
    """
    !Currently not used/working properly!
    Produces an interpolated cross-section of cube along a great circle between gc_start and gc_end
    (uses the level_heights of the cube)
    Parameters
    ----------
    cube : Cube
        the cube to be interpolated
    n : int
        the number of points along the great circle at which is interpolated
    gc_start : tuple
        lon/lat of the start of the great circle
    gc_end : tuple
        lon/lat of the end of the great circle

    Returns
    -------
    ndarray
        the interpolated cross-section

    """
    from scipy.interpolate import griddata

    crs_latlon = ccrs.PlateCarree()
    crs_rotated = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

    gc_model = np.array(
        [convert_to_ukv_coords(coords[0], coords[1], crs_latlon, crs_rotated) for coords in great_circle.T])
    grid = np.moveaxis(np.array(np.meshgrid(cube.coord('level_height').points,
                                            cube.coord('grid_longitude').points,
                                            cube.coord('grid_latitude').points)),
                       [0, 1, 2, 3], [-1, 2, 0, 1])
    points = grid.reshape(-1, grid.shape[-1])

    broadcast_lheights = np.broadcast_to(cube.coord('level_height').points,
                                         (n, cube.coord('level_height').points.shape[0])).T
    broadcast_gc = np.broadcast_to(gc_model, (cube.coord('level_height').points.shape[0], *gc_model.shape))

    # combine
    model_gc_with_heights = np.concatenate((broadcast_lheights[:, :, np.newaxis], broadcast_gc), axis=-1)

    start_time = time.clock()
    print('start interpolation...')
    xsect = griddata(points, cube[:, ::-1].data.flatten(), model_gc_with_heights)
    print(f'{time.clock() - start_time} seconds needed to interpolate')

    return xsect


def add_dist_coord(dists, *cubes):
    """
    adds great circle distance (in km) as an AuxCoord to cross-section cube(s)
    Parameters
    ----------
    dists : ndarray
        1d array containing the distances in (m) along the great circle corresponding to each index of the cross-section
    cubes

    Returns
    -------

    """
    dist_coord = iris.coords.AuxCoord(dists / 1000, long_name='distance_from_start', units='km')
    for cube in cubes:
        cube.add_aux_coord(dist_coord, data_dims=1)
    return cubes


def add_orography(orography_cube, *cubes):
    orog_coord = iris.coords.AuxCoord(orography_cube.data, standard_name=str(orography_cube.standard_name),
                                      long_name='orography', var_name='orog', units=orography_cube.units)
    for cube in cubes:
        sigma = cube.coord('sigma')
        delta = cube.coord('level_height')
        fac = iris.aux_factory.HybridHeightFactory(delta=delta, sigma=sigma, orography=orog_coord)
        cube.add_aux_coord(orog_coord,
                           (get_coord_index(cube, 'grid_latitude'), get_coord_index(cube, 'grid_longitude')))
        cube.add_aux_factory(fac)
    del orog_coord

    return cubes
