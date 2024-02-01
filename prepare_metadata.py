import json


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