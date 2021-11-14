import numpy as np
from sklearn.neighbors import NearestNeighbors
import pyreadr
import plotly.express as px
import plotly.graph_objects as go

def merge_position(rds_file, mapping_file):
    """
    Loads .RDS file as a pandas dataframe and adds lat and lon data column using the mapping file.

    Files in data dir should have the following columns in order: 
    date chl_n par_n chlorophyll par sst_n ice_n sst ice depth gridid index

    args:
    data_file: path to .RDS file
    mapping_file: path to file that contains mapping of gridid to lat, lon

    returns:
    df: pandas dataframe with columns: date chlorophyll par sst ice depth gridid meanlat meanlon
    """

    grid_lat_dict = {} # dict that maps gridid -> lat
    grid_lon_dict = {} # dict that maps gridid -> lon
    mapping = pyreadr.read_r(mapping_file)[None].to_numpy()

    for i in mapping:
        grid_lat_dict[i[0]] = i[1]
        grid_lon_dict[i[0]] = i[2]

    df = pyreadr.read_r(rds_file)[None]

    # drop columns
    df = df.drop(["chl_n", "par_n", "sst_n", "ice_n", "index"], axis=1)
    # add meanlat meanlon columns
    df["meanlat"] = df["gridid"].map(grid_lat_dict)
    df["meanlon"] = df["gridid"].map(grid_lon_dict)

    return df

def convert():
    """
    Converts df from merge_position to an array of form:
    [
        [chlorophyll_x0_y0, chlorophyll_x1_y0, ...]
        [chlorophyll_x0_y1, chlorophyll_x1_y1, ...]
        ...
    ]
    This array represents the spatial relationship between data. 
    The data at (0,0) is adjacent to (0,1) in lat/lon coordinates.
    If data is not in a perfect rectangle, values are padded with pad_val

    Args:
    df: dataframe with columns of data. Must have "meanlat" and "meanlon" columns (output of merge_position)
    pad_val: value to pad array if spatial data is not perfectly rectangular
    mapping_file: path to file that contains mapping of gridid to lat, lon

    Returns:
    data_array: Array of shape (num_time_steps, num_data_types, num_unique_lat, num_unique_lon)
    """
    pass

def create_mapping_dict(mapping_file):
    """
    Creates dict that maps lat,long to x,y coords in an array. This is fairly specific to
    the format of our data, so use caution.

    Args:
    mapping_file: path to file that contains mapping of gridid to lat, lon

    Returns:
    fml_dict: dict of form {(lat,long): (x,y)} for every point in mapping_file
    """

    # load mapping file
    mapping = pyreadr.read_r(mapping_file)[None].to_numpy()

    lat_vals, lon_vals = mapping[:,1], mapping[:,2]
    points = np.array((lat_vals,lon_vals)).T
    lat_arg = np.argsort(lat_vals)
    lon_arg = np.argsort(lon_vals)
    max_lat = lat_vals[lat_arg[-1]]
    min_lat = lat_vals[lat_arg[0]]
    min_lon = lon_vals[lon_arg[0]]
    max_lon = lon_vals[lon_arg[-1]]
    grid_w = 1.0125
    grid_h = 0.525

    lon_lines = np.arange(start=min_lon-(grid_w/2), stop=max_lon+(grid_w/2),  step=grid_w)

    # generate latitude lines by calculating midpoints between 
    # all lat values for points less than the second lon line (easier to visualize with plot)
    filtered_points_lat = []
    for p in points:
        # check long is less than bound
        if p[1] < lon_lines[1]:
            filtered_points_lat.append(p)
    
    lat_lines = []
    max = len(filtered_points_lat)-1
    sorted_points_lat = sorted(filtered_points_lat, key=lambda x: x[0])
    l=0
    while l < max:
        mid = (sorted_points_lat[l+1][0] + sorted_points_lat[l][0])/2
        lat_lines.append(mid)
        l+=1

    # Recalculate lon_lines to get them more centered using a similar technique as above
    # this time, using points between the 7th and 8th lat line (again, easier to understand by looking at graph)
    filtered_points_lon = []
    sorted_lats = sorted(lat_lines)
    for p in points:
        # check long is less than bound
        if (p[0] > sorted_lats[6]) and (p[0] < sorted_lats[7]):
            filtered_points_lon.append(p)

    lon_lines = []
    max = len(filtered_points_lon)-1
    sorted_points_lon = sorted(filtered_points_lon, key=lambda x: x[1])
    l=0
    while l < max:
        mid = (sorted_points_lon[l+1][1] + sorted_points_lon[l][1])/2
        lon_lines.append(mid)
        l+=1

    invert_lat = np.sort(np.array(lat_lines) * -1)
    x = np.searchsorted(invert_lat, lat_vals*-1)
    y = np.searchsorted(lon_lines, lon_vals)

    fml_dict = {}
    for i in range(len(x)):
        fml_dict[(lat_vals[i], lon_vals[i])] = (x[i], y[i])

    labely = []
    for i in range(len(lat_vals)):
        c = fml_dict[(lat_vals[i], lon_vals[i])]
        labely.append(str(c))    

    fig = go.Figure()
    for i in range(len(lat_lines)):
        for k in range(len(lon_lines)-1):
            fig.add_trace(
                go.Scattergeo(
                    lon = [lon_lines[k], lon_lines[k+1]],
                    lat = [lat_lines[i], lat_lines[i]],
                    mode = 'lines')
            )
    for j in range(len(lon_lines)):
        fig.add_trace(
            go.Scattergeo(
                lon = [lon_lines[j], lon_lines[j]],
                lat = [min_lat-grid_h, max_lat+grid_h],
                mode = 'lines')
        )

    fig.add_trace(go.Scattergeo(
    lon = lon_vals,
    lat = lat_vals,
    hovertext=labely))

    fig.show()

    return fml_dict


create_mapping_dict("data/Bering_full_grid_lookup_no_goa.RDS")

