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


def table_to_array(df, mapping_file, pad_val):
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

    # load mapping file
    mapping = pyreadr.read_r(mapping_file)[None].to_numpy()

    # get lat and lon values
    lati, long = mapping[:,1], mapping[:,2]
    points = np.array((lati,long)).T
    lat_arg = np.argsort(lati)
    lon_arg = np.argsort(long)
    max_lat = lati[lat_arg[-1]]
    min_lat = lati[lat_arg[0]]
    min_lon = long[lon_arg[0]]
    max_lon = long[lon_arg[-1]]
    grid_w = 1.0125
    grid_h = 0.525

    lons = np.arange(start=min_lon-(grid_w/2), stop=max_lon+(grid_w/2),  step=grid_w)

    filtered_points = []
    for p in points:
        # check long is less than bound
        if p[1] < lons[1]:
            filtered_points.append(p)
    
    lats = []

    max = len(filtered_points)-1
    sorty = sorted(filtered_points, key=lambda x: x[0])
    l=0
    while l < max:
        mid = (sorty[l+1][0] + sorty[l][0])/2
        lats.append(mid)
        l+=1
    # add first and last lines
    lats.append(sorty[0][0]-grid_h)
    lats.append(sorty[-1][0]+grid_h)
    

    fig = go.Figure()
    for i in range(len(lats)):
        for k in range(len(lons)-1):
            fig.add_trace(
                go.Scattergeo(
                    lon = [lons[k], lons[k+1]],
                    lat = [lats[i], lats[i]],
                    mode = 'lines')
            )
    for j in range(len(lons)):
        fig.add_trace(
            go.Scattergeo(
                lon = [lons[j], lons[j]],
                lat = [min_lat-grid_h, max_lat+grid_h],
                mode = 'lines')
        )

    fig.add_trace(go.Scattergeo(
    lon = long,
    lat = lati))

    fig.show()


table_to_array(merge_position("data/merged_sst_ice_chl_par_2003.RDS", "data/Bering_full_grid_lookup_no_goa.RDS"), 
    "data/Bering_full_grid_lookup_no_goa.RDS", -1)

