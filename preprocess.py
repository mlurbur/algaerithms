import numpy as np
import pandas as pd
from pandas.core.dtypes import missing
import pyreadr
import plotly.express as px
import plotly.graph_objects as go
import time

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

def convert(df, lat_dict, lon_dict, column_names):
    """
    Converts df from merge_position to an array of form:
    [
        [chlorophyll_x0_y0, chlorophyll_x1_y0, ...]
        [chlorophyll_x0_y1, chlorophyll_x1_y1, ...]
        ...
    ]
    This array represents the spatial relationship between data. 
    The data at (0,0) is adjacent to (0,1) in lat/lon coordinates.
    If data is not in a perfect rectangle, values are padded with -inf
    Replaces nan with inf
    

    Args:
    df: dataframe with columns of data. 
        Must **have** the following columns: "meanlat", "meanlon", "date", all columns in column_names 
        dict: dict mappings tuple of (lat,lon) -> (x,y)
        Must **not have** the following columns: "x", "y", "t"
    lat_dict: dict mapping lat->int
    lon_dict: dict mapping lon_int
    column_names: column names of data to include in data_array

    Returns:
    data_array: Array of shape (num_time_steps, num_data_types, num_unique_lat, num_unique_lon)
    """

    pad_val = -np.inf

    # TODO: 
    # - add date range argument
    # - replace all nans in ice data with zeros
    # convert nan to inf
    # df = df.replace({np.nan: np.inf})
    x_list = lat_dict.values()
    y_list = lon_dict.values()
    x_max = max(x_list)
    y_max = max(y_list)

    # get unique and ordered dates and dict
    date_array = df["date"].to_numpy()
    unique_dates = np.sort(np.unique(date_array))
    time_dict = {k: v for v, k in enumerate(unique_dates)}

    df["x"] = df["meanlat"].map(lat_dict)
    df["y"] = df["meanlon"].map(lon_dict)
    # create column that indicates relative time of data
    df["t"] = df["date"].map(time_dict)
    date_index = df.columns.get_loc("date")
    x_index = df.columns.get_loc("x")
    y_index = df.columns.get_loc("y")
    t_index = df.columns.get_loc("t")
    num_cols = len(column_names)
    column_indices = []
    for col in column_names:
        column_indices.append(df.columns.get_loc(col))

    # convert df to np array
    array_data = df.to_numpy()

    big_boy = np.full((len(unique_dates), num_cols, x_max+1, y_max+1), pad_val).astype('float')

    x_vals = array_data[:,x_index]
    y_vals = array_data[:,y_index]
    t_vals = array_data[:,t_index]
    num_indices = len(x_vals)
    col_holder = np.zeros((num_indices))
    index_array = np.array((t_vals, col_holder, x_vals,y_vals)).T

    for j in range(num_cols):
        col_data = array_data[:,column_indices[j]]
        # col_data = col_data.astype(float)
        index_array[:,1] = j
        index_array = index_array.astype(int)
        big_boy[index_array[:,0], index_array[:,1], index_array[:,2], index_array[:,3]] = col_data


    # for i in range(len(unique_dates)):
    #     print(i)
    #     # filter data to get info for curr date
    #     date_logic = array_data[:, date_index] == unique_dates[i]
    #     filtered_data = array_data[date_logic]

    #     # get list of tuples of x,y of data
    #     x_vals = filtered_data[:,x_index]
    #     y_vals = filtered_data[:,y_index]
    #     num_indices = len(x_vals)
    #     date_i = np.full((num_indices),i)
    #     col_holder = np.zeros((num_indices))
    #     index_array = np.array((date_i, col_holder, x_vals,y_vals)).T

    #     # get matching list of data values
    #     # put values into array for each data type
    #     for j in range(num_cols):
    #         col_data = filtered_data[:,column_indices[j]]
    #         index_array[:,1] = j
    #         index_array = index_array.astype(int)
    #         big_boy[index_array[:,0], index_array[:,1], index_array[:,2], index_array[:,3]] = col_data.astype(float)
        

    return big_boy

    

def create_mapping_dict(mapping_file):
    """
    Creates dict that maps lat,long to x,y coords in an array. This is fairly specific to
    the format of our data, so use caution.

    Args:
    mapping_file: path to file that contains mapping of gridid to lat, lon

    Returns:
    lat_dict: dict of form {lat: x} for every point in mapping_file
    lon_dict: dict of form {lon: y} for every point in mapping_file
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
    lat_dict = {}
    lon_dict = {}
    for i in range(len(x)):
        lat_dict[lat_vals[i]] = x[i]
        lon_dict[lon_vals[i]] = y[i]
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

    # fig.show()

    return lat_dict, lon_dict

def gen_data(big_data, t, w):
    """
    Generates array of data to be used for test/train.

    Args:
    big_data: output of convert()
    t: num time steps to include
    w: width of rect to cut around point

    Returns;
    data: array of shape (N, t, w*w-1, num data types)
    """

    # find indices of all chlorophyll values that are not nan
    # assume chlor data is first one (num_time_steps, num_data_types, num_unique_lat, num_unique_lon)

    # let's pretend I just have chlor

    # find location of missing values:
    # miss = np.isnan(big_data)
    miss = np.argwhere(np.isnan(big_data))
    t = 20
    n = 3
    means = []
    s = time.time()
    for point in miss:

        x = point[2]
        y = point[3]
        z = point[0]
        if z > t:
            slice = big_data[max(0,z-t):z,:, max(0,x-n):x+n+1,max(0,y-n):y+n+1]
            print(slice) # all nans?
            means.append(np.mean(slice))
    e = time.time()

    print(e-s)

    chlor_logic = big_data < np.inf 
    bounds_logic = big_data > -np.inf
    total_logic = np.logical_and(chlor_logic, bounds_logic)
    
    print("i want to be sedated")

    # slicing
    # t = 4
    # n = 1

    # x = 10
    # y = 10
    # z = 50 # time
    # slice = a[z-t:z,:, x-n:x+n+1,y-n:y+n+1]


# lat_dict, lon_dict = create_mapping_dict("data/Bering_full_grid_lookup_no_goa.RDS")
# df = merge_position("data/merged_sst_ice_chl_par_2003.RDS", "data/Bering_full_grid_lookup_no_goa.RDS")
# data = convert(df, lat_dict, lon_dict, ["chlorophyll"])

# np.save("data_test", data, allow_pickle=True)
# print("done")
gen_data(np.load("data_test.npy"), 8, 2)

