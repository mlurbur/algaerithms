import numpy as np
import pyreadr
import plotly.express as px
import plotly.graph_objects as go
import time

# TODO: 
# - Make gen_data look forward in time (right now just looks back)
# - Figure out what to do with other data at ground truth point and time. 
#   As of now, this gets removed because the data array would have impossible shape
# - handle gt values at edge of "map". As of now, if window doesn't fit, the gt is skipped

def merge_position(data_file, mapping_file):
    """
    Loads .RDS file as a pandas dataframe and adds lat and lon data column using the mapping file.

    Files in data dir should have the following columns in order: 
    date chl_n par_n chlorophyll par sst_n ice_n sst ice depth gridid index

    args:
    data_file: path to .RDS file containing data
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

    df = pyreadr.read_r(data_file)[None]

    # drop columns
    df = df.drop(["chl_n", "par_n", "sst_n", "ice_n", "index"], axis=1)
    # add meanlat meanlon columns
    df["meanlat"] = df["gridid"].map(grid_lat_dict)
    df["meanlon"] = df["gridid"].map(grid_lon_dict)

    return df

def create_mapping_dict(mapping_file):
    """
    Creates dict that maps lat,long to x,y coords in an array. This is specific to
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


def convert_map_to_array(df, lat_dict, lon_dict, column_names, min_d=1, max_d=365):
    """
    Converts df from merge_position to an array of form:
    [
        [chlorophyll_x0_y0, chlorophyll_x1_y0, ...]
        [chlorophyll_x0_y1, chlorophyll_x1_y1, ...]
        ...
    ]
    This array represents the spatial relationship between data. 
    The data at (0,0) is adjacent to (0,1) in lat/lon coordinates.
    
    If data is not in a perfect rectangle, values are padded with 
    -inf if there is no corresponding point in the dicts (land or 
    out of bounds of data). 

    If the column "ice" is in column names, replaces all nan values with zero.

    Note: If you want min_d and max d to be included in your dataset, you should add 
        some buffer otherwise the nan values will not be filled and the day will not end 
        up in the final output of gen_data.

    Args:
    df: dataframe with columns of data for one year (365 days).
        Must **have** the following columns: "meanlat", "meanlon", "date", all columns in column_names 
        Must **not have** the following columns: "x", "y", "t"
    lat_dict: dict mapping lat->int
    lon_dict: dict mapping lon->int
    column_names: column names of data for which to generate data_arrays 
    min_d: minimum julian day to include
    max_d: maximum julian day to include

    Returns:
    data_array_list: list of arrays of shape (num_time_steps, 1, num_unique_lat, num_unique_lon)
    """

    pad_val = -np.inf

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
    # replace all nans in ice column with zeros if "ice" is in column names
    if "ice" in column_names:
        df["ice"] = df["ice"].map({np.nan: 0})

    # create column that indicates relative time of data
    df["t"] = df["date"].map(time_dict)
    x_index = df.columns.get_loc("x")
    y_index = df.columns.get_loc("y")
    t_index = df.columns.get_loc("t")
    num_cols = len(column_names)
    column_indices = []
    for col in column_names:
        column_indices.append(df.columns.get_loc(col))

    # convert df to np array
    array_data = df.to_numpy()

    # check that min_day and max_day are valid
    if min_d < 1:
        print('min_day less than 1, setting to 1')
        min_d = 1
    total_days = len(unique_dates)
    if max_d > total_days:
        print('max_day greater than', total_days, ', setting to', total_days)
        max_d = total_days

    t_vals = array_data[:,t_index]
    x_vals = array_data[:,x_index]
    y_vals = array_data[:,y_index]

    # only include dates within specified range
    # t_vals = t_vals_orig[(t_vals_orig >= min_d-1) & (t_vals_orig <= max_d-1)]

    num_indices = len(x_vals)
    col_holder = np.zeros((num_indices))
    index_array = np.array([t_vals, col_holder,x_vals,y_vals]).T

    data_array_list = []
    for j in range(num_cols):
        big_boy = np.full((total_days, 1, x_max+1, y_max+1), pad_val).astype('float')
        col_data = array_data[:,column_indices[j]]

        index_array = index_array.astype(int)

        big_boy[index_array[:,0], 0, index_array[:,2], index_array[:,3]] = col_data
        time_trimmed = big_boy[min_d:max_d+1]
        data_array_list.append(time_trimmed)
        
    return data_array_list


def fill_missing(big_data_chlor):
    """
    Attempts to fill in all missing (nan) values in big_data_chlor. Starting with a time window of 1 
    (1 time unit before and after missing value) and neighbors of 1. Loops through missing values, filling 
    as many as possible with the current combination of neighbors and time window until all are filled or
    max_time is reached. Does not use approximated values to approximate other values.

    Args:
    big_data_chlor: chlorophyll data from convert_map_to_array

    Returns:
    filled_data: big_data_chlor with all possible nan values filled
    """

    filled_data = big_data_chlor

    # Define summarization regime, first pass will use window of 1, no neighbors
    # this fills ~90% of missing values in my experience
    n_regime = [0,0,0,1,1,1,1,2,2,2,2]
    t_regime = [1,2,3,4,5,6,7,8,9,10,11] 

    means = []
    nan_points = np.array([])
    for n, t in zip(n_regime, t_regime):
        # fill data with new means
        if len(nan_points) > 0: # handles first loop
            nan_z = nan_points[:,0]
            nan_x = nan_points[:,2]
            nan_y = nan_points[:,3]
            filled_data[nan_z, 0, nan_x, nan_y] = np.asarray(means)
        # generate new nan_indices
        nan_points = np.argwhere(np.isnan(filled_data))
        # clear means
        means = []
        # if no more missing values, stop
        if nan_points.shape == 0:
            break

        for point in nan_points:
            x = point[2]
            y = point[3]
            z = point[0]
            if z > t:
                slice = big_data_chlor[max(0,z-t):z+t+1,0, max(0,x-n):x+n+1,max(0,y-n):y+n+1]
                # take mean, ignore nan and -inf
                mean = np.ma.masked_invalid(slice).mean()
                # is mean is masked (no data in window and neighbors) still nan
                if np.ma.is_masked(mean):
                    means.append(np.nan)
                else:
                    means.append(mean)
            else:
                means.append(np.nan)
        print("Filling missing values. Missing values remaining:", len(nan_points))

    if len(nan_points) > 0:
        print("Filled as many missing values as possible. Missing values remaining:", len(nan_points))
    else:
        print("Filled all missing values!")
    
    return filled_data

def gen_data(original_chlor_data, data_array_list, t, n):
    """
    Generates array of data to be used for testing and training. Skips values if time and n window
    does not fit or if part of the window overlaps with land/non data area.

    Args:
    original_data: *just* non augmented chlorophyll data with ground truth values. From convert_map_to_array.
    data_array_list: [filled chlorophyll data, other data of same shape]
    t: num time steps to include, only looks backward as of now
    n: num neighbors to include (kinda)

    Returns:
    data: array of shape (N, t, (2n+1)(2n+1)-1, num data types)
    """

    # find location of ground truth values, assume that non data regions are filled with -np.inf
    g_t_indices = np.argwhere(original_chlor_data > -np.inf)
    _, _, max_x, max_y = original_chlor_data.shape
    data = []
    # get data slice from filled_data for each ground truth
    for point in g_t_indices:
            x = point[2]
            y = point[3]
            z = point[0]
            if z > t:
                data_bit = []
                for data_type in data_array_list:
                    x_min = x-n
                    x_max = x+n+1
                    y_min = y-n
                    y_max = y+n+1
                    # check that slice is in bounds of data
                    if (x_min < 0) or (y_min < 0):
                        data_bit = []
                        break
                    if (x_max > max_x-1) or (y_max > max_y-1):
                        data_bit = []
                        break
                    slice = data_type[z-t:z,0, x_min:x_max,y_min:y_max]
                    # skip if contains nan or inf
                    if (np.isnan(slice).any() or np.isinf(slice).any()):
                        data_bit = []
                        break
                    # remove gt val
                    flat_slice = np.ndarray.flatten(slice)
                    i = np.floor(len(flat_slice)/2)
                    data_bit.append(np.delete(flat_slice, int(i)))
                if data_bit != []:
                    data_bit = np.array(data_bit)
                    data.append(data_bit)

    return np.array(data)


def preprocess(data_file, mapping_file, params_of_interest, min_day, max_day, time_window, num_neighbors, save_file):
    """
    Performs data preprocessing. Saves final data in save_file.

    Args:
    data_file: path to .RDS file containing data
    mapping_file: path to .RDS file that contains mapping of gridid to lat, lon
    params_of_interest: column names of data to include. "chlorophyll" must be first.
    min_day: minimum julian day to include
    max_day: maximum julian day to include
    time_window: time window to include in data
    num_neighbors: num neighbors (kinda) to include in data
    save_file: file path to save data file
    """

    if params_of_interest[0] != "chlorophyll":
        raise ValueError("'chlorophyll' is not first value in params_of_interest.")

    start = time.time()
    print('Starting data preprocessing...')
    print('Creating mapping dictionary...')
    lat_dict, lon_dict = create_mapping_dict(mapping_file)

    print('Merging positional data...')
    df = merge_position(data_file, mapping_file)

    print('Converting data to array form representing spatial layout...')
    data_array = convert_map_to_array(df, lat_dict, lon_dict, params_of_interest, min_d=min_day, max_d=max_day)

    print("Filling in missing chlorophyll values...")
    filled_data = fill_missing(data_array[0])

    print("Generating data set from valid ground truth values...")
    # replace original data with filled data in data_array
    data_array_with_filled = data_array
    data_array_with_filled[0] = filled_data
    data = gen_data(data_array[0], data_array_with_filled, time_window, num_neighbors)

    print("Saving data to", save_file)
    np.save(save_file, data, allow_pickle=True)
    end = time.time()
    total_time = end - start
    print("Preprocessing complete. Took ", total_time, "seconds.")
    print("Total data points generated:", data.shape[0])

# example call to preprocess
preprocess("data/merged_sst_ice_chl_par_2003.RDS", "data/Bering_full_grid_lookup_no_goa.RDS", ["chlorophyll", "ice"],
    50, 244, 3, 1, "data.npy")

