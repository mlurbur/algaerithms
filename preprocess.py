from operator import gt
import numpy as np
import pyreadr
import pandas as pd
from plotly.graph_objects import Figure, Scattergeo
import time
import datetime
from common import merge_position

# TODO: 
# - Make gen_data look forward in time (right now just looks back)
# - Figure out what to do with other data at ground truth point and time. 
#   As of now, this gets removed because the data array would have impossible shape
# - handle gt values at edge of "map". As of now, if window doesn't fit, the gt is skipped

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

    # get mapping's latitudinal and longitudinal values
    lat_vals, lon_vals = mapping[:,1], mapping[:,2]

    # build array of all points ((lat_val, lon_val) pairs)
    points = np.array((lat_vals,lon_vals)).T

    # get the max and min latitude/longitudes
    lat_arg, lon_arg = np.argsort(lat_vals), np.argsort(lon_vals)
    min_lat, max_lat = lat_vals[lat_arg[0]], lat_vals[lat_arg[-1]]
    min_lon, max_lon = lon_vals[lon_arg[0]], lon_vals[lon_arg[-1]]

    # define the individual grid's width and height
    grid_w, grid_h = 1.0125, 0.525

    # using min/max_lon, and grid_w, create array with longitudinal line values
    lon_lines = np.arange(start=min_lon-(grid_w/2), stop=max_lon+(grid_w/2),  step=grid_w)

    # generate latitudinal line values by calculating latitudinal midpoints between latitudinal values in first "column" (points left of second longitudinal line)
    points_in_first_col = list(filter(lambda point : point[1] < lon_lines[1], points))
    lat_lines = []
    sorted_points_lat = sorted(points_in_first_col, key=lambda x: x[0])
    for i in range(len(points_in_first_col) - 1):
        lat_lines.append((sorted_points_lat[i+1][0] + sorted_points_lat[i][0])/2)

    # recalculate lon_lines to get them more centered using a similar technique as above, this time, using points between the 7th and 8th latitudinal line
    points_in_seventh_row = list(filter(lambda point : point[0] > lat_lines[6] and point[0] < lat_lines[7], points))
    lon_lines = []
    sorted_points_lon = sorted(points_in_seventh_row, key=lambda x: x[1])
    for i in range(len(points_in_seventh_row) - 1):
        lon_lines.append((sorted_points_lon[i+1][1] + sorted_points_lon[i][1]) / 2)

    # potential todo: this can be improved (remedying duplicates) by making a
    # grid_coord ((0,1)) -> point_coord (((lat_val, lon_val), dist)) dict and
    # using a k-means-esque method to get the point_coord with the lowest dist
    # to each midpoint value (will also need a dict mapping grid_coord to
    # midpoint_coord for calculating distances). you iterate over all of the
    # first dictionary's keys (grid_coords) until a while condition checking
    # if there are still point_coords left to assign breaks. though with this
    # method, every point in the map would need to be gridded (otherwise the
    # loop would run infinitely)

    # ...
    # x = np.searchsorted(invert_lat, lat_midpoint_vals * -1)
    # y = np.searchsorted(lon_lines, lon_midpoint_vals * -1)
    #
    # 
    # while points:
    #     for grid_coord in zip(x,y):
    #         point_with_least_dist, dist = find_point_with_least_dist(grid_to_midpoint[grid_coord], points)
    #         if not grid_to_point[grid_coord]:
    #             grid_to_point[grid_coord] = (point_with_least_dist, dist)
    #             points.remove(point_with_least_dist)
    #             continue
    #         if grid_to_point[grid_coord][1] > dist:
    #             if grid_to_point[grid_coord] is not point_with_least_dist:
    #                 points.append(grid_to_point[grid_coord][0])
    #             grid_to_point[grid_coord] = (point_with_least_dist, dist)

    invert_lat = np.sort(np.array(lat_lines) * -1)
    x = np.searchsorted(invert_lat, lat_vals * -1)
    y = np.searchsorted(lon_lines, lon_vals)

    lat_dict = {}
    lon_dict = {}
    fml_dict = {}

    # populate coordinate dictionaries
    for i in range(len(x)):
        lat_dict[lat_vals[i]] = x[i]
        lon_dict[lon_vals[i]] = y[i]
        fml_dict[(lat_vals[i], lon_vals[i])] = (x[i], y[i])

    # generate text labels
    labely = []
    for i in range(len(lat_vals)):
        c = fml_dict[(lat_vals[i], lon_vals[i])]
        labely.append(str(c))

    # draw the figure
    fig = Figure()
    for i in range(len(lat_lines)):
        for k in range(len(lon_lines)-1):
            fig.add_trace(Scattergeo(
                lon = [lon_lines[k], lon_lines[k+1]],
                lat = [lat_lines[i], lat_lines[i]],
                mode = 'lines'))
    for j in range(len(lon_lines)):
        fig.add_trace(Scattergeo(
            lon = [lon_lines[j], lon_lines[j]],
            lat = [min_lat-grid_h, max_lat+grid_h],
            mode = 'lines'))

    fig.add_trace(Scattergeo(
        lon = lon_vals,
        lat = lat_vals,
        hovertext=labely))

    # uncomment to see the grid divisions ((x,y) coordinates of each point)
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

    QUESTION: Are these still the min_d'th and max_d'th julian days? Does the
    data need to be complete for this (365 measurements/year) to be true? - data just needs to start on Jan 1, which it does in our case

    Returns:
    data_array_list: list of arrays of shape (num_time_steps, 1, num_unique_lat, num_unique_lon)
    """

    PAD_VAL = -np.inf

    x_list = lon_dict.values()
    y_list = lat_dict.values()
    x_max = max(x_list)
    y_max = max(y_list)

    # get unique and ordered dates and dict
    date_array = df["date"].to_numpy()
    unique_dates = np.sort(np.unique(date_array))

    time_dict = {k: v for v, k in enumerate(unique_dates)}

    df["x"] = df["meanlon"].map(lon_dict)
    df["y"] = df["meanlat"].map(lat_dict)
    # replace all nans in ice column with zeros if "ice" is in column names
    if "ice" in column_names:
        df["ice"] = df["ice"].fillna(0)

    # create column that indicates relative time of data
    df["t"] = df["date"].map(time_dict)
    x_index = df.columns.get_loc("x")
    y_index = df.columns.get_loc("y")
    t_index = df.columns.get_loc("t")
    num_cols = len(column_names)
    column_indices = [df.columns.get_loc(col_name) for col_name in column_names]

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
        big_boy = np.full((total_days, 1, x_max+1, y_max+1), PAD_VAL).astype('float')
        col_data = array_data[:,column_indices[j]]

        index_array = index_array.astype(int)

        big_boy[index_array[:,0], 0, index_array[:,2], index_array[:,3]] = col_data
        time_trimmed = big_boy[min_d-1:max_d+1] 
        data_array_list.append(time_trimmed)
        
    return data_array_list


def fill_missing(big_data_chlor):
    """
    Attempts to fill in all missing (nan) values in big_data_chlor. Starting with a time window of 1 
    (1 time unit before and after missing value) and neighbors of 1. Loops through missing values, filling 
    as many as possible with the current combination of neighbors and time window until all are filled or
    max_time is reached. Does not use approximated values to approximate other values.

    **Must have at least 12 time periods of data to do anything useful**

    Args:
    big_data_chlor: chlorophyll data from convert_map_to_array of shape (t,0,x,y)

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
        if nan_points.shape[0] == 0:
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
    t: num time steps to include, only looks backward as of now.
    n: num neighbors to include (kinda)

    Returns:
    data_array: data of shape: (N, num_data_types, (t*(2n+1)^2) -1)
    gt_values: ground truth values of shape (N,)
    """

    # find location of ground truth values, assume that non data regions are filled with -np.inf
    gt_indices = np.argwhere(np.isfinite(original_chlor_data))
    _, _, max_x, max_y = original_chlor_data.shape
    data_array = []
    gt_values = []
    # get data slice from filled_data for each ground truth
    for point in gt_indices:
            x = point[2]
            y = point[3]
            z = point[0]
            if z >= t-1:
                data_bit = []
                for i in range(len(data_array_list)):
                    x_min = x-n
                    x_max = x+n
                    y_min = y-n
                    y_max = y+n
                    # check that slice is in bounds of data
                    if (x_min < 0) or (y_min < 0):
                        break
                    if (x_max > max_x-1) or (y_max > max_y-1):
                        break
                    slice = data_array_list[i][z-(t-1):z+1,0, x_min:x_max+1,y_min:y_max+1]
                    # skip if contains nan or inf
                    if (np.isnan(slice).any() or np.isinf(slice).any()):
                        break
                    # remove gt val
                    flat_slice = np.ndarray.flatten(slice)
                    middle = np.floor(((x_max+1)-x_min)/2)
                    # this calculation is shitty
                    j = -int(middle*((x_max+1)-x_min)+middle)-1
                    if i == 0: # only add gt of chlorophyll
                        gt_val = flat_slice[j]
                    data_bit.append(np.delete(flat_slice, j))
                        
                # only keep data_bit if all data were included
                if len(data_bit) == len(data_array_list):
                    data_bit = np.array(data_bit)
                    data_array.append(data_bit)
                    gt_values.append(gt_val)
    return np.asarray(data_array), np.asarray(gt_values)


def preprocess(data_file, mapping_file, params_of_interest, min_day, max_day, time_window, num_neighbors, save_data_file, save_gt_file):
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
    save_data_file: file path to save data
    save_gt_file: file path to save ground truth values
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
    data_array, gt_array= gen_data(data_array[0], data_array_with_filled, time_window, num_neighbors)

    print("Saving data to", save_data_file)
    np.save(save_data_file, data_array, allow_pickle=True)
    print("Saving ground truth values to", save_gt_file)
    np.save(save_gt_file, gt_array, allow_pickle=True)
    end = time.time()
    total_time = end - start
    print("Preprocessing complete. Took ", total_time, "seconds.")
    print("Total data points generated:", gt_array.shape[0])


def test_stuff():
    """
    Does some basic testing
    """

    # make dataframe for full 5x5 map with chlor and ice values for two days
    lon_dict = {}
    lat_vals = []
    lon_vals = []
    for i in range(5):
        lon_vals += list(range(5))
        lat_vals += [i for j in range(5)]
        lon_dict[i] = i

    arr1 = np.array([datetime.datetime(2000, 1, 1) for i in range(25)])
    arr2 = np.array([datetime.datetime(2000, 1, 2) for i in range(25)])
    dates = np.concatenate((arr1, arr2))
    chlor = [0 for i in range(25)] + [1 for i in range(25)]
    ice = [10 for i in range(25)] + [20 for i in range(25)]
    columnify = np.zeros((50,4))
    columnify[:, 0] = np.array(lon_vals+lon_vals)
    columnify[:, 1] = np.array(lat_vals+lat_vals)
    columnify[:, 2] = np.array(chlor)
    columnify[:, 3] = np.array(ice)
    df = pd.DataFrame(columnify, columns=["meanlat", "meanlon", "chlorophyll", "ice"])
    df["date"] = dates

    # whew, dataframe built
    full_map = convert_map_to_array(df, lon_dict, lon_dict, ["chlorophyll", "ice"])

    # make dataframe for 4x4 map with missing chlor and nan ice values and "land" for one day
    lon_dict = {0:0, 1:1, 2:2, 3:3}
    lat_vals = [0,0,0,0,1,1,1,1,2,2,3,3]
    lon_vals = [0,1,2,3,0,1,2,3,0,1,0,1]
    chlor = [1,1,np.nan, np.nan, 1,1,np.nan,np.nan,1,1,1,1]
    ice = [np.nan for i in range(12)]

    dates = np.array([datetime.datetime(2000, 1, 1) for i in range(12)])
    columnify = np.zeros((12,4))
    columnify[:, 0] = np.array(lon_vals)
    columnify[:, 1] = np.array(lat_vals)
    columnify[:, 2] = np.array(chlor)
    columnify[:, 3] = np.array(ice)
    df_2 = pd.DataFrame(columnify, columns=["meanlat", "meanlon", "chlorophyll", "ice"])
    df_2["date"] = dates

    # whew, dataframe built
    missing_map = convert_map_to_array(df_2, lon_dict, lon_dict, ["chlorophyll", "ice"])

    # check that chlorophyll and ice values are as expected for full map
    expected = np.array([np.full((1,5,5), 0), np.full((1,5,5), 1)])
    assert np.array_equal(full_map[0], expected), "Damn"
    expected = np.array([np.full((1,5,5), 10), np.full((1,5,5), 20)])
    assert np.array_equal(full_map[1], expected), "Damn"

    # check that chlor and ice values are as expected for missing and border map
    chlor = np.array([[[[1,1,np.nan, np.nan], [1,1,np.nan, np.nan], [1,1,-np.inf, -np.inf], [1,1,-np.inf, -np.inf]]]])
    ice = np.array([[[[0,0,0,0], [0,0,0,0], [0,0,-np.inf,-np.inf],[0,0,-np.inf,-np.inf]]]])

    assert np.allclose(missing_map[0], chlor, equal_nan=True), "Damn"
    assert np.array_equal(missing_map[1], ice), "Damn"

    print("Tests for convert_map_to_array passed")

    # testing fill_missing
    fake_missing = np.ones((20,1,3,3))
    ice = np.zeros((20,1,3,3))
    # create missing values
    fake_missing[10,0,0,:] = np.nan
    # create "land"
    fake_missing[:,0,2,:] = -np.inf
    filled = fill_missing(fake_missing)

    assert np.array_equal(fake_missing[10,0,0,:], [1,1,1]), "Damn"
    assert np.array_equal(fake_missing[:,0,2,:], fake_missing[:,0,2,:]), "Damn"
    
    print("Tests for fill_missing passed")

    # testing gen_data
    test_original_chlor = np.ones((5,1,3,3))
    test_original_chlor[0,0,1,1] = 1
    test_original_chlor[1,0,1,1] = 3
    test_original_chlor[2,0,1,1] = np.nan
    test_original_chlor[3,0,1,1] = 4
    test_original_chlor[4,0,1,1] = np.nan
    original_copy = np.copy(test_original_chlor)
    test_filled_chlor = fill_missing(test_original_chlor)

    data_array, gt_array= gen_data(np.ones((20,1,3,3)), [np.ones((20,1,3,3)), ice], 3, 1)

    assert data_array.shape == (18, 2, 26), "Damn"
    assert gt_array.shape == (18,), "Damn"

    data_array, gt_array = gen_data(original_copy, [test_filled_chlor, ice], 2, 1)
    assert np.array_equal(gt_array, np.array([3,4])), "Damn"

    # test that if land is in range, gt val is not generated
    test_original_chlor = np.ones((5,1,3,3))
    test_original_chlor[:,0,0,0] = -np.inf
    original_copy = np.copy(test_original_chlor)
    test_filled_chlor = fill_missing(test_original_chlor)
    data_array, gt_array = gen_data(original_copy, [test_filled_chlor], 2, 1)
    assert np.array_equal(gt_array, np.array([])), "Damn"

    print("Tests for gen_data passed")
    
test_stuff()

# example call to preprocess
preprocess("data/merged_sst_ice_chl_par_2003.RDS", "data/Bering_full_grid_lookup_no_goa.RDS", ["chlorophyll", "ice"],
    50, 244, 3, 1, "data.npy", "gt.npy")