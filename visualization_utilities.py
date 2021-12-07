import numpy as np
import plotly.express as px
import pyreadr
import imageio
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocess_utilities import create_mapping_dict, merge_position, merge_position, convert_map_to_array, fill_missing


def fill_with_model(model, data_file, mapping_file, min_day, max_day, t, n):
    """
    who knows
    """
    lat_dict, lon_dict = create_mapping_dict(mapping_file)

    print('Merging positional data...')
    df = merge_position(data_file, mapping_file)

    print('Converting data to array form representing spatial layout...')
    data_array = convert_map_to_array(df, lat_dict, lon_dict, ["chlorophyll"], min_d=min_day, max_d=max_day)
    print("Filling in missing chlorophyll values...")
    filled_data = fill_missing(data_array[0])
    data_array_with_filled = np.copy(data_array)
    data_array_with_filled[0] = filled_data
    model_data, valid_indices, percent_filled = find_predictable(data_array[0], data_array_with_filled, t, n)

    predicted_values = model.call(model_data, None)
    data_with_predicted = np.copy(data_array[0])
    for point, val in zip(valid_indices, predicted_values):
        data_with_predicted[point[0], point[1], point[2]] = val
    
    visualize_two_as_gif(data_array[0],data_with_predicted, 'wooweewah.gif')
    print("Filled", 100*percent_filled, "percent of missing values.")



def find_predictable(original_chlor_data, data_array_list, t, n):
    # for all nan in original, if n fits and t fits the fillable
    # return batch of data and indices of missing vals

    missing_indices = np.argwhere(np.isnan(original_chlor_data))
    _, max_x, max_y = original_chlor_data.shape
    data_array = []
    valid_indices = []
    # get data slice from filled_data for each ground truth
    for point in tqdm(missing_indices):
            x = point[1]
            y = point[2]
            z = point[0]
            if z >= t:
                # loop through all data types for a given time period
                chlor_valid = False
                data_count = 0
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
                    slice = data_array_list[i][z-t:z, x_min:x_max+1,y_min:y_max+1]
                    # skip if contains nan or inf
                    if (np.isnan(slice).any() or np.isinf(slice).any()):
                        break
                    
                    flatter_slice = np.reshape(slice, (t, slice.shape[1] * slice.shape[2]))
                    if i == 0: # 
                        fat_slice = flatter_slice
                        chlor_valid = True
                        data_count += 1
                    elif chlor_valid:
                        fat_slice = np.concatenate((fat_slice,flatter_slice), axis=1)
                        data_count += 1
                        
                # only keep data_bit if all data were included
                if data_count == len(data_array_list):
                    data_array.append(fat_slice)
                    valid_indices.append(point)

    return np.array(data_array), np.array(valid_indices), np.round(len(valid_indices)/len(missing_indices),2)


def visualize_two_as_gif(data1, data2, gif_path):
    """
    Creates gif from data. Assumes data is of shape (n, num_lon, num_lat) where n is num time steps

    Args:
    data1: array of data of shape (num_lon, num_lat)
    data2: array of data of shape (num_lon, num_lat)
    gif_path: path to save gif. Will also be used to save images being used for gif 
    """
    images = []
    print("Creating animation")
    for i in tqdm(range(len(data1))):
        file_path = "fig" + str(i) + ".png"
        compare_two(data1[i], data2[i], file_path) 
        images.append(imageio.imread(file_path))
        os.remove(file_path)
    imageio.mimsave(gif_path, images)

def compare_two(data1, data2, file_path):
    """
    plots two graphs side by side. data1 and data2 must have same dimensions.

    Args:
    data1: array of data of shape (num_lon, num_lat)
    data2: array of data of shape (num_lon, num_lat)
    file_path: path to save image of fig
    """

    x_1 = []
    x_2 = []
    y_1 = []
    y_2 = []
    height_1 = []
    height_2 = []
    max_height = 30

    h, w = data1.shape

    # loop through 2d array
    for i in range(h):
        for j in range(w):
            d_1 = min(data1[i,j], max_height)
            d_2 = min(data2[i,j], max_height)
            # only add if not missing and not land
            if np.isfinite(d_1):
                height_1.append(d_1)
                x_1.append(i)
                y_1.append(j)
            
            if np.isfinite(d_2):
                height_2.append(d_2)
                x_2.append(i)
                y_2.append(j)

    # let's make some 3d bar graphs
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.title.set_text('Original data')
    ax2.title.set_text('Filled data')
    ax1.set_zlabel("Chlorphyll")
    ax2.set_zlabel("Chlorphyll")
    ax1.axes.set_zlim3d(bottom=0, top=max_height) 
    ax2.axes.set_zlim3d(bottom=0, top=max_height)
    ax1.axes.set_xlim3d(left=0, right=23) 
    ax2.axes.set_xlim3d(left=0, right=23) 
    ax1.axes.set_ylim3d(bottom=0, top=23) 
    ax2.axes.set_ylim3d(bottom=0, top=23)
    if len(x_1) > 0:
        ax1.bar3d(x_1, y_1, np.zeros(len(height_1)), 1, 1, height_1, shade=True)
    if len(x_2) > 0:
        ax2.bar3d(x_2, y_2, np.zeros(len(height_2)), 1, 1, height_2, shade=True)

    # save larger figure
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(file_path, dpi=100)
    plt.close()
    

def visualize_day_from_array(data, lat_dict, lon_dict, image_path=None):
    """
    Visualizes a single day of data.

    Args:
    data: array of data of shape (num_lon, num_lat)
    lat_dict: dict of form {lat: x} for every point in mapping_file
    lon_dict: dict of form {lon: y} for every point in mapping_file
    image_path: Path to save image. In none, image is not saved

    Returns:
    Not sure yet, image?
    """
    new_lat_dict = {}
    new_lon_dict = {}

    for i in lat_dict.items():
        if i[1] in new_lat_dict:
            continue
        else:
            new_lat_dict[i[1]] = i[0]

    for i in lon_dict.items():
        if i[1] in new_lon_dict:
            continue
        else:
            new_lon_dict[i[1]] = i[0]

    lat_list = sorted(new_lat_dict.items(), key=lambda x: x[1])
    lon_list = sorted(new_lon_dict.items(), key=lambda x: x[1])

    lat_vals = []
    lon_vals = []
    data_vals = []

    # loop through 2d array
    for i in lon_list:
        for j in lat_list:
            d = data[i[0],j[0]]
            # only add if not missing and not land
            if np.isfinite(d):
                data_vals.append(d)
                lon_vals.append(i[1])
                lat_vals.append(j[1])

    # let's make some 3d bar graphs

    # here lies the code for map plotting. RIP.

    fig = px.scatter_mapbox(lat=lat_vals, lon=lon_vals, hover_name=data_vals, 
        color=data_vals, zoom=3, height=400, width=400)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    if image_path != None:
        fig.write_image(image_path)
    else:
        fig.show()
