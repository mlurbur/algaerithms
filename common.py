import numpy as np
import plotly.express as px
import pyreadr
import imageio
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


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
