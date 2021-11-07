import numpy as np
import pyreadr
import os

def process_raw(data_file, mapping_file):
    """
    Converts data to numpy array of stuff
    files in data dir should have the following columns in order: date chl_n par_n chlorophyll par sst_n ice_n sst ice depth gridid index
    """

    # for file in os.listdir(data_dir):
    #     if not file.endswith(".RDS"):
    #         continue
    grid_lat_dict = {} # dict that maps gridid -> lat
    grid_lon_dict = {} # dict that maps gridid -> lon
    mapping = pyreadr.read_r(mapping_file)[None].to_numpy()

    for i in mapping:
        grid_lat_dict[i[0]] = i[1]
        grid_lon_dict[i[0]] = i[2]

    df = pyreadr.read_r(data_file)[None]
    # drop columns
    df = df.drop(["chl_n", "par_n", "sst_n", "ice_n", "index"], axis=1)
    # add lat/lon columns
    df["meanlat"] = df["gridid"].map(grid_lat_dict)
    df["meanlon"] = df["gridid"].map(grid_lon_dict)

    arr = df.to_numpy()
    print(arr)






process_raw('/Users/mlurbur/downloads/merged_sst_ice_chl_par_2021.RDS', '/Users/mlurbur/downloads/Bering_full_grid_lookup_no_goa.RDS')

