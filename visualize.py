import pandas as pd
import pyreadr
import datetime
import plotly.express as px

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


def visualize_day(df, date, show_null=False):
    """
    Visualizes chlorophyll for a given day on a map. 

    args:
    df: pandas dataframe that has columns: date chlorophyll meanlat meanlon
    date: datetime.date format date
    show_null: boolean indicating whether to visualize null values, defaults to false

    returns:
    None
    """

    date_df = df[df["date"] == date]
    if not show_null:
        date_df = date_df[~df["chlorophyll"].isnull()]

    fig = px.scatter_mapbox(date_df, lat="meanlat", lon="meanlon", hover_name="chlorophyll", 
        color="chlorophyll", zoom=3, height=400, width=400)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()


# visualize data
visualize_day(merge_position("data/merged_sst_ice_chl_par_2003.RDS", 
    "data/Bering_full_grid_lookup_no_goa.RDS"), datetime.date(2003, 6, 1))