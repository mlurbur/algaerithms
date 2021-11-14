import pyreadr
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys

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


def visualize_data(df, date, show_null=False):
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

    fig = px.scatter_mapbox(date_df, lat="meanlat", lon="meanlon", hover_name="gridid", 
        color="chlorophyll", zoom=3, height=400, width=400)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

def visualize_grid(mapping_file):
    """
    Visualizes spatial grid 

    Args:
    mapping_file: path to file that contains mapping of gridid to lat, lon
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

    # correct lats
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
    # add last lines
    lats.append(sorty[-1][0]+grid_h)

    # correct lons

    filtered_points = []
    sorted_lats = sorted(lats)
    for p in points:
        # check long is less than bound
        if (p[0] > sorted_lats[6]) and (p[0] < sorted_lats[7]):
            filtered_points.append(p)

    lons = []
    max = len(filtered_points)-1
    sorty = sorted(filtered_points, key=lambda x: x[1])
    l=0
    while l < max:
        mid = (sorty[l+1][1] + sorty[l][1])/2
        lons.append(mid)
        l+=1

    x = np.searchsorted(lats, lati)
    y = np.searchsorted(lons, long)

    fml_dict = {}
    for i in range(len(x)):
        fml_dict[(lati[i], long[i])] = (x[i], y[i])

    labely = []
    for i in range(len(lati)):
        c = fml_dict[(lati[i], long[i])]
        labely.append(str(c))    

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
    lat = lati,
    hovertext=labely))

    fig.show()


def main():

    # collect and parse inputs
    DATE_ARG_LENGTH = 10 # format is MM/DD/YY, e.g. 06/01/2003
    USAGE_ERROR_MSG = "    USAGE: python visualize.py <MM/DD/YYYY> [null]"
    if len(sys.argv) < 2:
        print("visualize.py: requires a date argument")
        print(USAGE_ERROR_MSG)
        exit()
    date_arg = sys.argv[1]
    if len(date_arg) != DATE_ARG_LENGTH:
        print("visualize.py: incorrect date format")
        print(USAGE_ERROR_MSG)
        exit()
    date = datetime.datetime.strptime(date_arg, "%m/%d/%Y").date()
    show_null = False
    if len(sys.argv) > 2:
        show_null = sys.argv[2] == "null"

    # visualize data
    visualize_data(merge_position("data/merged_sst_ice_chl_par_" + str(date.year)
        + ".RDS", "data/Bering_full_grid_lookup_no_goa.RDS"), date, show_null)

if __name__ == '__main__':
    main()
