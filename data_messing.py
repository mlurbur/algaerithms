import pyreadr
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

result = pyreadr.read_r('/Users/mlurbur/downloads/Bering_full_grid_lookup_no_goa.RDS')


df = result[None]

fig = px.scatter_mapbox(df, lat="meanlat", lon="meanlon", hover_name="gridid", color_discrete_sequence=["fuchsia"], zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
