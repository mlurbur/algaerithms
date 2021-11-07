# Algaerithms

Definitions:
- sst: sea surface temperature
- par: photosynthetically active radiation

## Downloading data

Run `python download_data.py`

This will create a `data/` folder containing 19 `.RDS` files (R data files) containing, among other things, chlorophyll, depth, sst, ice and par data. Each file contains a year's data. There will also be a file that maps grid id to mean latitude and longitude.

## Inspecting data

Run `python visualize.py`

This will visualize the chlorophyll data for a particular day.

## Data info

The data is summarized over eight day periods and grouped into grid blocks. For a better visualization of the grid, set `show_null=True` in `visualize_day()` when running `visualize.py`.

Ice, par and sst values are present for every time period. An ice value of null/nan means no ice was present.

