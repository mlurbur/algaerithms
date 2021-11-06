import pyreadr
import numpy as np

result = pyreadr.read_r('/Users/mlurbur/downloads/merged_sst_ice_chl_par_2021.RDS')


df1 = result[None]
head = df1.head()
for i in head:
    print(i)


