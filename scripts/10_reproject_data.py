import datashader as das
from pyhdf.SD import SD, SDC
import os
import cv2
import pandas as pd
import numpy as np

root = 'xxxx./'

fl = [f'{root}modis/hdf/...{i}' + i for i in os.listdir( f'{root}modis/hdf/...' )]

for i, filename in enumerate(fl):
    if i%10==0: print(i)
    newname = filename.split('/')[-1].strip('.hdf')
    if os.path.exists(f'{root}modis/nc/{newname}.nc') == False:

        #### re-project to even lat lon with datashader ##
        hdf = SD(filename, SDC.READ)
        data = hdf.select('Cloud_Water_Path').get()
        lat = hdf.select('Latitude').get()
        lon = hdf.select('Longitude').get()
        lat = cv2.resize(lat,(data.shape[1], data.shape[0])).ravel()  #might want a better solution here
        lon = cv2.resize(lon,(data.shape[1], data.shape[0])).ravel()
        df = pd.DataFrame({'lat':lat,'lon':lon,'clwp':data.ravel()})


        #### re-project to even lat lon with datashader ##
        cvs = das.Canvas(plot_width=2000, plot_height=2000)
        agg = cvs.points(df, 'lon', 'lat', das.mean('clwp'))
        agg.to_netcdf(f'{root}modis/nc/{newname}.nc)

