#!pip install --upgrade pyhdf
import os, cv2, numpy as np
from pyhdf.SD import SD, SDC

source = './workspace/data/raw/MOD06_L2/'
dest = './workspace/data/processed/patch_256/'
file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source) for f in filenames if os.path.splitext(f)[1] == '.hdf']

patch = 256
stride = 200
file_list = file_list[:2000]

for filename in file_list:
    name = filename.split('/')[-1].strip('.hdf')
    if os.path.exists(f'{dest}{name}_00.jpg') == False:
        print(name)
        out = np.empty((2030, 1350, 3), dtype=np.float32)
        hdf = SD(filename, SDC.READ)
        wp = hdf.select('Cloud_Water_Path').get() / 6000
        if wp.max() > 0:
            tt = hdf.select('cloud_top_temperature_1km').get() / 19000
            er = hdf.select('Cloud_Effective_Radius').get() / 6000
            out[:,:,0] = wp[:2030,:1350]
            out[:,:,1] = tt[:2030,:1350]
            out[:,:,2] = er[:2030,:1350]
            out[out>1] = 1.0
            out[out<0] = 0.0
            out = out * 255

            k = 0
            for i in range(np.floor(1350/stride).astype(int)):
                for j in range(np.floor(2030/stride).astype(int)):

                    cv2.imwrite(f'{dest}{name}_{k:02}.jpg', out[j*stride:j*stride+patch,i*stride:i*stride+patch,:])
                    k +=1


# for radiances #
for filename in file_list:
    name = filename.split('/')[-1].strip('.hdf')
    if os.path.exists(f'{dest}{name}_00.jpg') == False:
        print(name)
        out = np.empty((2030, 1350, 3), dtype=np.float32)
        hdf = SD(filename, SDC.READ)
        wp = hdf.select('Cloud_Water_Path').get() / 6000
        if wp.max() > 0:
            tt = hdf.select('cloud_top_temperature_1km').get() / 19000
            er = hdf.select('Cloud_Effective_Radius').get() / 6000
            out[:,:,0] = wp[:2030,:1350]
            out[:,:,1] = tt[:2030,:1350]
            out[:,:,2] = er[:2030,:1350]
            out[out>1] = 1.0
            out[out<0] = 0.0
            out = out * 255

            k = 0
            for i in range(np.floor(1350/stride).astype(int)):
                for j in range(np.floor(2030/stride).astype(int)):

                    cv2.imwrite(f'{dest}{name}_{k:02}.jpg', out[j*stride:j*stride+patch,i*stride:i*stride+patch,:])
                    k +=1