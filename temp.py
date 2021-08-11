# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:37:54 2021

@author: shires
"""

import h5py
import numpy as np
import napari
#%%
imstack = []

sns = np.arange(201,209,dtype = int)
for sn in sns:
    fn = f'D:/TPM/JK/h5/027/plane_1/027_9999_{sn}_plane_1.h5'
    f = h5py.File(fn, 'r')
    data = f['data']
    numFrames, height, width = data.shape
    for i in range(numFrames):
        img = data[i,:,:]
        imstack.append(img)

#%%
viewer = napari.view_image(imstack)