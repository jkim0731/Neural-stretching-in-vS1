# -*- coding: utf-8 -*-
"""
Test the effect of blank edge in mice 52-56 for registration quality.
This is likely because of different bidirectional alignment during imaging.
I used fixed values to remove left (10 pixels) and right edges (100 pixels). Cf: convert_to_H5_JK.m
Now, 52-56 needs right edge removal for 70 pixels. 
I am not going to check if left edge was removed too much, since it will take too long to convert
sbx files to h5 files and running registration all again.
I'll just check if removing the right edge blank makes the registration better (they are all in
ROI parameter search process already, and almost done...)

But how? What is the metric to compare the registration quality?

2021/08/05 JK
"""

#%% Basic settings
import h5py
from matplotlib import pyplot as plt
import numpy as np
from suite2p.run_s2p import run_s2p, default_ops
import os, glob, shutil
import napari
from suite2p.io.binary import BinaryFile

import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'
fastDir = 'C:/JK/' # This better be in SSD
# mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
# refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
# zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
# freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]

rightBlank = 70 # Number of pixels to remove from the right side

ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 100
ops['batch_size'] = 5000
ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True

#%% Choose the test session, and convert the h5 file to a test file after removing 70 pixels on the right side
'''
BE EXTREMELY CAREFULL!! 
OG FILE CAN BE OVERWRITTEN!!
'''
mouse = 52
pn = 4
session = 29 # select from 1,4, 6:29, only from regular sessions (spont and piezo sessions need much more computation, and it's not worth it)
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'

fnOG = f'{mouse:03}_{session:03}_000_plane_{pn}.h5'
fnNew = f'{mouse:03}_{session:03}x_000_plane_{pn}.h5'

with h5py.File(f'{planeDir}{fnOG}', 'r') as f:
    data = f['data'][:,:,:-rightBlank]
with h5py.File(f'{planeDir}{fnNew}', 'w') as f:
    f['data'] = data
# plt.imshow(data.astype(np.float64).mean(axis=0))

#%% Check new h5 file
with h5py.File(f'{planeDir}{fnOG}', 'r') as f:
    data = np.array(f['data'])
with h5py.File(f'{planeDir}{fnNew}', 'r') as f:
    newdata = np.array(f['data'])

c = data[:newdata.shape[0], :newdata.shape[1], :newdata.shape[2]] - newdata
print(any(c.flatten())) # This should be 'False'
del c

#%% Run suite2p on the new file
db = {'h5py': f'{planeDir}{fnNew}',
    'h5py_key': ['data'],
    'data_path': [],
    'save_path0': planeDir,
    'save_folder': f'{session:03}x',
    'fast_disk': f'{fastDir}',
    'roidetect': False,
}
run_s2p(ops,db)
rawbinFn = f'{planeDir}{session:03}x/plane0/data_raw.bin'
os.remove(rawbinFn)

#%% Compare the quality to the OG file
# How....?

#%% (1) Visual inspection of the mean image
# Is there any difference?

opsfnOG = f'{planeDir}{session:03}/plane0/ops.npy'
binfnOG = f'{planeDir}{session:03}/plane0/data.bin'

opsfnNew = f'{planeDir}{session:03}x/plane0/ops.npy'
binfnNew = f'{planeDir}{session:03}x/plane0/data.bin'

opsOG = np.load(opsfnOG, allow_pickle=True).item()
opsNew = np.load(opsfnNew, allow_pickle=True).item()

OGmimg = opsOG['meanImg'][:,:-rightBlank]
newmimg = opsNew['meanImg']

viewer = napari.Viewer()

viewer.add_image(np.stack((OGmimg, newmimg)))

miximg = np.zeros((OGmimg.shape[0],OGmimg.shape[1],3), np.int32)
miximg[:,:,0] = OGmimg.astype(np.int32)
miximg[:,:,2] = OGmimg.astype(np.int32)
miximg[:,:,1] = newmimg.astype(np.int32)
viewer.add_image(miximg, rgb=True)

diffImg = OGmimg - newmimg
viewer.add_image(diffImg)

#%% (2) Correlation value between each frame and the ref img
ycrop = 30
xcrop = 60
template = opsOG['refImg'][ycrop:-ycrop,xcrop:-xcrop-rightBlank].flatten()
corrOG = np.zeros((opsOG['nframes'],))
with BinaryFile(Ly = opsOG['Ly'], Lx = opsOG['Lx'], read_filename = binfnOG) as f:
    data = f.data[:,ycrop:-ycrop,xcrop:-xcrop-rightBlank]
    for i in range(data.shape[0]):
        frame = data[i,:,:]
        corrOG[i] = np.corrcoef(frame.flatten(),template)[0,1]

template = opsNew['refImg'][ycrop:-ycrop,xcrop:-xcrop].flatten()
corrNew = np.zeros((opsNew['nframes'],))
with BinaryFile(Ly = opsNew['Ly'], Lx = opsNew['Lx'], read_filename = binfnNew) as f:
    data = f.data[:,ycrop:-ycrop,xcrop:-xcrop]
    for i in range(data.shape[0]):
        frame = data[i,:,:]
        corrNew[i] = np.corrcoef(frame.flatten(),template)[0,1]
        
meanVal = (corrOG - corrNew).mean()
stdVal = (corrOG - corrNew).std()
plt.plot(corrOG)
plt.plot(corrNew)
plt.legend(['OG', 'New'])
plt.xlabel('Frame')
plt.ylabel('Intensity correlation')
plt.title(f'JK{mouse:03} plane{pn} session {session:03}\n OG - New: {meanVal:.4f} $\pm$ {stdVal:.4f}')

#%% Showing the OG mean image
# napari.view_image(opsOG['meanImg'])
plt.imshow(opsOG['meanImg'], cmap='gray')