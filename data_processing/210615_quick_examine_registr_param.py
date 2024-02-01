# -*- coding: utf-8 -*-
"""
Quickly examine registration parameters after selecting FOV-matching sessions

2021/06/15 JK
"""

import h5py
import numpy as np
from suite2p.run_s2p import run_s2p, default_ops
import os, glob, shutil
import matplotlib.pyplot as plt
from skimage import exposure
import napari
import ffmpeg
import cv2
from suite2p.io.binary import BinaryFile

# CLAHE each mean images
def clahe_each(img):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img)) * (2**16-1)
    newimg = exposure.equalize_adapthist(newimg.astype(np.uint16))
    return newimg

h5Dir = 'D:/TPM/JK/h5/'
mice = [25,27,30,36,37,38,39,41,52,53,54,56]
zoom = [2,   2,   2,   1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
freq = [7.7, 7.7, 7.7, 7.7, 6.1, 6.1, 6.1, 6.1, 7.7, 7.7, 7.7, 7.7]
#%%
mi = 1
planes = [1,2,3,4]
framesPerSession = 20
# removeSessionNames = ['025_016_', '025_018_', '025_5555_001', '025_5555_103', '025_9999_1']
removeSessionNames = ['027_007_', '027_025_', '027_5555_001', '027_5555_104', '027_9999_1']

ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 100
ops['batch_size'] = 5000
ops['maxregshift'] = 0.2
ops['block_size'] = [64, 64]
ops['maxregshiftNR'] = 20

ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True

ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']


#%%
mouse = mice[mi]

planei = 1
mouseDir = f'{h5Dir}{mouse:03}/'
planeDir = f'{mouseDir}plane_{planei}/'
tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{planei}.h5')
repFn = [fn for fn in tempFnList if all(rsn not in fn for rsn in removeSessionNames)]

#%% pick random 'framesPerSession' frames from each session, make a new h5 file
f = h5py.File(repFn[0], 'r')
data = f['data']
numFrames, height, width = data.shape
wfn = f'{planeDir}selected.h5'
newdata = np.zeros((len(repFn)*framesPerSession, height, width), dtype = np.uint16)
for i, fn in enumerate(repFn):
    f = h5py.File(fn, 'r')
    data = f['data']
    numFrames, height, width = data.shape
    frames = np.random.choice(numFrames,framesPerSession)
    for j in range(len(frames)):
        newdata[i*framesPerSession+j, :, :] = data[frames[j],:,:]
#%%
with h5py.File(wfn, 'w') as wf:
    wf.create_dataset('data', data=newdata, dtype='uint16')

    

#%% Run this new h5 file
db = {'h5py': wfn,
        'h5py_key': ['data'],
        'data_path': [],
        'save_path0': planeDir,
        'save_folder': 'test',
        'fast_disk': f'{planeDir}/test',
        'roidetect': False,
        'testFileList': repFn,
        'framesPerSession': framesPerSession,
    }
run_s2p(ops,db)

#%% import the registered data.bin
dataDir = f'{planeDir}/test/plane0/'
binfn = f'{dataDir}data.bin'
opsfn = f'{dataDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()
Ly = ops['Ly']
Lx = ops['Lx']
nframes = ops['nframes']
framesPerSession = ops['framesPerSession']
numSessions = int(nframes/framesPerSession)
# perFileMeanImg = np.zeros(shape = (Ly,Lx,numSessions))

#%%napari view
viewer = napari.Viewer()
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
        frames = f.ix(indices=inds).astype(np.float32)
        # perFileMeanImg[:,:,i] = frames.mean(axis=0)
        viewer.add_image(frames.mean(axis=0))


#%%
ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 100
ops['batch_size'] = 5000
ops['maxregshift'] = 0.2
ops['block_size'] = [64, 64]
ops['maxregshiftNR'] = 20

ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True

ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']




testDn = 'test2'
ops['snr_thresh'] = 1

db = {'h5py': wfn,
        'h5py_key': ['data'],
        'data_path': [],
        'save_path0': planeDir,
        'save_folder': testDn,
        'fast_disk': f'{planeDir}/{testDn}',
        'roidetect': False,
        'testFileList': repFn,
        'framesPerSession': framesPerSession,
    }
run_s2p(ops,db)

dataDir = f'{planeDir}/{testDn}/plane0/'
binfn = f'{dataDir}data.bin'
opsfn = f'{dataDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()
Ly = ops['Ly']
Lx = ops['Lx']
nframes = ops['nframes']
framesPerSession = ops['framesPerSession']
numSessions = int(nframes/framesPerSession)
# perFileMeanImg = np.zeros(shape = (Ly,Lx,numSessions))
viewer = napari.Viewer()
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
        frames = f.ix(indices=inds).astype(np.float32)
        # perFileMeanImg[:,:,i] = frames.mean(axis=0)
        viewer.add_image(frames.mean(axis=0))
        
        
        
        
#%%
ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 100
ops['batch_size'] = 5000
ops['maxregshift'] = 0.2
ops['block_size'] = [64, 64]
ops['maxregshiftNR'] = 20

ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True

ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']




testDn = 'test3'
ops['snr_thresh'] = 1
ops['nonrigid'] = False

db = {'h5py': wfn,
        'h5py_key': ['data'],
        'data_path': [],
        'save_path0': planeDir,
        'save_folder': testDn,
        'fast_disk': f'{planeDir}/{testDn}',
        'roidetect': False,
        'testFileList': repFn,
        'framesPerSession': framesPerSession,
    }
run_s2p(ops,db)

dataDir = f'{planeDir}/{testDn}/plane0/'
binfn = f'{dataDir}data.bin'
opsfn = f'{dataDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()
Ly = ops['Ly']
Lx = ops['Lx']
nframes = ops['nframes']
framesPerSession = ops['framesPerSession']
numSessions = int(nframes/framesPerSession)
# perFileMeanImg = np.zeros(shape = (Ly,Lx,numSessions))
viewer = napari.Viewer()
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
        frames = f.ix(indices=inds).astype(np.float32)
        # perFileMeanImg[:,:,i] = frames.mean(axis=0)
        viewer.add_image(frames.mean(axis=0))


#%%
ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 100
ops['batch_size'] = 5000
ops['maxregshift'] = 0.2

ops['maxregshiftNR'] = 20

ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True

ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']




testDn = 'test4'
ops['snr_thresh'] = 1
ops['nonrigid'] = True
ops['block_size'] = [128, 128]

db = {'h5py': wfn,
        'h5py_key': ['data'],
        'data_path': [],
        'save_path0': planeDir,
        'save_folder': testDn,
        'fast_disk': f'{planeDir}/{testDn}',
        'roidetect': False,
        'testFileList': repFn,
        'framesPerSession': framesPerSession,
    }
run_s2p(ops,db)

dataDir = f'{planeDir}/{testDn}/plane0/'
binfn = f'{dataDir}data.bin'
opsfn = f'{dataDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()
Ly = ops['Ly']
Lx = ops['Lx']
nframes = ops['nframes']
framesPerSession = ops['framesPerSession']
numSessions = int(nframes/framesPerSession)
# perFileMeanImg = np.zeros(shape = (Ly,Lx,numSessions))
viewer = napari.Viewer()
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
        frames = f.ix(indices=inds).astype(np.float32)
        # perFileMeanImg[:,:,i] = frames.mean(axis=0)
        viewer.add_image(frames.mean(axis=0))


#%%
ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 100
ops['batch_size'] = 5000
ops['maxregshift'] = 0.2

ops['maxregshiftNR'] = 20

ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True

ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']




testDn = 'test5'
ops['snr_thresh'] = 1.2
ops['nonrigid'] = True
ops['block_size'] = [128, 128]

db = {'h5py': wfn,
        'h5py_key': ['data'],
        'data_path': [],
        'save_path0': planeDir,
        'save_folder': testDn,
        'fast_disk': f'{planeDir}/{testDn}',
        'roidetect': False,
        'testFileList': repFn,
        'framesPerSession': framesPerSession,
    }
run_s2p(ops,db)

dataDir = f'{planeDir}/{testDn}/plane0/'
binfn = f'{dataDir}data.bin'
opsfn = f'{dataDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()
Ly = ops['Ly']
Lx = ops['Lx']
nframes = ops['nframes']
framesPerSession = ops['framesPerSession']
numSessions = int(nframes/framesPerSession)
# perFileMeanImg = np.zeros(shape = (Ly,Lx,numSessions))
viewer = napari.Viewer()
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
        frames = f.ix(indices=inds).astype(np.float32)
        # perFileMeanImg[:,:,i] = frames.mean(axis=0)
        viewer.add_image(frames.mean(axis=0))


#%%
ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 100
ops['batch_size'] = 5000
ops['maxregshift'] = 0.2



ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True

ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']




testDn = 'test6'
ops['snr_thresh'] = 1.2
ops['nonrigid'] = True
ops['block_size'] = [128, 128]
ops['maxregshiftNR'] = 10

db = {'h5py': wfn,
        'h5py_key': ['data'],
        'data_path': [],
        'save_path0': planeDir,
        'save_folder': testDn,
        'fast_disk': f'{planeDir}/{testDn}',
        'roidetect': False,
        'testFileList': repFn,
        'framesPerSession': framesPerSession,
    }
run_s2p(ops,db)

dataDir = f'{planeDir}/{testDn}/plane0/'
binfn = f'{dataDir}data.bin'
opsfn = f'{dataDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()
Ly = ops['Ly']
Lx = ops['Lx']
nframes = ops['nframes']
framesPerSession = ops['framesPerSession']
numSessions = int(nframes/framesPerSession)
# perFileMeanImg = np.zeros(shape = (Ly,Lx,numSessions))
viewer = napari.Viewer()
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
        frames = f.ix(indices=inds).astype(np.float32)
        # perFileMeanImg[:,:,i] = frames.mean(axis=0)
        viewer.add_image(frames.mean(axis=0))
        
        
#%%
ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 100
ops['batch_size'] = 5000
ops['maxregshift'] = 0.2



ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True

ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']




testDn = 'test7'
ops['snr_thresh'] = 1.2
ops['nonrigid'] = True
ops['block_size'] = [128, 128]
ops['maxregshiftNR'] = 5

db = {'h5py': wfn,
        'h5py_key': ['data'],
        'data_path': [],
        'save_path0': planeDir,
        'save_folder': testDn,
        'fast_disk': f'{planeDir}/{testDn}',
        'roidetect': False,
        'testFileList': repFn,
        'framesPerSession': framesPerSession,
    }
run_s2p(ops,db)

dataDir = f'{planeDir}/{testDn}/plane0/'
binfn = f'{dataDir}data.bin'
opsfn = f'{dataDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()
Ly = ops['Ly']
Lx = ops['Lx']
nframes = ops['nframes']
framesPerSession = ops['framesPerSession']
numSessions = int(nframes/framesPerSession)
# perFileMeanImg = np.zeros(shape = (Ly,Lx,numSessions))
viewer = napari.Viewer()
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
        frames = f.ix(indices=inds).astype(np.float32)
        # perFileMeanImg[:,:,i] = frames.mean(axis=0)
        viewer.add_image(frames.mean(axis=0))



        
#%% Make it automatic
'''
Too many combinations of parameters to change.
Make it automatic, and select the best parameter.
QC: Both phase correlation and intensity correlation at the center of FOV
(Crop edges that can be circshifted. It depends on mouse.)

210618_auto_reg_param_test
'''










