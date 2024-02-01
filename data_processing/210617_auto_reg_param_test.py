# -*- coding: utf-8 -*-
"""
Too many combinations of parameters to change.
Make it automatic, and select the best parameter.
QC: Both phase correlation and intensity correlation at the center of FOV
(Crop edges that can be circshifted. It depends on mouse.)
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
#%%
mi = 1
framesPerSession = 20
# removeSessionNames = ['025_016_', '025_018_', '025_5555_001', '025_5555_103', '025_9999_1']
# removeSessionNames = ['025_025_', '025_5554_103', '025_9999_2']
removeSessionNames = ['027_007_', '027_025_', '027_5555_001', '027_5555_104', '027_9999_1']
mouse = mice[mi]

for planei in range(1,5):
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
        
    #%%
    
    wfn = f'{planeDir}selected.h5'
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
    ops['fs'] = freq[mi]
    ops['zoom'] = zoom[mi]
    ops['umPerPix'] = 1.4/ops['zoom']
    
    maxregshiftList = [0.1, 0.2, 0.3]
    maxregshiftNRList = [5, 10, 20]
    block_sizeList = [[128,128], [96, 96], [64, 64]]
    snr_threshList = [1, 1.1, 1.2, 1.3]
    
    paramSetInd = 0
    
    ops['nonrigid'] = False
    for mrs in maxregshiftList:
        testDn = f'test{paramSetInd:02}'
        
        ops['maxregshift'] = mrs
        
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
        paramSetInd += 1
    
    ops['nonrigid'] = True
    for mrs in maxregshiftList:
        for mrsn in maxregshiftNRList:
            for bs in block_sizeList:
                for st in snr_threshList:
                    testDn = f'test{paramSetInd:02}'
                    
                    ops['maxregshift'] = mrs
                    ops['maxregshiftNR'] = mrsn
                    ops['block_size'] = bs
                    ops['snr_thresh'] = st
                    
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
                    paramSetInd += 1
        

# #%% Visual inspection to decide FOV proportion selection

# dataDir = f'{planeDir}/{testDn}/plane0/'
# binfn = f'{dataDir}data.bin'
# opsfn = f'{dataDir}ops.npy'
# ops = np.load(opsfn, allow_pickle = True).item()
# Ly = ops['Ly']
# Lx = ops['Lx']
# nframes = ops['nframes']
# framesPerSession = ops['framesPerSession']
# numSessions = int(nframes/framesPerSession)
# # perFileMeanImg = np.zeros(shape = (Ly,Lx,numSessions))
# viewer = napari.Viewer()
# with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
#     for i in range(numSessions):
#         inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
#         frames = f.ix(indices=inds).astype(np.float32)
#         # perFileMeanImg[:,:,i] = frames.mean(axis=0)
#         viewer.add_image(frames.mean(axis=0))
        
        
# #%% Set top bottom left right margin in "pixel number"
# topMargin = 100
# bottomMargin = 20
# leftMargin = 20
# rightMargin = 50

# phaseCorr = np.zeros(shape = (paramSetInd,len(numSessions)))
# intensityCorr = np.zeros(shape = (paramSetInd,len(numSessions)))

# def phase_corr(a,b):
#     R = np.fft.fft2(a) * np.fft.fft2(b).conj()
#     R /=np.absolute(R)
