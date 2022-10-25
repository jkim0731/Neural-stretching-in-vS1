# -*- coding: utf-8 -*-
"""
Session selection based on depth tolerance.
1. Set desired depth tolerance (10, 15, 20 um)

2. Go to each representative plane (either 1 or 5), gather z-drift across all the sessions.
3. Manually select relative z positions to include.
4. Show registered images (use bilinear rolling 3 sessions) 
and remove visually different session.

Repeat 2-4 across mice and volumes.

5. Save session numbers into pandas dataframe.
Results in 'selected_training_sessions_xx_micron.cvs'

Copied from the beginning part of 220117_roi_collection.py

2022/04/27 JK
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
import napari
import os, glob, shutil
from pystackreg import StackReg
from skimage import exposure
from suite2p.registration import rigid, nonrigid

import gc
gc.enable()

# h5Dir = 'D:/TPM/JK/h5/'
h5Dir = 'D:/'

mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
expSessions =   [19,    10,   21,   17,     0,      0,      23,     0,      21,     0,      0,      0]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]

def clahe_each(img: np.float64, kernel_size = None, clip_limit = 0.01, nbins = 2**16):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    newimg = exposure.equalize_adapthist(newimg, kernel_size = kernel_size, clip_limit = clip_limit, nbins=nbins)    
    return newimg
#%% 1. Set desired depth tolerance (10, 15, 20 um)
depth_tolerance = 10 # in microns

#%% 2. Go to each representative plane (either 1 or 5), gather z-drift across all the sessions.
mi = 0
mouse = mice[mi]
vi = 1 # volume index, either 1 or 5
expTestSnum = expSessions[mi] # Expert test session number
if mi == 1 & vi == 1: # An exception for JK027 upper layer
    expTestSnum = 16

# Load z-drift data
zdrift = np.load(f"{h5Dir}JK{mouse:03}_zdrift_plane{vi}.npy", allow_pickle=True).item()

# Select training sessions only
# Re-order sessions if necessary
siArr = np.where([len(sn.split('_'))==2 for sn in zdrift['info']['sessionNames']])[0]
snums = np.array([int(sn.split('_')[1]) for sn in zdrift['info']['sessionNames'] if len(sn.split('_'))==2])
siSorted = siArr[np.argsort(snums)]

#% Manually select depths from z-drift data across sessions and choose corresponding sessions
allSnums = np.array([int(sn.split('_')[1]) for sn in zdrift['info']['sessionNames']])
fig, ax = plt.subplots()
for xi, si in enumerate(siSorted):
    xrange = np.linspace(xi,xi+1, len(zdrift['zdriftList'][si]))
    if allSnums[si] in [refSessions[mi], expTestSnum]:
        ax.plot(xrange, zdrift['zdriftList'][si], 'r-')
    else:
        ax.plot(xrange, zdrift['zdriftList'][si], 'k-')
ax.set_xlabel('Training session index')
ax.set_ylabel(r'Relative imaging plane (2 $\mu$m)')
ax.set_title(f'JK{mouse:03} plane {vi}\nZ-drift, time normalized per session')
#%% Manual selection of depths - relative values
# selDepthsRV = [7,17]  # JK025 upper
# selDepthsRV = [18,28]
# selDepthsRV = [20,30] # JK027 upper
# selDepthsRV = [25,35]
# selDepthsRV = [17,27] # JK030 upper
# selDepthsRV = [22,32]
# selDepthsRV = [16,26] # JK036 upper
# selDepthsRV = [12,22]
# selDepthsRV = [22,32] # JK039 upper
selDepthsRV = [17,27] # JK039 lower
# selDepthsRV = [27,37] # JK052 upper
# selDepthsRV = [17,27] # JK052 lower

ax.plot([0,len(siSorted)], [selDepthsRV[0]-0.3, selDepthsRV[0]-0.3], '--', color=[0.6, 0.6, 0.6])
ax.plot([0,len(siSorted)], [selDepthsRV[1]+0.3, selDepthsRV[1]+0.3], '--', color=[0.6, 0.6, 0.6])

#%% Selected sessions (>=30 min of selected depths)
selectedSi = np.array([si for si in siSorted if \
              sum(np.logical_and(zdrift['zdriftList'][si]>=selDepthsRV[0], zdrift['zdriftList'][si]<=selDepthsRV[1])) >=3 ])
selectedSnums = [int(sname.split('_')[1]) for sname in np.array(zdrift['info']['sessionNames'])[selectedSi]]

#%% Plot highlighting included trials (time)
# And show registered images
fig, ax = plt.subplots()
for xi, si in enumerate(siSorted):
    xrange = np.linspace(xi,xi+1, len(zdrift['zdriftList'][si]))
    for xri in range(len(xrange)-1):
        tempDepths = zdrift['zdriftList'][si][xri:xri+2]
        tempRange = xrange[xri:xri+2]
        if np.logical_and(tempDepths >= selDepthsRV[0], tempDepths <= selDepthsRV[1]).all():
            ax.plot(tempRange, tempDepths, 'b-')
        else:
            ax.plot(tempRange, tempDepths, '-', color=[0.8, 0.8, 0.8])
    
ax.plot([0,len(siSorted)], [selDepthsRV[0]-0.3, selDepthsRV[0]-0.3], '--', color=[0.6, 0.6, 0.6])
ax.plot([0,len(siSorted)], [selDepthsRV[1]+0.3, selDepthsRV[1]+0.3], '--', color=[0.6, 0.6, 0.6])
        
ax.set_xlabel('Training session index')
ax.set_ylabel(r'Relative imaging plane (2 $\mu$m)')
ax.set_title(f'JK{mouse:03} plane {vi}\nZ-drift, time normalized per session')

# Show registered images 
prevN = 3
leftBuffer = 30
rightBuffer = 30 if mouse < 50 else 100
bottomBuffer = 10
topBuffer = 50

viewer = napari.Viewer()
for pn in range(vi,vi+4):
    planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
    regFn = f'{planeDir}s2p_nr_reg.npy'
    reg = np.load(regFn, allow_pickle=True).item()
    regImgs = reg['regImgs'][selectedSi,topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
    
    numSelected = len(selectedSi)
    mimgs = np.zeros_like(regImgs)
    mimgClahe = np.zeros_like(regImgs)
    for si, sn in enumerate(selectedSnums):
        opsFn = f'{planeDir}{sn:03}/plane0/ops.npy'
        ops = np.load(opsFn, allow_pickle=True).item()
        mimgs[si,:,:] = ops['meanImg'][topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
        mimgClahe[si,:,:] = clahe_each(mimgs[si,:,:])
    
    srBi = StackReg(StackReg.BILINEAR)
    regBi = np.zeros_like(mimgs)
    tformsBi = []
    
    for si in range(numSelected):
        if si == 0:
            regBi[si,:,:] = mimgs[si,:,:]
        else:
            previStart = max(0,si-prevN)
            
            refBi = clahe_each(np.mean(regBi[previStart:si,:,:], axis=0))
            tform = srBi.register(refBi, mimgClahe[si,:,:])
            regBi[si,:,:] = srBi.transform(mimgs[si,:,:], tmat=tform)
            tformsBi.append(tform)
    
    viewer.add_image(regBi, name=f'plane {pn}')

#%% Get session info from previous registration tests
# 

#%% Manually remove sessions (that clearly does not seem to have matched FOV)
