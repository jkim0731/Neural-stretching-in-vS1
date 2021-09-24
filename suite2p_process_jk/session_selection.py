"""
Select sessions that have matched FOV with the reference session.
Using pixel correlation, and comparing with those of same-session same-plane and between-plane registration.
(Results from 'register_to_reference.py')
Also using visual inspection.

Up to here, based on 210806_session_selection.py
"""

#%% BS

import numpy as np
from matplotlib import pyplot as plt
from suite2p.registration import nonrigid
import os, glob
import napari
from suite2p.io.binary import BinaryFile
from skimage import exposure
import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]

def get_session_names(baseDir, mouse, planeNum):
    tempFnList = glob.glob(f'{baseDir}{mouse:03}_*_plane_{planeNum}.h5')
    tempFnList = [fn for fn in tempFnList if 'x' not in fn] # Treatment for h5 files with 'x' (testing blank edge removal)
    midNum = np.array([int(fn.split('\\')[1].split('_')[1]) for fn in tempFnList])
    trialNum = np.array([int(fn.split('\\')[1].split('_')[2][0]) for fn in tempFnList])
    spontSi = np.where( (midNum>5000) & (midNum<6000) )[0]
    piezoSi = np.where(midNum>9000)[0]
    
    if np.any(spontSi): 
        spontTrialNum = np.unique(trialNum[spontSi]) # used only for mouse > 50
    
    if np.any(piezoSi):
        piezoTrialNum = np.unique(trialNum[piezoSi])
    
    sessionNum = np.unique(midNum)
    regularSni = np.where(sessionNum < 1000)[0]
    
    sessionNames = []
    for sni in regularSni:
        sn = sessionNum[sni]
        sname = f'{mouse:03}_{sn:03}'
        sessionNames.append(sname)
    if mouse < 50:
        for si in spontSi:
            sessionNames.append(tempFnList[si].split('\\')[1].split('.h5')[0][:-8])
    else:
        for stn in spontTrialNum:
            sn = midNum[spontSi[0]]
            sname = f'{mouse:03}_{sn}_{stn}'
            sessionNames.append(sname)
    for ptn in piezoTrialNum:
        sn = midNum[piezoSi[0]]
        sname = f'{mouse:03}_{sn}_{ptn}'
        sessionNames.append(sname)
    return sessionNames

#%% Plot correlation values
# For each mouse, and each volume, import reg results
mi = 0
pnTop = 5 # either 1 or 5
volume = 'upper' if pnTop < 5 else 'lower'
mouse = mice[mi]
mouseDir = f'{h5Dir}{mouse:03}/'
val = np.load(f'{mouseDir}same_session_regCorrVals_JK{mouse:03}.npy', allow_pickle=True).item()
upperCorr = val['upperCorr'].copy()
lowerCorr = val['lowerCorr'].copy()

controlCorr = upperCorr.copy() if pnTop < 5 else lowerCorr.copy()
samePlaneCorr = np.array([np.diagonal(controlCorr[:,:,i]) for i in range(controlCorr.shape[2])]).flatten()
onePlaneCorr = np.array([np.diagonal(controlCorr[:,:,i],1) for i in range(controlCorr.shape[2])]).flatten()

sessionNames =get_session_names(f'{mouseDir}plane_{pnTop}/', mouse, pnTop)
numSession = len(sessionNames)

fig, ax = plt.subplots(figsize=(13,7))
# Plot mean +/- sd of same plane correlation value
ax.plot(range(numSession), np.repeat(samePlaneCorr.mean(), numSession), 'k-', label=None)
ax.fill_between(range(numSession),np.repeat(samePlaneCorr.mean()-samePlaneCorr.std(), numSession), 
                np.repeat(samePlaneCorr.mean()+samePlaneCorr.std(), numSession), alpha=0.2, facecolor='k')
# Plot mean +/- sd of one-diff plane correlation value
ax.plot(range(numSession), np.repeat(onePlaneCorr.mean(), numSession), 'k-', label=None)
ax.fill_between(range(numSession), np.repeat(onePlaneCorr.mean()-onePlaneCorr.std(), numSession), 
                np.repeat(onePlaneCorr.mean()+onePlaneCorr.std(), numSession), alpha=0.2, facecolor='k')

# Plot each plane correlation values with the ref image across sessions
for pn in range(pnTop,pnTop+4):
    planeDir = f'{mouseDir}plane_{pn}/'
    reg = np.load(f'{planeDir}s2p_nr_reg.npy', allow_pickle=True).item()
    ax.plot(reg['corrVals'], label=f'Plane {pn}')
ax.legend()
ax.set_title(f'JK{mouse:03} {volume} volume', fontsize=20)
ax.set_xlabel('Session index', fontsize=15)
ax.set_ylabel('Intensity correlation', fontsize=15)
fig.tight_layout()


#%% Optionally, using mean squared difference
# upperMsd = val['upperMsd'].copy()
# lowerMsd= val['lowerMsd'].copy()
# controlMsd = upperMsd.copy() if pnTop < 5 else lowerMsd.copy()
# samePlaneMsd = np.array([np.diagonal(controlMsd[:,:,i]) for i in range(controlMsd.shape[2])]).flatten()
# onePlaneMsd = np.array([np.diagonal(controlMsd[:,:,i],1) for i in range(controlMsd.shape[2])]).flatten()
# fig, ax = plt.subplots(figsize=(13,7))
# # Plot mean +/- sd of same plane correlation value
# ax.plot(range(numSession), np.repeat(samePlaneMsd.mean(), numSession), 'k-', label=None)
# ax.fill_between(range(numSession),np.repeat(samePlaneMsd.mean()-samePlaneMsd.std(), numSession), 
#                 np.repeat(samePlaneMsd.mean()+samePlaneMsd.std(), numSession), alpha=0.2, facecolor='k')
# # Plot mean +/- sd of one-diff plane correlation value
# ax.plot(range(numSession), np.repeat(onePlaneMsd.mean(), numSession), 'k-', label=None)
# ax.fill_between(range(numSession), np.repeat(onePlaneMsd.mean()-onePlaneMsd.std(), numSession), 
#                 np.repeat(onePlaneMsd.mean()+onePlaneMsd.std(), numSession), alpha=0.2, facecolor='k')

# # Plot each plane correlation values with the ref image across sessions
# for pn in range(pnTop,pnTop+4):
#     planeDir = f'{mouseDir}plane_{pn}/'
#     reg = np.load(f'{planeDir}s2p_nr_reg.npy', allow_pickle=True).item()
#     ax.plot(reg['msdVals'], label=f'Plane {pn}')
# ax.legend()
# ax.set_title(f'JK{mouse:03} {volume} volume', fontsize=20)
# ax.set_xlabel('Session index', fontsize=15)
# ax.set_ylabel('Mean squared difference', fontsize=15)
# fig.tight_layout()

#%% Visual inspection
# In each plane, each session, along with the ref session mimg and the overlap

# Find the reference session index
refSession = refSessions[mi]
refSn = f'{mouse:03}_{refSession:03}'
refSi = np.where([refSn==sn for sn in sessionNames])[0]
for pn in range(pnTop, pnTop+4):
    # New napari viewer for each plane
    viewer = napari.Viewer()
    planeDir = f'{mouseDir}plane_{pn}/'
    reg = np.load(f'{planeDir}s2p_nr_reg.npy', allow_pickle=True).item()
    refImg = reg['regImgs'][refSi,:,:]
    refImg = refImg/np.amax(refImg)
    for si in range(numSession):
        tempImg = reg['regImgs'][si,:,:]
        tempImg = tempImg/np.amax(tempImg)    
        # Make an RGB matrix for visual inspection
        mat = np.zeros((3,tempImg.shape[0],tempImg.shape[1],3))
        mat[0,:,:,:] = np.moveaxis(np.tile(refImg,(3,1,1)),0,-1)
        mat[1,:,:,:] = np.moveaxis(np.tile(tempImg,(3,1,1)),0,-1)
        mat[2,:,:,0] = refImg
        mat[2,:,:,2] = refImg
        mat[2,:,:,1] = tempImg
        sname = sessionNames[si]
        viewer.add_image(mat, rgb=True, name=f'{si}: {sname} p{pn}', visible=False)

