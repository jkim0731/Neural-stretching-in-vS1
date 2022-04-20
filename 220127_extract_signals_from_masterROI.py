# -*- coding: utf-8 -*-
"""
Extract signal from master ROIs reversed to each session.

2022/01/27 JK
"""

import numpy as np
import matplotlib.pyplot as plt
import napari
from suite2p.registration import rigid, nonrigid
from suite2p.gui.drawroi import masks_and_traces
import time
import os
import gc
gc.enable()

def twostep_register_reverse(img, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                     rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2):
    '''
    Reverse twostep_register
    All the inputs are the same as those used for the original twostep_register
    '''
    frames = img.copy().astype(np.float32)
    if len(frames.shape) == 2:
        frames = np.expand_dims(frames, axis=0)
    elif len(frames.shape) < 2:
        raise('Dimension of the frames should be at least 2')
    elif len(frames.shape) > 3:
        raise('Dimension of the frames should be at most 3')
    (Ly, Lx) = frames.shape[1:]
    
    # 1st nonrigid shift (reversing 2nd nonrigid)
    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size2)
    ymax1 = np.tile(-nonrigid_y2, (frames.shape[0],1))
    xmax1 = np.tile(-nonrigid_x2, (frames.shape[0],1))
    frames = nonrigid.transform_data(data=frames, nblocks=nblocks, 
        xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
    
    # 1st rigid shift (reversing 2nd rigid)
    frames = np.roll(frames, (rigid_y2, rigid_x2), axis=(1,2))
    
    # 2nd nonrigid shift (reversing 1st nonrigid)
    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size1)
    ymax1 = np.tile(-nonrigid_y1, (frames.shape[0],1))
    xmax1 = np.tile(-nonrigid_x1, (frames.shape[0],1))
    frames = nonrigid.transform_data(data=frames, nblocks=nblocks, 
        xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
    
    # 2nd rigid shift (reversing 1st rigid)
    frames = np.roll(frames, (rigid_y1, rigid_x1), axis=(1,2))
    
    return frames

def calculate_regCell_threshold(cellMap, numPix, thresholdResolution = 0.01):
    trPrecision = len(str(thresholdResolution).split('.')[1])
    thresholdRange = np.around(np.arange(0.3,1+thresholdResolution/10,thresholdResolution), trPrecision)
    threshold = thresholdRange[np.argmin([np.abs(numPix - np.sum(cellMap>=threshold)) for threshold in thresholdRange])]
    cutMap = (cellMap >= threshold).astype(bool)
    return cutMap, threshold

def imblend_for_napari(refImg, testImg):
    if (len(refImg.shape) != 2) or (len(testImg.shape) != 2):
        raise('Both images should have 2 dims.')
    if any(np.array(refImg.shape)-np.array(testImg.shape)):
        raise('Both images should have matching dims')
    refImg = img_norm(refImg.copy())
    testImg = img_norm(testImg.copy())
    refRGB = np.moveaxis(np.tile(refImg,(3,1,1)), 0, -1)
    testRGB = np.moveaxis(np.tile(testImg,(3,1,1)), 0, -1)
    blended = imblend(refImg, testImg)
    return np.array([refRGB, testRGB, blended])

def img_norm(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

def imblend(refImg, testImg):
    if (len(refImg.shape) != 2) or (len(testImg.shape) != 2):
        raise('Both images should have 2 dims.')
    if any(np.array(refImg.shape)-np.array(testImg.shape)):
        raise('Both images should have matching dims')
    Ly,Lx = refImg.shape
    blended = np.zeros((Ly,Lx,3))
    blended[:,:,0] = refImg
    blended[:,:,2] = testImg
    blended[:,:,1] = refImg
    return blended


h5Dir = 'D:/TPM/JK/h5/'
s2pDir = 'D:/TPM/JK/s2p/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]


#%%

mi = 0
mouse = mice[mi]
for pn in range(1,3):
    planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
    reg = np.load(f'{planeDir}s2p_nr_reg.npy', allow_pickle=True).item()
    block_size1 = reg['block_size1']
    block_size2 = reg['block_size2']
    
    masterROI = np.load(f'{planeDir}JK{mouse:03}plane{pn}masterROI.npy', allow_pickle=True).item()
    masterMap = masterROI['masterMap']
    
    for sn in masterROI['selectedSnums']:
        sessionDir = f'{planeDir}{sn:03}/plane0/'
        print(sessionDir)
        ops = np.load(f'{sessionDir}ops.npy', allow_pickle=True).item()
        ops['reg_file'] = f'{sessionDir}data.bin'
        stat = np.load(f'{sessionDir}stat.npy', allow_pickle=True)
        iscell = np.load(f'{sessionDir}iscell.npy')
        
        regSi = np.where([sntmp[4:] == f'{sn:03}' for sntmp in reg['sessionNames']])[0][0]
        
        rigid_y1= reg['rigid_offsets_1st'][0][0][regSi]
        rigid_x1= reg['rigid_offsets_1st'][0][1][regSi]
        
        nonrigid_y1 = reg['nonrigid_offsets_1st'][0][0][regSi,:]
        nonrigid_x1 = reg['nonrigid_offsets_1st'][0][1][regSi,:]
        
        rigid_y2= reg['rigid_offsets_2nd'][0][0][regSi]
        rigid_x2= reg['rigid_offsets_2nd'][0][1][regSi]
        
        nonrigid_y2 = reg['nonrigid_offsets_2nd'][0][0][regSi,:]
        nonrigid_x2 = reg['nonrigid_offsets_2nd'][0][1][regSi,:]
        
        revMap = twostep_register_reverse(masterMap, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                             rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2).squeeze()
        
        # fig, ax = plt.subplots()
        # ax.imshow(np.sum(revMap, axis=0))
        
        numROI = revMap.shape[0]
        statBlank = []
        stat0 = []
        for ci in range(numROI):
            numPix = np.sum(revMap[ci,:,:])
            revROI, _ = calculate_regCell_threshold(revMap[ci,:,:], numPix, thresholdResolution = 0.01)
            (ypix, xpix) = np.unravel_index(np.where(revROI.flatten()==1)[0], revROI.shape)
            lam = np.ones(ypix.shape)/len(ypix)
            med = (np.median(ypix), np.median(xpix))
            stat0.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size, 'med': med})
        masterF, masterFneu, _, _, masterSpks, masterOps, masterStat = masks_and_traces(ops, stat0, statBlank)
        masterIscell = np.ones((numROI,2), 'uint8')
        
        masterDir = f'{sessionDir}master/'
        
        if not os.path.isdir(masterDir):
            os.mkdir(masterDir)
        np.save(f'{masterDir}F.npy', masterF)
        np.save(f'{masterDir}Fneu.npy', masterFneu)
        np.save(f'{masterDir}spks.npy', masterSpks)
        np.save(f'{masterDir}ops.npy', masterOps)
        np.save(f'{masterDir}stat.npy', masterStat)
        np.save(f'{masterDir}iscell.npy', masterIscell)
    
    
#%%
allSpikes = []
for sn in masterROI['selectedSnums']:
    sessionSpk = []
    for pn in range(1,3):
        planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
        sessionDir = f'{planeDir}{sn:03}/plane0/master/'
        planeSpk = np.load(f'{sessionDir}spks.npy')
        if len(sessionSpk)==0:
            sessionSpk = planeSpk
        else:
            sessionSpk = np.vstack((sessionSpk, planeSpk))
    if len(allSpikes) == 0:
        allSpikes = sessionSpk
    else:
        allSpikes = np.hstack((allSpikes, sessionSpk))
#%%
sorti = np.flip(np.argsort(np.sum(allSpikes,axis=1)))

#%%
fig, ax = plt.subplots(2,1, figsize=(10,5))
ax[0].imshow(allSpikes[sorti[:50],:], aspect=100, vmax=300)
ax[0].set_xticks([])
ax[1].imshow(allSpikes[sorti[50:],:], aspect=100, vmax=50)
ax[1].set_xlabel('Frame')        
ax[1].set_ylabel('Cell')
fig.tight_layout()
fig.subplots_adjust(wspace=0.005, hspace=0.0001)    
fig.suptitle('JK025 planes 1-2')