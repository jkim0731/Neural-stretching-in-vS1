# -*- coding: utf-8 -*-
"""
Test inverse suite2p nonrigid registration
Take minus of all parameters, and see if it works.

2022/01/27 JK
"""

import numpy as np
from suite2p.registration import rigid, nonrigid
import napari
import matplotlib.pyplot as plt

def twostep_register(img, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                     rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2):
    frames = img.copy().astype(np.float32)
    if len(frames.shape) == 2:
        frames = np.expand_dims(frames, axis=0)
    elif len(frames.shape) < 2:
        raise('Dimension of the frames should be at least 2')
    elif len(frames.shape) > 3:
        raise('Dimension of the frames should be at most 3')
    (Ly, Lx) = frames.shape[1:]
    # 1st rigid shift
    frames = np.roll(frames, (-rigid_y1, -rigid_x1), axis=(1,2))
    # 1st nonrigid shift
    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size1)
    ymax1 = np.tile(nonrigid_y1, (frames.shape[0],1))
    xmax1 = np.tile(nonrigid_x1, (frames.shape[0],1))
    frames = nonrigid.transform_data(data=frames, nblocks=nblocks, 
        xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
    # 2nd rigid shift
    frames = np.roll(frames, (-rigid_y2, -rigid_x2), axis=(1,2))
    # 2nd nonrigid shift            
    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size2)
    ymax1 = np.tile(nonrigid_y2, (frames.shape[0],1))
    xmax1 = np.tile(nonrigid_x2, (frames.shape[0],1))
    frames = nonrigid.transform_data(data=frames, nblocks=nblocks, 
        xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
    return frames

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
    
    # # 1st rigid shift
    # frames = np.roll(frames, (-rigid_y1, -rigid_x1), axis=(1,2))
    # # 1st nonrigid shift
    # yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size1)
    # ymax1 = np.tile(nonrigid_y1, (frames.shape[0],1))
    # xmax1 = np.tile(nonrigid_x1, (frames.shape[0],1))
    # frames = nonrigid.transform_data(data=frames, nblocks=nblocks, 
    #     xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
    # # 2nd rigid shift
    # frames = np.roll(frames, (-rigid_y2, -rigid_x2), axis=(1,2))
    # # 2nd nonrigid shift            
    # yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size2)
    # ymax1 = np.tile(nonrigid_y2, (frames.shape[0],1))
    # xmax1 = np.tile(nonrigid_x2, (frames.shape[0],1))
    # frames = nonrigid.transform_data(data=frames, nblocks=nblocks, 
    #     xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
    return frames

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

def calculate_regCell_threshold(cellMap, numPix, thresholdResolution = 0.01):
    trPrecision = len(str(thresholdResolution).split('.')[1])
    thresholdRange = np.around(np.arange(0.3,1+thresholdResolution/10,thresholdResolution), trPrecision)
    threshold = thresholdRange[np.argmin([np.abs(numPix - np.sum(cellMap>=threshold)) for threshold in thresholdRange])]
    cutMap = (cellMap >= threshold).astype(bool)
    return cutMap, threshold
#%% Load data
h5Dir = 'D:/TPM/JK/h5/'
mouse = 25
pn = 2
sn = 19

planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
reg = np.load(f'{planeDir}s2p_nr_reg.npy', allow_pickle=True).item()
sessionDir = f'{planeDir}{sn:03}/plane0/'
ops = np.load(f'{sessionDir}ops.npy', allow_pickle=True).item()
stat = np.load(f'{sessionDir}stat.npy', allow_pickle=True)
iscell = np.load(f'{sessionDir}iscell.npy')

block_size1 = reg['block_size1']
block_size2 = reg['block_size2']

si = np.where([sntmp[4:] == f'{sn:03}' for sntmp in reg['sessionNames']])[0][0]

rigid_y1= reg['rigid_offsets_1st'][0][0][si]
rigid_x1= reg['rigid_offsets_1st'][0][1][si]

nonrigid_y1 = reg['nonrigid_offsets_1st'][0][0][si,:]
nonrigid_x1 = reg['nonrigid_offsets_1st'][0][1][si,:]

rigid_y2= reg['rigid_offsets_2nd'][0][0][si]
rigid_x2= reg['rigid_offsets_2nd'][0][1][si]

nonrigid_y2 = reg['nonrigid_offsets_2nd'][0][0][si,:]
nonrigid_x2 = reg['nonrigid_offsets_2nd'][0][1][si,:]

#%% Test with mean image
mimg = ops['meanImg']

mimgNR = twostep_register(mimg, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                     rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2)

mimgNRrev = twostep_register_reverse(mimgNR, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                     rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2).squeeze()

blended = imblend_for_napari(mimg, mimgNRrev)

npr = napari.Viewer()
npr.add_image(mimg, name='mimg')
npr.add_image(mimgNR, name='mimgNR')
npr.add_image(mimgNRrev, name='mimgNRrev')
npr.add_image(blended, name='blended')


#%% Now test how much it affect for ROI reverse-mapping

masterROI = np.load(f'{planeDir}JK{mouse:03}plane{pn}masterROI.npy', allow_pickle=True).item()
masterMap = masterROI['masterMap']
si = np.where(masterROI['selectedSnums']==sn)[0][0]
roiSessionInd = np.where(masterROI['roiSessionInd'].astype(int)==si)
masterMapSession = masterMap[roiSessionInd,:,:].squeeze()

revMap = twostep_register_reverse(masterMapSession, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                     rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2).squeeze()

revCutMap = np.zeros((revMap.shape), 'bool')
numROI = revMap.shape[0]
for ci in range(numROI):
    numPix = np.sum(masterMapSession[ci,:,:])
    revCutMap[ci,:,:], _ = calculate_regCell_threshold(revMap[ci,:,:], numPix, thresholdResolution = 0.01)

cellInds = np.where(iscell[:,0]==1)[0]
numCell = len(cellInds)
cellMap = np.zeros((numCell,ops['Ly'], ops['Lx']), 'bool')
for ci in range(numCell):
    cellMap[ci,stat[cellInds[ci]]['ypix'],stat[cellInds[ci]]['xpix']] = 1
    
##%%
blended = imblend_for_napari(np.amax(revCutMap,axis=0).astype(int), np.amax(cellMap, axis=0).astype(int))
napari.view_image(blended)

'''
Overall it works great
'''
#%% Calculate misplaced pixels (quantification)
pn = 2
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
reg = np.load(f'{planeDir}s2p_nr_reg.npy', allow_pickle=True).item()
masterROI = np.load(f'{planeDir}JK{mouse:03}plane{pn}masterROI.npy', allow_pickle=True).item()
masterMap = masterROI['masterMap']
selectedSnums = masterROI['selectedSnums']
numSession = len(selectedSnums)

block_size1 = reg['block_size1']
block_size2 = reg['block_size2']
    
overlapPropAll = np.zeros(0)
for si, sn in enumerate(selectedSnums):
    print(f'Processing {sn:03} {si}/{numSession-1}')
    sessionDir = f'{planeDir}{sn:03}/plane0/'
    ops = np.load(f'{sessionDir}ops.npy', allow_pickle=True).item()
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

    roiSessionInd = np.where(masterROI['roiSessionInd'].astype(int)==si)
    masterMapSession = masterMap[roiSessionInd,:,:].squeeze()
    
    revMap = twostep_register_reverse(masterMapSession, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                         rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2).squeeze()
    
    revCutMap = np.zeros((revMap.shape), 'bool')
    numROI = revMap.shape[0]
    for ci in range(numROI):
        numPix = np.sum(masterMapSession[ci,:,:])
        revCutMap[ci,:,:], _ = calculate_regCell_threshold(revMap[ci,:,:], numPix, thresholdResolution = 0.01)
    
    numPixRev = np.sum(revCutMap, axis=(1,2))
    
    cellInds = np.where(iscell[:,0]==1)[0]
    numCell = len(cellInds)
    cellMap = np.zeros((numCell,ops['Ly'], ops['Lx']), 'bool')
    for ci in range(numCell):
        cellMap[ci,stat[cellInds[ci]]['ypix'],stat[cellInds[ci]]['xpix']] = 1
        
    numPixCell = np.sum(cellMap, axis=(1,2))
    
    overlaps = np.zeros((numROI, numCell), 'uint16')
    for ci in range(numCell):
        overlaps[:,ci] = np.sum(revCutMap*cellMap[ci,:,:], axis=(1,2))
    
    overlapProp = np.zeros(numROI)
    for ri in range(numROI):
        ci = np.argmax(overlaps[ri,:].squeeze())
        overlapProp[ri] = np.amax(overlaps[ri,:]) / (numPixRev[ri] + numPixCell[ci] - np.amax(overlaps[ri,:]))
    
    overlapPropAll = np.concatenate((overlapPropAll, overlapProp))

opMean = np.mean(overlapPropAll)
opStd = np.std(overlapPropAll)
#%%
fix, ax = plt.subplots()
ax.hist(overlapPropAll, bins = 100, range = (0, 1), density=True, cumulative=True)
ax.set_xlabel('Overlap proportion (intersect/union)')
ax.set_ylabel('Cumulative proportion')
ax.set_title(f'JK{mouse:03} plane {pn}\n Mean $\pm$ SD = {opMean:.2f} $\pm$ {opStd:.2f}')
        
#%%

        
        
        
