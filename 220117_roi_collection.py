# -*- coding: utf-8 -*-
"""
220117 ROI collection
Collect ROIs from multiple sessions that have matching depth
- Depth tolerance of 20 um for now
- Using training sessions only for now (ignore spontaneous and passive deflection sessions)
- Use mean image registration results (check the quality first)
- Overlap threshold = 0.5
- When there are matching ROIs from the new cell map, leave the one with lower perimeter/area ratio
- When there are multiple matching ROIs,
    - Remove weird ROI at that session (using suite2p GUI)
    - Or if there are matching pairs, leave the pair with lower mean perimeter/area ratio

Curate trials from z-drift results
- Should load the results from z-drfit resutls files
- Should also load .trials files
-- Compare baseline fluorescence change and inferred spikes change against depth change

"""

#%% BS
import numpy as np
import matplotlib.pyplot as plt
import napari
import os, glob, shutil

from suite2p.registration import rigid, nonrigid

import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'
s2pDir = 'D:/TPM/JK/s2p/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]

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

def calculate_regCell_threshold(cellMap, numPix, thresholdResolution = 0.01):
    trPrecision = len(str(thresholdResolution).split('.')[1])
    thresholdRange = np.around(np.arange(0.3,1+thresholdResolution/10,thresholdResolution), trPrecision)
    threshold = thresholdRange[np.argmin([np.abs(numPix - np.sum(cellMap>=threshold)) for threshold in thresholdRange])]
    cutMap = (cellMap >= threshold).astype(bool)
    return cutMap, threshold

def perimeter_area_ratio(img: bool):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    if len(img.shape) != 3:
        raise('Input image dimension should be either 2 or 3.')
    img = img.astype('bool')
    numCell = img.shape[0]
    par = np.zeros(numCell)
    for ci in range(numCell):
        tempImg = img[ci,:,:]
        inside = tempImg*np.roll(tempImg,1,axis=0)*np.roll(tempImg,-1,axis=0)*np.roll(tempImg,1,axis=1)*np.roll(tempImg,-1,axis=1)
        perimeter = np.logical_xor(tempImg, inside)
        par[ci] = np.sum(perimeter)/np.sum(tempImg) # tempImg instead of inside just to prevent dividing by 0 for some scattered rois
    return par

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

def check_multi_match_pair(multiMatchMasterInd, multiMatchNewInd, masterPar, newPar, 
                           overlapMatrix, overlaps, delFromMasterInd, delFromNewInd):
    tempDfMasterInd = np.zeros(0,'int')
    tempDfNewInd = np.zeros(0,'int')
    remFromMMmaster = np.zeros(0,'int') # Collect multiMatchMasterInd that is already processed (in the for loop)
    remFromMMnew = np.zeros(0,'int') # Collect multiMatchNewInd that is already processed (in the for loop)
    # Remove remFromMM* at the end.
    
    # First, deal with delFromMasterInd
    if len(multiMatchMasterInd)>0:
        for mci in range(len(multiMatchMasterInd)):
            masterCi = multiMatchMasterInd[mci]
            if masterCi in remFromMMmaster:
                continue
            else:
                newCis = np.where(overlapMatrix[masterCi,:])[0]
                    
                masterBestMatchi = np.zeros(len(newCis), 'int')
                for i, nci in enumerate(newCis):
                    masterBestMatchi[i] = np.argmax(overlaps[:,nci]).astype(int)
                # Check if there are multiple same matched IDs 
                # In this case, just remove the masterInd
                if any([len(np.where(masterBestMatchi==mi)[0])>1 for mi in masterBestMatchi]):
                    tempDfMasterInd = np.hstack((tempDfMasterInd, [masterCi]))
                    remFromMMmaster = np.hstack((remFromMMmaster, [masterCi]))
                
                # Else, check if there is a matching pair (or multiples)
                else:
                    newBestMatchi = np.zeros(len(masterBestMatchi), 'int')
                    for i, mci in enumerate(masterBestMatchi):
                        newBestMatchi[i] = np.argmax(overlaps[mci,:]).astype(int)

                    if all(newCis == newBestMatchi): # found a matching pair
                        # Calculate mean perimeter/area ratio
                        masterMeanpar = np.mean(masterPar[masterBestMatchi])
                        newMeanpar = np.mean(newPar[newBestMatchi])
                        
                        # Remove the pair with lower mean par
                        if masterMeanpar >= newMeanpar:
                            tempDfNewInd = np.hstack((tempDfNewInd, newBestMatchi))
                        else:
                            tempDfMasterInd = np.hstack((tempDfMasterInd, masterBestMatchi))
    
                        # Collect indices already processed
                        remFromMMmaster = np.hstack((remFromMMmaster, masterBestMatchi))
                        remFromMMnew = np.hstack((remFromMMnew, newBestMatchi))
                              
    # Then, deal with delFromNewInd
    if len(multiMatchNewInd)>0:
        for nci in range(len(multiMatchNewInd)):
            newCi = multiMatchNewInd[nci]
            if newCi in remFromMMnew:
                continue
            else:
                masterCis = np.where(overlapMatrix[:,newCi])[0]
                    
                newBestMatchi = np.zeros(len(masterCis), 'int')
                for i, mci in enumerate(masterCis):
                    newBestMatchi[i] = np.argmax(overlaps[mci,:]).astype(int)
                    
                # Check if there are multiple same matched IDs 
                # In this case, just remove the masterInd
                if any([len(np.where(newBestMatchi==ni)[0])>1 for ni in newBestMatchi]):
                    tempDfNewInd = np.hstack((tempDfNewInd, [newCi]))
                    remFromMMnew = np.hstack((remFromMMnew, [newCi]))
                
                # Else, check if there is a matching pair (or multiples)
                else:
                    masterBestMatchi = np.zeros(len(newBestMatchi), 'int')
                    for i, nci in enumerate(newBestMatchi):
                        masterBestMatchi[i] = np.argmax(overlaps[:,nci]).astype(int)

                    if all(masterCis == masterBestMatchi): # found a matching pair
                        # Calculate mean perimeter/area ratio
                        masterMeanpar = np.mean(masterPar[masterBestMatchi])
                        newMeanpar = np.mean(newPar[newBestMatchi])
                        
                        # Remove the pair with lower mean par
                        if masterMeanpar >= newMeanpar:
                            tempDfNewInd = np.hstack((tempDfNewInd, newBestMatchi))
                        else:
                            tempDfMasterInd = np.hstack((tempDfMasterInd, masterBestMatchi))
    
                        # Collect indices already processed
                        remFromMMmaster = np.hstack((remFromMMmaster, masterBestMatchi))
                        remFromMMnew = np.hstack((remFromMMnew, newBestMatchi))
    
    # Remove collected indices
    if len(tempDfMasterInd)>0:
        delFromMasterInd.extend(tempDfMasterInd)        
    if len(tempDfNewInd)>0:
        delFromNewInd.extend(tempDfNewInd)
    multiMatchMasterInd = np.array([mi for mi in multiMatchMasterInd if mi not in remFromMMmaster])
    multiMatchNewInd = np.array([ni for ni in multiMatchNewInd if ni not in remFromMMnew])
    
    return delFromMasterInd, delFromNewInd, multiMatchMasterInd, multiMatchNewInd


#%% ROI collection
'''
Collect ROIs from multiple sessions with matching depths
(1) Select depths
- Manually by looking at the z-drift graph, trying to include both 7-angle test sessions before and after learning.
- If this excludes too many sessions, have a second set of depths that can better collect training progression
- - There can be multiple dataset curation from a single imaging volume.
(2) Collect ROIs
- Start from the first session within the selected depth
- - Add ROIs to the collection
- Advance to the next session, add ROIs that are not matched with existing collection
- - Use mean image registration between sessions (check the quality first)
- When all ROIs are collected, go back to the beginning session and apply the collection to each session, 
extract signals, and apply spike inference.
'''

#%% ROI collection - one example (JK025 upper volume)
roiOverlapThresh = 0.5
# When two ROIs have an overlap, if this overlap/area is larger than the threshold
# for EITHER of the ROIs, then these two ROIs are defined as matching
# This is the same approach as in suite2p, and stricter than that of CaImAn (where intersection/union is used instead)

mi = 1
mouse = mice[mi]
vi = 1 # volume index, either 1 or 5
expTestSnum = 16 # Expert test session number

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

#%% Set depths (relative value)
# selDepthsRV = [7,17]  # JK025 upper
# selDepthsRV = [18,28]
selDepthsRV = [20,30] # JK027 upper
# selDepthsRV = [25,35]
# selDepthsRV = [17,27] # JK030 upper
# selDepthsRV = [22,32]
# selDepthsRV = [16,26] # JK036 upper
# selDepthsRV = [12,22]
# selDepthsRV = [22,32] # JK039 upper

# selDepthsRV = [27,37] # JK052 upper
# selDepthsRV = [17,27] # JK052 lower

ax.plot([0,len(siSorted)], [selDepthsRV[0]-0.3, selDepthsRV[0]-0.3], '--', color=[0.6, 0.6, 0.6])
ax.plot([0,len(siSorted)], [selDepthsRV[1]+0.3, selDepthsRV[1]+0.3], '--', color=[0.6, 0.6, 0.6])
# #%%
# fig, ax = plt.subplots()
# for xi, si in enumerate(siSorted):
#     xrange = np.linspace(xi,xi+0.1*len(zdrift['zdriftList'][si]), len(zdrift['zdriftList'][si]))
#     if allSnums[si] in [refSessions[mi], expTestSnum]:
#         ax.plot(xrange, zdrift['zdriftList'][si], 'r-')
#     else:
#         ax.plot(xrange, zdrift['zdriftList'][si], 'k-')
# ax.set_xlabel('Training session index')
# ax.set_ylabel(r'Relative imaging plane (2 $\mu$m)')
# ax.set_title(f'JK{mouse:03} plane {vi}\nZ-drift, accounting for time difference across sessions')
# #%%
# # Set depths (relative value)
# selDepthsRV = [7,17]
# ax.plot([0,len(siSorted)], [selDepthsRV[0]-0.3, selDepthsRV[0]-0.3], '--', color=[0.6, 0.6, 0.6])
# ax.plot([0,len(siSorted)], [selDepthsRV[1]+0.3, selDepthsRV[1]+0.3], '--', color=[0.6, 0.6, 0.6])

#%% Selected sessions (>=30 min of selected depths)
selectedSi = np.array([si for si in siSorted if \
              sum(np.logical_and(zdrift['zdriftList'][si]>=selDepthsRV[0], zdrift['zdriftList'][si]<=selDepthsRV[1])) >=3 ])
selectedSnums = [int(sname.split('_')[1]) for sname in np.array(zdrift['info']['sessionNames'])[selectedSi]]

##%% Plot highlighting included trials (time)
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

#%% Load and check registration results
# Test with all imaged planes from an imaging volume
viewer = napari.Viewer()
for pn in range(vi,vi+4):
    planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
    regFn = f'{planeDir}s2p_nr_reg.npy'
    reg = np.load(regFn, allow_pickle=True).item()
    
    regImgs = np.array(reg['regImgs'])[selectedSi]
    viewer.add_image(regImgs, name=f'plane {pn}')
    
#%% Adjust selected sessions manually
# manRmvSi = np.array([7,14]) # the index is from selectedSi, not from all the sessions
# manRmvSi = np.array([])
# manRmvSi = np.array([10,11,12])
# manRmvSi = np.array([])
# manRmvSi = np.array([])
# manRmvSi = np.array([19])
# manRmvSi = np.array([17])
# manRmvSi = np.array([19])
if len(manRmvSi)>0:
    selectedSi = np.delete(selectedSi, manRmvSi)
    selectedSnums = np.delete(selectedSnums, manRmvSi)
    
#%% Check the registration again
# Now with two_step registration check (2022/02/28)
fig, ax = plt.subplots()
numSession = len(selectedSnums)

for pn in range(vi,vi+4):
    viewer = napari.Viewer()
    planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
    regFn = f'{planeDir}s2p_nr_reg.npy'
    reg = np.load(regFn, allow_pickle=True).item()
    block_size1 = reg['block_size1']
    block_size2 = reg['block_size2']
    
    regImgs = np.array(reg['regImgs'])[selectedSi]
    viewer.add_image(regImgs, name=f'plane {pn}')
    
    transMimgs = []
    # transBlended = []
    for ssi, sn in enumerate(selectedSnums):
        tempOps = np.load(f'{planeDir}{sn:03}/plane0/ops.npy', allow_pickle=True).item()
        mimg = tempOps['meanImg']
        
        rigid_y1 = reg['rigid_offsets_1st'][0][0][selectedSi[ssi]]
        rigid_x1 = reg['rigid_offsets_1st'][0][1][selectedSi[ssi]]
        nonrigid_y1 = reg['nonrigid_offsets_1st'][0][0][selectedSi[ssi],:]
        nonrigid_x1 = reg['nonrigid_offsets_1st'][0][1][selectedSi[ssi],:]
        
        rigid_y2 = reg['rigid_offsets_2nd'][0][0][selectedSi[ssi]]
        rigid_x2 = reg['rigid_offsets_2nd'][0][1][selectedSi[ssi]]
        nonrigid_y2 = reg['nonrigid_offsets_2nd'][0][0][selectedSi[ssi],:]
        nonrigid_x2 = reg['nonrigid_offsets_2nd'][0][1][selectedSi[ssi],:]
        
        tempTransformed = twostep_register(mimg, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                             rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2)
        transMimgs.append(tempTransformed)
        # tempBlend = imblend_for_napari(regImgs[ssi,:,:], np.squeeze(tempTransformed))
        # transBlended.append(np.squeeze(tempBlend[-1,:,:]))
    viewer.add_image(np.squeeze(np.array(transMimgs)), name='Transformed')
    # viewer.add_image(np.array(transBlended), name='Blended', rgb=True)
    picorr = np.zeros(numSession)
    for si in range(numSession):
        picorr[si] = np.corrcoef(regImgs[si,:,:].flatten(), transMimgs[si].flatten())[0,1]
    ax.plot(picorr, label=f'plane {pn}')
ax.legend()





    
    



#%%###########################################################################
# Run through all the sessions again and collect all ROIs
# and save the master map.
# If there is multiple match, then stop and raise an error.
pn = 4

planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
regFn = f'{planeDir}s2p_nr_reg.npy'
reg = np.load(regFn, allow_pickle=True).item()
regImgs = np.array(reg['regImgs'])[selectedSi]

Ly, Lx = regImgs.shape[1:]
regMasterInds = [rsn in [f'{mouse:03}_{snum:03}' for snum in selectedSnums] 
                for rsn in reg['sessionNames']]
ybuffer = np.amax(np.abs(reg['rigid_offsets_1st'][0][0][regMasterInds]))
xbuffer = np.amax(np.abs(reg['rigid_offsets_1st'][0][1][regMasterInds]))

bufferEdge = np.ones(regImgs.shape[1:], 'uint8')
bufferEdge[ybuffer:Ly-ybuffer,xbuffer:Lx-xbuffer] = 0

# Set a master ROI map
masterMap = np.zeros((0,*regImgs.shape[1:]), 'bool')
masterCellThresh = np.zeros(0)
masterPAR = np.zeros(0) # perimeter-area ratio
roiSessionInd = np.zeros(0) # Recording which session (index) the ROIs came from
# Go through sessions and collect ROIs into the master ROI map
# Pre-sessions (901 and 902) should be at the beginning
# for snum in selectedSnums:

block_size1 = reg['block_size1']
block_size2 = reg['block_size2']

fig, ax = plt.subplots(2,3, figsize=(14,7))
###########
for si in range(len(selectedSi)):
    snum = selectedSnums[si]
    sname = f'{mouse:03}_{snum:03}'
    print(f'Processing {sname} {si}/{len(selectedSi)-1}')
    tempRegi = np.where(np.array(reg['sessionNames'])==sname)[0][0]
    
    rigid_y1 = reg['rigid_offsets_1st'][0][0][tempRegi]
    rigid_x1 = reg['rigid_offsets_1st'][0][1][tempRegi]
    nonrigid_y1 = reg['nonrigid_offsets_1st'][0][0][tempRegi,:]
    nonrigid_x1 = reg['nonrigid_offsets_1st'][0][1][tempRegi,:]
    
    rigid_y2 = reg['rigid_offsets_2nd'][0][0][tempRegi]
    rigid_x2 = reg['rigid_offsets_2nd'][0][1][tempRegi]
    nonrigid_y2 = reg['nonrigid_offsets_2nd'][0][0][tempRegi,:]
    nonrigid_x2 = reg['nonrigid_offsets_2nd'][0][1][tempRegi,:]
    
    # Gather cell map
    tempStat = np.load(f'{planeDir}{snum:03}/plane0/stat.npy', allow_pickle=True)
    tempIscell = np.load(f'{planeDir}{snum:03}/plane0/iscell.npy', allow_pickle=True)
    tempCelli = np.where(tempIscell[:,0])[0]
    numCell = len(tempCelli)
    tempMap = np.zeros((numCell,*regImgs.shape[1:]), 'bool')
    for n, ci in enumerate(tempCelli):
        for pixi in range(len(tempStat[ci]['ypix'])):
            xi = tempStat[ci]['xpix']
            yi = tempStat[ci]['ypix']
            tempMap[n,yi,xi] = 1
    # Transform
    tempRegMap = twostep_register(tempMap, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                         rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2)
    
    # Select threshold per cell, to have (roughly) matching # of pixels
    # Save this threshold value per cell per session
    
    cutMap = np.zeros((numCell, *regImgs.shape[1:]), 'bool')
    delFromCut = []
    warpCellThresh = np.zeros(numCell)
    for ci in range(numCell):
        numPix = np.sum(tempMap[ci,:,:])
        cutMap[ci,:,:], warpCellThresh[ci] = calculate_regCell_threshold(tempRegMap[ci,:,:], numPix, thresholdResolution = 0.01)
        # Remove ROIs that have pixels within the edge buffers (:ybuffer, Ly-ybuffer:, :xbuffer, Lx-xbuffer:)
        if (cutMap[ci,:,:] * bufferEdge).flatten().any():
            delFromCut.append(ci)
    cutMap = np.delete(cutMap, np.array(delFromCut), axis=0)
    numCell -= len(delFromCut)
    # Chronological matching and addition of ROIs
    # When there are matching ROIs, choose the one that has lower perimeter/area ratio
    
    # if masterMap.shape[0]>0:
    masterArea = np.sum(masterMap, axis=(1,2))
    newArea = np.sum(cutMap, axis=(1,2))
    masterPar = perimeter_area_ratio(masterMap)
    newPar = perimeter_area_ratio(cutMap)
    overlaps = np.zeros((masterMap.shape[0], numCell), 'uint16')
    
    # Find if there is any matched ROI, per new cells
    # Calculate overlap and applying the threshold
    for ci in range(numCell):
        overlaps[:,ci] = np.sum(masterMap*cutMap[ci,:,:], axis=(1,2))
    overlapRatioMaster = overlaps/np.tile(np.expand_dims(masterArea, axis=1), (1,numCell))
    overlapRatioNew = overlaps/np.tile(np.expand_dims(newArea, axis=0), (masterMap.shape[0],1))
    overlapMatrixOld = np.logical_or(overlapRatioMaster>=roiOverlapThresh, overlapRatioNew>=roiOverlapThresh)
    # # Added matching calculation: Overlap pix # > roiOverlapThresh of median ROI pix #
    # # Median ROI calcualted from masterMap. If masterMap does not exist, then cutMap
    if len(masterArea) > 0:
        roiPixThresh = roiOverlapThresh * np.median(masterArea)
    else:
        roiPixThresh = roiOverlapThresh * np.median(newArea)
    overlapMatrix = np.logical_or(overlaps > roiPixThresh, overlapMatrixOld)
    
    # Deal with error cases where there can be multiple matching
    multiMatchNewInd = np.where(np.sum(overlapMatrix, axis=0)>1)[0]
    multiMatchMasterInd = np.where(np.sum(overlapMatrix, axis=1)>1)[0]
    
    # Deal with multi-matching pairs
    # First with master ROI, then with new ROIs, because there can be redundancy in multi-matching pairs
    delFromMasterInd = []
    delFromNewInd = []
    delFromMasterInd, delFromNewInd, multiMatchMasterInd, multiMatchNewInd  = \
        check_multi_match_pair(multiMatchMasterInd, multiMatchNewInd, masterPar, newPar, 
                                overlapMatrix, overlaps, delFromMasterInd, delFromNewInd)
    
    if len(multiMatchNewInd)>0 or len(multiMatchMasterInd)>0:
        print(f'{len(multiMatchNewInd)} multi-match for new rois')
        print(f'{len(multiMatchMasterInd)} multi-match for master rois')
        raise('Multiple matches found after fixing multi-match pairs.')
    else:
        ################ Select what to remove for matched cells based on Par
        # For now, treat if there is no multiple matches
        # Do this until I comb through all examples that I have and decide how to treat
        # multiple matches

        for ci in range(numCell): # for every new roi
            if ci in delFromNewInd:
                continue
            else:
                matchedMasterInd = np.where(overlapMatrix[:,ci]==True)[0]
                matchedMasterInd = np.array([mi for mi in matchedMasterInd if mi not in delFromMasterInd])
                if len(matchedMasterInd)>0: # found a match in the master roi
                    # Compare perimeter-area ratio (par) between the matches
                    # Keep smaller par, remove larger par
                    if masterPar[matchedMasterInd] <= newPar[ci]:
                        delFromNewInd.append(ci)
                    else:
                        delFromMasterInd.append(matchedMasterInd[0])
        if len(delFromMasterInd)>0:
            newMasterMap = np.delete(masterMap, np.array(delFromMasterInd), axis=0)
            roiSessionInd = np.delete(roiSessionInd, np.array(delFromMasterInd))
        else:
            newMasterMap = masterMap.copy()
        if len(delFromNewInd)>0:
            newMap = np.delete(cutMap, np.array(delFromNewInd), axis=0)
        else:
            newMap = cutMap.copy()
        roiNewSessionInd = np.ones(newMap.shape[0])*si    
        print(f'Delete from Master {delFromMasterInd}')    
        print(f'Delete from New {delFromNewInd}')
    
        finalMasterMap = np.vstack((newMasterMap, newMap))
        roiSessionInd = np.concatenate((roiSessionInd, roiNewSessionInd))
        
        ################## Show all rois summed up to check overlaps
        # fig, ax = plt.subplots(2,3, figsize=(14,7))
        ax[0,0].imshow(np.sum(masterMap, axis=0), vmin=0, vmax=2)
        ax[0,0].set_title(f'Master map before {sname}')
        ax[0,0].set_axis_off()
        
        ax[1,0].imshow(np.sum(newMasterMap, axis=0), vmin=0, vmax=2)
        ax[1,0].set_title(f'Master map after\nremoving overlaps with {sname}')
        ax[1,0].set_axis_off()
        
        ax[0,1].imshow(np.sum(cutMap, axis=0), vmin=0, vmax=2)
        ax[0,1].set_title(f'all ROIs from {sname}')
        ax[0,1].set_axis_off()
        
        ax[1,1].imshow(np.sum(newMap, axis=0), vmin=0, vmax=2)
        ax[1,1].set_title(f'new ROIs from {sname}')
        ax[1,1].set_axis_off()
        
        ax[1,2].imshow(np.sum(finalMasterMap, axis=0)+np.sum(newMap, axis=0)*2, vmin=0, vmax=2)
        ax[1,2].set_title(f'Master map after {sname}')
        ax[1,2].set_axis_off()
        
        ax[0,2].set_axis_off()
        
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.005, hspace=0.001)
        fig.savefig(f'C:/Users/shires/Dropbox/Works/Projects/2020 Neural stretching in S1/Analysis/JK{mouse:03}_pn{pn}_si{si}.png', bbox_inches='tight')
        
        masterMap = finalMasterMap.copy()
        print(f'{sname} done.')
print('Collection done.')


#%% Save the results
# Depths
# Selected sessions (removed session index, including all session names)
# Master map

saveFn = f'{planeDir}JK{mouse:03}plane{pn}masterROI.npy'

masterROI = {}
masterROI['info'] = {}
masterROI['info']['mouse'] = mouse
masterROI['info']['vi'] = vi
masterROI['info']['plane'] = pn
masterROI['info']['expTestSnum'] = expTestSnum
masterROI['info']['refSnum'] = refSessions[mi]
masterROI['info']['manRmvSi'] = manRmvSi # the index is from selectedSi, not from all the sessions. As a result, selectedSnums is modified.

masterROI['selDepthsRV'] = selDepthsRV
masterROI['selectedSnums'] = selectedSnums

masterROI['masterMap'] = masterMap
masterROI['roiSessionInd'] = roiSessionInd

np.save(saveFn,masterROI)

print('Saving done.')




#%% Weird registration error

napari.view_image(regImgs)

#%%
transMimg = []
transCellmap = []
mimgs = []
cellMaps = []
for si in [0,2,3,4]:
    snum = selectedSnums[si]
    sname = f'{mouse:03}_{snum:03}'
    print(f'Processing {sname} {si}/{len(selectedSi)-1}')
    tempRegi = np.where(np.array(reg['sessionNames'])==sname)[0][0]
    
    rigid_y1 = reg['rigid_offsets_1st'][0][0][tempRegi]
    rigid_x1 = reg['rigid_offsets_1st'][0][1][tempRegi]
    nonrigid_y1 = reg['nonrigid_offsets_1st'][0][0][tempRegi,:]
    nonrigid_x1 = reg['nonrigid_offsets_1st'][0][1][tempRegi,:]
    
    rigid_y2 = reg['rigid_offsets_2nd'][0][0][tempRegi]
    rigid_x2 = reg['rigid_offsets_2nd'][0][1][tempRegi]
    nonrigid_y2 = reg['nonrigid_offsets_2nd'][0][0][tempRegi,:]
    nonrigid_x2 = reg['nonrigid_offsets_2nd'][0][1][tempRegi,:]
    
    # Gather cell map
    tempStat = np.load(f'{planeDir}{snum:03}/plane0/stat.npy', allow_pickle=True)
    tempIscell = np.load(f'{planeDir}{snum:03}/plane0/iscell.npy', allow_pickle=True)
    tempCelli = np.where(tempIscell[:,0])[0]
    numCell = len(tempCelli)
    tempMap = np.zeros((numCell,*regImgs.shape[1:]), 'bool')
    for n, ci in enumerate(tempCelli):
        for pixi in range(len(tempStat[ci]['ypix'])):
            xi = tempStat[ci]['xpix']
            yi = tempStat[ci]['ypix']
            tempMap[n,yi,xi] = 1
    
    tempOps = np.load(f'{planeDir}{snum:03}/plane0/ops.npy', allow_pickle=True).item()
    
    mimg = tempOps['meanImg']
    mimgs.append(mimg)
    cellMaps.append(np.sum(tempMap, axis=0))
    
    
    transMimg.append(twostep_register(mimg, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                             rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2))
    transCellmap.append(twostep_register(np.sum(tempMap, axis=0), rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                             rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2))

viewer = napari.Viewer()
viewer.add_image(np.array(transMimg))
viewer.add_image(np.array(transCellmap))




#%% When there are multi-matches for master rois
mci = 1
masterCi = multiMatchMasterInd[mci]
newCis = np.where(overlapMatrix[masterCi,:])[0]

viewer = napari.Viewer()
viewer.add_image(masterMap[masterCi,:,:], name='from Master')
errorMap = imblend_for_napari(cutMap[newCis[0],:,:].astype(int), cutMap[newCis[1],:,:].astype(int))
viewer.add_image(errorMap, rgb=True, name='new map')

print(f'Master ROI index = {masterCi}')
for nc in newCis:
    # print(f'New ROI index from stat.npy = {tempCelli[nc]}') # to print out ROI index from stat.npy
    print(f'New ROI index from cutMap = {nc}')



#%% When there are multi-matches for new rois
nci = 0
newCi = multiMatchNewInd[nci]
masterCis = np.where(overlapMatrix[:,newCi])[0]

viewer = napari.Viewer()
viewer.add_image(cutMap[newCi,:,:], name='New map')
errorMap = imblend_for_napari(masterMap[masterCis[0],:,:].astype(int), masterMap[masterCis[1],:,:].astype(int))
viewer.add_image(errorMap, rgb=True, name='Master map')

# print(f'New ROI index from stat.npy = {tempCelli[newCi]}') 
print(f'New ROI index from cutMap = {newCi}')
for mc in masterCis:
    print(f'Master ROI index = {mc}')





#%% Get the index of a specific neuron at a specific position
ypix = 141
xpix = 672

positionMap = np.zeros((masterMap.shape[1:]))
positionMap[ypix,xpix]=1
masterCi = np.where(np.sum(masterMap*positionMap, axis=(1,2)))[0]

napari.view_image(imblend_for_napari(masterMap[masterCi[0],:,:].astype(int), masterMap[masterCi[1],:,:].astype(int)))


#%% Find the sessions
fromSession = roiSessionInd[masterCi].astype(int)
print(sname[fromSession[0]])
print(sname[fromSession[1]])





















#%% Pair searching test for new map
nci = 1
newCi = multiMatchNewInd[nci]
print(f'newCi = {newCi}')

masterCis = np.where(overlapMatrix[:,newCi])[0]
print(f'masterCis = {masterCis}')
    
newBestMatchi = np.zeros(len(masterCis), 'int')
for i, mci in enumerate(masterCis):
    newBestMatchi[i] = np.argmax(overlaps[mci,:]).astype(int)
print(f'newBestMatchi = {newBestMatchi}')

masterBestMatchi = np.zeros(len(newBestMatchi), 'int')
for i, nci in enumerate(newBestMatchi):
    masterBestMatchi[i] = np.argmax(overlaps[:,nci]).astype(int)
print(f'masterBestMatchi = {masterBestMatchi}')
    
if all(masterCis == masterBestMatchi): # found a matching pair
    # Calculate mean perimeter/area ratio
    masterMeanpar = np.mean(masterPar[masterCis])
    newMeanpar = np.mean(newPar[newBestMatchi])

#%% Show the overlap with emphasis on the multi-matching cell
nci = 1
newCi = multiMatchNewInd[nci]
masterCis = np.where(overlapMatrix[:,newCi])[0]

viewer = napari.Viewer()
newMMmap = imblend_for_napari(np.sum(cutMap,axis=0).astype(int), cutMap[newCi,:,:].astype(int))
viewer.add_image(newMMmap, name='New map')
errorMap = imblend_for_napari(masterMap[masterCis[0],:,:].astype(int), masterMap[masterCis[1],:,:].astype(int))
viewer.add_image(errorMap, rgb=True, name='Master map')


#%% Pair searching test for master map
mci = 0
masterCi = multiMatchMasterInd[mci]
print(f'masterCi = {masterCi}')

newCis = np.where(overlapMatrix[masterCi,:])[0]
print(f'newCis = {newCis}')
    
masterBestMatchi = np.zeros(len(newCis), 'int')
for i, nci in enumerate(newCis):
    masterBestMatchi[i] = np.argmax(overlaps[:,nci]).astype(int)
print(f'masterBestMatchi = {masterBestMatchi}')

newBestMatchi = np.zeros(len(masterBestMatchi), 'int')
for i, mci in enumerate(masterBestMatchi):
    newBestMatchi[i] = np.argmax(overlaps[mci,:]).astype(int)
print(f'newBestMatchi = {newBestMatchi}')
    
if all(newCis == newBestMatchi): # found a matching pair
    # Calculate mean perimeter/area ratio
    masterMeanpar = np.mean(masterPar[masterBestMatchi])
    newMeanpar = np.mean(newPar[newBestMatchi])
    
#%%
viewer = napari.Viewer()
viewer.add_image(masterMap[masterCi,:,:], name='from Master')
errorMap = imblend_for_napari(cutMap[newCis[0],:,:].astype(int), cutMap[newCis[1],:,:].astype(int))
viewer.add_image(errorMap, rgb=True, name='new map')

#%%
tempDfMasterInd = np.zeros(0,'int')
tempDfNewInd = np.zeros(0,'int')
remFromMMmaster = np.zeros(0,'int') # Collect multiMatchMasterInd that is already processed (in the for loop)
remFromMMnew = np.zeros(0,'int') # Collect multiMatchNewInd that is already processed (in the for loop)
# Remove remFromMM* at the end.

# First, deal with delFromMasterInd
if len(multiMatchMasterInd)>0:
    for mci in range(len(multiMatchMasterInd)):
        masterCi = multiMatchMasterInd[mci]
        if masterCi in remFromMMmaster:
            continue
        else:
            newCis = np.where(overlapMatrix[masterCi,:])[0]
                
            masterBestMatchi = np.zeros(len(newCis), 'int')
            for i, nci in enumerate(newCis):
                masterBestMatchi[i] = np.argmax(overlaps[:,nci]).astype(int)
            # Check if there are multiple same matched IDs 
            # In this case, just remove the masterInd
            if check_multi_element(masterBestMatchi):    
                tempDfMasterInd = np.hstack((tempDfMasterInd, [masterCi]))
                remFromMMmaster = np.hstack((remFromMMmaster, [masterCi]))
            
            # Else, check if there is a matching pair (or multiples)
            else:
                newBestMatchi = np.zeros(len(masterBestMatchi), 'int')
                for i, mci in enumerate(masterBestMatchi):
                    newBestMatchi[i] = np.argmax(overlaps[mci,:]).astype(int)

                if all(newCis == newBestMatchi): # found a matching pair
                    # Calculate mean perimeter/area ratio
                    masterMeanpar = np.mean(masterPar[masterBestMatchi])
                    newMeanpar = np.mean(newPar[newBestMatchi])
                    
                    # Remove the pair with lower mean par
                    if masterMeanpar >= newMeanpar:
                        tempDfNewInd = np.hstack((tempDfNewInd, newBestMatchi))
                    else:
                        tempDfMasterInd = np.hstack((tempDfMasterInd, masterBestMatchi))

                    # Collect indices already processed
                    remFromMMmaster = np.hstack((remFromMMmaster, masterBestMatchi))
                    remFromMMnew = np.hstack((remFromMMnew, newBestMatchi))
                          
# Then, deal with delFromNewInd
if len(multiMatchNewInd)>0:
    for nci in range(len(multiMatchNewInd)):
        newCi = multiMatchNewInd[nci]
        if newCi in remFromMMnew:
            continue
        else:
            masterCis = np.where(overlapMatrix[:,newCi])[0]
                
            newBestMatchi = np.zeros(len(masterCis), 'int')
            for i, mci in enumerate(masterCis):
                newBestMatchi[i] = np.argmax(overlaps[mci,:]).astype(int)
                
            # Check if there are multiple same matched IDs 
            # In this case, just remove the masterInd
            if check_multi_element(newBestMatchi):    
                tempDfNewInd = np.hstack((tempDfNewInd, [newCi]))
                remFromMMnew = np.hstack((remFromMMnew, [newCi]))
            
            # Else, check if there is a matching pair (or multiples)
            else:
                masterBestMatchi = np.zeros(len(newBestMatchi), 'int')
                for i, nci in enumerate(newBestMatchi):
                    masterBestMatchi[i] = np.argmax(overlaps[:,nci]).astype(int)

                if all(masterCis == masterBestMatchi): # found a matching pair
                    # Calculate mean perimeter/area ratio
                    masterMeanpar = np.mean(masterPar[masterBestMatchi])
                    newMeanpar = np.mean(newPar[newBestMatchi])
                    
                    # Remove the pair with lower mean par
                    if masterMeanpar >= newMeanpar:
                        tempDfNewInd = np.hstack((tempDfNewInd, newBestMatchi))
                    else:
                        tempDfMasterInd = np.hstack((tempDfMasterInd, masterBestMatchi))

                    # Collect indices already processed
                    remFromMMmaster = np.hstack((remFromMMmaster, masterBestMatchi))
                    remFromMMnew = np.hstack((remFromMMnew, newBestMatchi))

# Remove collected indices
delFromMasterInd.extend(tempDfMasterInd)
delFromNewInd.extend(tempDfNewInd)
multiMatchMasterInd = delete_elements_in_another_array(multiMatchMasterInd, remFromMMmaster)
multiMatchNewInd = delete_elements_in_another_array(multiMatchNewInd, remFromMMnew)
    
#%%
print(f'delFromMasterInd = {delFromMasterInd}')
print(f'delFromNewInd = {delFromNewInd}')
print(f'multiMatchMasterInd = {multiMatchMasterInd}')
print(f'multiMatchNewInd = {multiMatchNewInd}')


#%%%%%%%%%%%%%%%%%%


# Visual inspection of the master map

# Go through each session and apply master map into each session

# Gather fluorescence and inferred spikes data

# Check the data






