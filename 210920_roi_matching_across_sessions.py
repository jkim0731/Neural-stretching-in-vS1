# -*- coding: utf-8 -*-
"""
ROI matching across sessions

- Base ROI map from the stitched.
- Add ROIs from each session. One by one, chronological order.
- But, how??

2021/09/20 JK
"""
import numpy as np
from matplotlib import pyplot as plt
from suite2p.run_s2p import run_s2p
# from auto_pre_cur import auto_pre_cur
import os, glob, shutil
from suite2p.io.binary import BinaryFile
from suite2p.registration.register import enhanced_mean_image
from suite2p.registration import rigid, nonrigid
from suite2p.gui import drawroi

import napari
import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'
s2pDir = 'D:/TPM/JK/s2p/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]

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

def twostep_register(img, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                     rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2):
    frames = img.copy()
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
    
#%% Matching vs non-matching
# First, by spatial footprint.
# Distribution of spatial overlap proportion. Pix # of new / Intersection.

mi = 0
mouse = mice[mi]
pn = 3
s2pPlaneDir = f'{s2pDir}{mouse:03}/plane_{pn}/plane0/'
h5PlaneDir = f'{h5Dir}{mouse:03}/plane_{pn}/'

stitchedOps = np.load(f'{s2pPlaneDir}stitched_ops.npy', allow_pickle=True).item()
Ly = stitchedOps['opsList'][0]['Ly']
Lx = stitchedOps['opsList'][0]['Lx']
umPerPix = stitchedOps['opsList'][0]['umPerPix']

allOps = np.load(f'{s2pPlaneDir}ops.npy', allow_pickle=True).item()
allSpk = np.load(f'{s2pPlaneDir}spks.npy', allow_pickle=True).tolist()
allStat = np.load(f'{s2pPlaneDir}stat.npy', allow_pickle=True).tolist()
allF = np.load(f'{s2pPlaneDir}F.npy', allow_pickle=True).tolist()
allIscell = np.load(f'{s2pPlaneDir}iscell.npy', allow_pickle=True).tolist()

allCellInds = np.where(np.array([ic[0] for ic in allIscell]))[0]
allSpk = [np.array(allSpk[i]) for i in allCellInds]
allF = [np.array(allF[i]) for i in allCellInds]
allStat = [allStat[i] for i in allCellInds]
allCellMap = np.zeros((len(allCellInds), Ly, Lx), dtype=bool)
for i, _ in enumerate(allCellInds):
    allCellMap[i,allStat[i]['ypix'], allStat[i]['xpix']] = 1
allMed = np.array([stat['med'] for stat in allStat])
frameNums = np.concatenate(([0], np.array(np.cumsum(stitchedOps['nFrames']))))

jaccardInd = [] # Jaccard index (intersection / union)
centDiff = [] # centroid difference (= distance)
sigCorr = [] # signal correlation
distOrder = [] # 1,2, or 3 (just include the closest 3 that overlaps)
pixValThresh = 0.33
numpixThresh = 10
nSessions = len(stitchedOps['useSessionNames'])

# for si in range(nSessions)
si = 0
sn = stitchedOps['useSessionNames'][si][4:]
sessionPlaneDir = f'{h5PlaneDir}{sn}/plane0/'
sessionOps = np.load(f'{sessionPlaneDir}ops.npy', allow_pickle=True).item()
sessionSpk = np.load(f'{sessionPlaneDir}spks.npy', allow_pickle=True).tolist()
sessionStat = np.load(f'{sessionPlaneDir}stat.npy', allow_pickle=True).tolist()
sessionF = np.load(f'{sessionPlaneDir}F.npy', allow_pickle=True).tolist()
sessionIscell = np.load(f'{sessionPlaneDir}iscell.npy', allow_pickle=True).tolist()

sessionCellInds = np.where(np.array([ic[0] for ic in sessionIscell]))[0]
sessionSpk = [np.array(sessionSpk[i]) for i in sessionCellInds]
sessionF = [np.array(sessionF[i]) for i in sessionCellInds]
sessionStat = [sessionStat[i] for i in sessionCellInds]
sessionCellMap = np.zeros((len(sessionCellInds), Ly, Lx), dtype=bool)
for i, _ in enumerate(sessionCellInds):
    sessionCellMap[i,sessionStat[i]['ypix'], sessionStat[i]['xpix']] = 1

# 2-step registration of the ROI map
rigid_y1 = stitchedOps['rigid_offsets_y1'][si]
rigid_x1 = stitchedOps['rigid_offsets_x1'][si]
nonrigid_y1 = stitchedOps['nonrigid_offsets_y1'][si]
nonrigid_x1 = stitchedOps['nonrigid_offsets_x1'][si]
block_size1 = stitchedOps['block_size1']
rigid_y2 = stitchedOps['rigid_offsets_y2'][si]
rigid_x2 = stitchedOps['rigid_offsets_x2'][si]
nonrigid_y2 = stitchedOps['nonrigid_offsets_y2'][si]
nonrigid_x2 = stitchedOps['nonrigid_offsets_x2'][si]
block_size2 = stitchedOps['block_size2']

Ly, Lx = sessionOps['Ly'], sessionOps['Lx']
tempCellMap = np.zeros((len(sessionCellInds),Ly,Lx), dtype=np.int16)
for ci in range(len(sessionCellInds)):
    tempCellMap[ci,sessionStat[ci]['ypix'], sessionStat[ci]['xpix']] = 1
sessionRegCellMap = twostep_register(tempCellMap, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                     rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2)    

sessionRegCellMap[np.where(sessionRegCellMap>pixValThresh)] = 1
sessionRegCellMap = np.round(sessionRegCellMap).astype(bool)
edgeMap = np.zeros((Ly,Lx),dtype=bool)
edgeMap[0,:] = 1
edgeMap[-1,:] = 1
edgeMap[:,0] = 1
edgeMap[:,-1] = 1

edgeRoiBool = np.sum(edgeMap * sessionRegCellMap, axis=(1,2)) > 0
smallRoiBool = np.sum(sessionRegCellMap) < numpixThresh
withinFOVInd = np.where(edgeRoiBool + smallRoiBool == 0)[0]


'''
From here everything's in withinFOVInd 
'''
#%% Distribution of proportion set diff 

numpixOverlap = np.zeros((len(withinFOVInd),len(allCellInds)), dtype=np.int16)
numpixNew = np.zeros((len(withinFOVInd),), dtype=np.int16)
allpixMap = np.sum(allCellMap, axis=0) > 0
for i, ci in enumerate(withinFOVInd):
    numpixNew[i] = len(np.where(sessionRegCellMap[ci,:,:] - allpixMap.astype(np.int8) == 1)[0])
    for j in range(len(allCellInds)):
        numpixOverlap[i,j] = np.sum(sessionRegCellMap[ci,:,:]*allCellMap[j,:,:])

numpixAll = np.sum(allCellMap, axis=(1,2))
maxOverlapInd = np.argmax(numpixOverlap, axis=1)
numpixAllOverlapping = numpixAll[maxOverlapInd]
numpixSession = np.sum(sessionRegCellMap[withinFOVInd,:,:],axis=(1,2))

#%%
# overlapThreshold = allOps['max_overlap']
overlapThreshold = 0.5
propOverlapNew = np.amax(numpixOverlap, axis=1) / numpixSession
propOverlapOld = np.amax(numpixOverlap, axis=1) / numpixAllOverlapping
tempValidInd = np.intersect1d(np.where(propOverlapNew < overlapThreshold)[0],
                          np.where(propOverlapOld < overlapThreshold)[0])
validNewRoiInd = withinFOVInd[tempValidInd]

propNew = numpixNew / numpixSession


#%%
newRoiMap = np.sum(sessionRegCellMap[validNewRoiInd,:,:], axis=0)
fig, ax = plt.subplots()
ax.imshow(imblend(allpixMap, newRoiMap))

#%%
newRoiMap = np.sum(sessionRegCellMap, axis=0)
fig, ax = plt.subplots()
ax.imshow(imblend(allpixMap, newRoiMap))

'''
First pass with overlap proportion works well.
It should be applied in both ways.
There are some heavily overlapped ROIs.
'''
#%%

# fig, ax = plt.subplots()
# ax.hist(propDiff, bins=np.linspace(0,1,20))
# ax.set_xlabel('Proportion of additional pix / ROI pix')
# ax.set_ylabel('# of cells')

#%% Signal correlation vs overlap proportion

sigcorOverlap = np.zeros((len(validNewRoiInd),))
for vi, ci in enumerate(validNewRoiInd):
    tempSessionSpk = sessionSpk[ci]
    tempOverlapSpk = allSpk[maxOverlapInd[tempValidInd[vi]]][frameNums[si]:frameNums[si+1]]
    sigcorOverlap[vi] = np.corrcoef(tempSessionSpk, tempOverlapSpk)[0,1]

fig, ax = plt.subplots()
ax.plot(propOverlapNew[tempValidInd], sigcorOverlap, 'k.')
ax.set_xlabel('Overlap pix proportion')
ax.set_ylabel('Signal correlation')


#%% Signal extracted from new additional pixels

stat0 = []
for ci in validNewRoiInd:
    ypix = sessionStat[ci]['ypix']
    xpix = sessionStat[ci]['xpix']
    lam = np.ones(ypix.shape)
    # stat0.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size, 'med': med})
    stat0.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size})

# It takes about 5 min for 223 rois with 116k frames
F, Fneu, F_chan2, Fneu_chan2, spks, ops, stat = drawroi.masks_and_traces(allOps, stat0, allStat)

#%% Signal correlation vs prop set diff,
# after adding ROIs (removing overlapping pixels from signal calculation)
# Across all frames
newRoiSigCorr = np.zeros(len(validNewRoiInd))
for ci in range(len(validNewRoiInd)):
    tempSessionSpk = spks[ci]
    tempOverlapSpk = allSpk[maxOverlapInd[tempValidInd[ci]]]
    newRoiSigCorr[ci] = np.corrcoef(tempSessionSpk, tempOverlapSpk)[0,1]
        # tempSessionF = F[ci]
        # tempOverlapF = allF[maxIntersectInd[ci]]
        # newRoiSigCorr[ci] = np.corrcoef(tempSessionF, tempOverlapF)[0,1]

fig, ax = plt.subplots()
ax.plot(propOverlapNew[tempValidInd], newRoiSigCorr, 'k.')
ax.set_xlabel('Overlap pix proportion')
ax.set_ylabel('Signal correlation (Stitched)')
ax.set_title(f'Overlap threshold {overlapThreshold}')
#%% Distribution of signal correlation between nearest neuron pairs
# vs distance?
nearestPairSigCorr = np.zeros(len(allCellInds))
nearestDist = np.zeros(len(allCellInds))
for ci in range(len(allCellInds)):
    allDist = (np.sum((allMed - allStat[ci]['med'])**2, axis=1))**0.5 * allOps['umPerPix']
    nearestDistTemp = np.amin(allDist[allDist>0.5]) # to remove 0. Same neuron
    nearestInd = np.where(allDist==nearestDistTemp)[0][0]
    nearestPairSigCorr[ci] = np.corrcoef(allSpk[ci], allSpk[nearestInd])[0,1]
    nearestDist[ci] = nearestDistTemp

fig, ax = plt.subplots()
ax.plot(nearestDist,nearestPairSigCorr,  'k.')
ax.set_xlabel('Centroid difference (um)')
ax.set_ylabel('Signal correlation')
ax.set_title('Nearest pairs in the stitched')

'''
All nearest pairs have < 0.5 signal correlation over all the sessions.
There are some that exceeds 0.3 and one even larger than 0.4.
Newly added had maximum 0.2, so I think it is OK.
'''




#%% Test with all the sessions
# Save new roi map into a subfolder
# 

mi = 0
mouse = mice[mi]
pn = 3
s2pPlaneDir = f'{s2pDir}{mouse:03}/plane_{pn}/plane0/'
h5PlaneDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
resultDir = f'{s2pPlaneDir}/roimerge/'

stitchedOps = np.load(f'{s2pPlaneDir}stitched_ops.npy', allow_pickle=True).item()
Ly = stitchedOps['opsList'][0]['Ly']
Lx = stitchedOps['opsList'][0]['Lx']
# umPerPix = stitchedOps['opsList'][0]['umPerPix']

allOps = np.load(f'{s2pPlaneDir}ops.npy', allow_pickle=True).item()
allSpk = np.load(f'{s2pPlaneDir}spks.npy', allow_pickle=True).tolist()
allStat = np.load(f'{s2pPlaneDir}stat.npy', allow_pickle=True).tolist()
allF = np.load(f'{s2pPlaneDir}F.npy', allow_pickle=True).tolist()
allFneu = np.load(f'{s2pPlaneDir}Fneu.npy', allow_pickle=True).tolist()
allIscell = np.load(f'{s2pPlaneDir}iscell.npy', allow_pickle=True).tolist()

allCellInds = np.where(np.array([ic[0] for ic in allIscell]))[0]
allSpk = [np.array(allSpk[i]) for i in allCellInds]
allF = [np.array(allF[i]) for i in allCellInds]
allFneu = [np.array(allFneu[i]) for i in allCellInds]
allStat = [allStat[i] for i in allCellInds]
allCellMap = np.zeros((len(allCellInds), Ly, Lx), dtype=bool)
for i, _ in enumerate(allCellInds):
    allCellMap[i,allStat[i]['ypix'], allStat[i]['xpix']] = 1
# allMed = np.array([stat['med'] for stat in allStat])
# frameNums = np.concatenate(([0], np.array(np.cumsum(stitchedOps['nFrames']))))
allIscell = [allIscell[i] for i in allCellInds]
# Run each session
pixValThresh = 0.33
numpixThresh = 10
overlapThreshold = 0.5
nSessions = len(stitchedOps['useSessionNames'])
statAdd = []
# fig, ax = plt.subplots(figsize=(25.5,15))
for si in range(nSessions):
    print(f'Processing JK{mouse:03} session #{si}/{nSessions}')
    sn = stitchedOps['useSessionNames'][si][4:]
    sessionPlaneDir = f'{h5PlaneDir}{sn}/plane0/'
    # sessionOps = np.load(f'{sessionPlaneDir}ops.npy', allow_pickle=True).item()
    # sessionSpk = np.load(f'{sessionPlaneDir}spks.npy', allow_pickle=True).tolist()
    sessionStat = np.load(f'{sessionPlaneDir}stat.npy', allow_pickle=True).tolist()
    # sessionF = np.load(f'{sessionPlaneDir}F.npy', allow_pickle=True).tolist()
    sessionIscell = np.load(f'{sessionPlaneDir}iscell.npy', allow_pickle=True).tolist()
    
    sessionCellInds = np.where(np.array([ic[0] for ic in sessionIscell]))[0]
    # sessionSpk = [np.array(sessionSpk[i]) for i in sessionCellInds]
    # sessionF = [np.array(sessionF[i]) for i in sessionCellInds]
    sessionStat = [sessionStat[i] for i in sessionCellInds]
    sessionCellMap = np.zeros((len(sessionCellInds), Ly, Lx), dtype=bool)
    for i, _ in enumerate(sessionCellInds):
        sessionCellMap[i,sessionStat[i]['ypix'], sessionStat[i]['xpix']] = 1
    sessionIscell = [sessionIscell[i] for i in sessionCellInds]
    
    # 2-step registration of the ROI map
    rigid_y1 = stitchedOps['rigid_offsets_y1'][si]
    rigid_x1 = stitchedOps['rigid_offsets_x1'][si]
    nonrigid_y1 = stitchedOps['nonrigid_offsets_y1'][si]
    nonrigid_x1 = stitchedOps['nonrigid_offsets_x1'][si]
    block_size1 = stitchedOps['block_size1']
    rigid_y2 = stitchedOps['rigid_offsets_y2'][si]
    rigid_x2 = stitchedOps['rigid_offsets_x2'][si]
    nonrigid_y2 = stitchedOps['nonrigid_offsets_y2'][si]
    nonrigid_x2 = stitchedOps['nonrigid_offsets_x2'][si]
    block_size2 = stitchedOps['block_size2']
    
    # Ly, Lx = sessionOps['Ly'], sessionOps['Lx']
    tempCellMap = np.zeros((len(sessionCellInds),Ly,Lx), dtype=np.int16)
    for ci in range(len(sessionCellInds)):
        tempCellMap[ci,sessionStat[ci]['ypix'], sessionStat[ci]['xpix']] = 1
    sessionRegCellMap = twostep_register(tempCellMap, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                         rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2)    
    
    # Use the ROIs that are not touching the edge and have reasonable # of pix
    sessionRegCellMap[np.where(sessionRegCellMap>pixValThresh)] = 1
    sessionRegCellMap = np.round(sessionRegCellMap).astype(bool)
    edgeMap = np.zeros((Ly,Lx),dtype=bool)
    edgeMap[0,:] = 1
    edgeMap[-1,:] = 1
    edgeMap[:,0] = 1
    edgeMap[:,-1] = 1
    
    edgeRoiBool = np.sum(edgeMap * sessionRegCellMap, axis=(1,2)) > 0
    smallRoiBool = np.sum(sessionRegCellMap) < numpixThresh
    withinFOVInd = np.where(edgeRoiBool + smallRoiBool == 0)[0]
    
    # From here everything's in withinFOVInd 

    # Distribution of overlap proportion 
    allpixMap = (np.sum(allCellMap, axis=0) > 0).astype(np.int8)
    negPixMap = 1 - allpixMap
    propNew = np.sum(sessionRegCellMap[withinFOVInd,:,:] * negPixMap, axis=(1,2)) / np.sum(sessionRegCellMap[withinFOVInd,:,:], axis=(1,2))
    propOverlap = np.zeros((len(withinFOVInd),allCellMap.shape[0]))
    for i, ci in enumerate(withinFOVInd):
        propOverlap[i,:] = np.sum(sessionRegCellMap[ci,:,:] * allCellMap, axis=(1,2)) / np.sum(allCellMap, axis=(1,2))
    
    
    tempValidInd = np.intersect1d(np.where(propNew > (1-overlapThreshold))[0],
                              np.where(np.amax(propOverlap,axis=1) < overlapThreshold)[0])
    
    # From here back to session index
    validNewRoiInd = withinFOVInd[tempValidInd]
    newIscell = [sessionIscell[i] for i in validNewRoiInd]
    newCellMap = sessionRegCellMap[validNewRoiInd,:,:]
    for i in range(newCellMap.shape[0]):
        ypix, xpix = np.where(newCellMap[i,:,:])
        lam = np.ones(ypix.shape)
        # stat0.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size, 'med': med})
        statAdd.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size})
        
    # ax.imshow(imblend(np.sum(allCellMap,axis=0), np.sum(newCellMap,axis=0)))
    # plt.pause(0.05)
    # input('Press Enter to continue...')
    allIscell = [*allIscell, *newIscell]
    # Update allCellMap
    allCellMap = np.concatenate((allCellMap, newCellMap), axis=0)
    #%%
# It takes about 5 min for 223 rois with 116k frames
F, Fneu, F_chan2, Fneu_chan2, spks, ops, stat = drawroi.masks_and_traces(allOps, statAdd, allStat)

# Merge results
mergeStat = allStat + stat
mergeF = [*allF, *F]
mergeFneu = [*allFneu, *Fneu]
mergeSpks = [*allSpk, *spks]
#%% Save files
if ~os.path.isdir(resultDir):
    os.mkdir(resultDir)
np.save(f'{resultDir}ops.npy', ops)
np.save(f'{resultDir}F.npy', mergeF)
np.save(f'{resultDir}Fneu.npy', mergeFneu)
np.save(f'{resultDir}spks.npy', mergeSpks)
np.save(f'{resultDir}stat.npy', mergeStat)
np.save(f'{resultDir}iscell.npy', allIscell)


#%%
allMed = np.array([st['med'] for st in mergeStat])
nearestPairSigCorr = np.zeros(len(mergeStat))
nearestDist = np.zeros(len(mergeStat))
for ci in range(len(mergeStat)):
    allDist = (np.sum((allMed - mergeStat[ci]['med'])**2, axis=1))**0.5 * allOps['umPerPix']
    nearestDistTemp = np.amin(allDist[allDist>0.5]) # to remove 0. Same neuron
    nearestInd = np.where(allDist==nearestDistTemp)[0][0]
    nearestPairSigCorr[ci] = np.corrcoef(mergeSpks[ci], mergeSpks[nearestInd])[0,1]
    nearestDist[ci] = nearestDistTemp

fig, ax = plt.subplots()
ax.plot(nearestDist,nearestPairSigCorr,  'k.')
ax.set_xlabel('Centroid difference (um)')
ax.set_ylabel('Signal correlation')
ax.set_title('Nearest pairs in the stitched\nAfter ROI merging')


#%%
aspk = np.array(mergeSpks)
fig, ax = plt.subplots(1,2)
# ax[0].imshow(aspk>1, aspect='auto')
ax[0].imshow(aspk, interpolation = 'none', aspect='auto')
ax[0].set_xlabel('Frames from the stitched')
ax[0].set_ylabel('ROI #')
ax[0].set_title('Frames with ANY activity')
ax[1].plot(np.mean(aspk, axis=1))
ax[1].set_xlabel('ROI #')
ax[1].set_ylabel('Mean activity (spk; AU)')
fig.tight_layout()

#%%
dF = np.array(mergeF) - ops['neucoeff'] * np.array(mergeFneu)
fig, ax = plt.subplots(1,2)
ax[0].imshow(dF, aspect='auto', interpolation = 'none', vmax = 5000)
ax[0].set_xlabel('Frames from the stitched')
ax[0].set_ylabel('ROI #')
ax[0].set_title('dF')
ax[1].plot(np.mean(dF, axis=1))
ax[1].set_xlabel('ROI #')
ax[1].set_ylabel('Mean activity (dF)')
fig.tight_layout()

#%%
dF = np.array(mergeF) - ops['neucoeff'] * np.array(mergeFneu)
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(mergeF, aspect='auto', interpolation = 'none')
ax[0,0].set_xlabel('Frames from the stitched')
ax[0,0].set_ylabel('ROI #')
ax[0,0].set_title('F')
ax[0,1].plot(np.mean(np.array(mergeF), axis=1))
ax[0,1].set_xlabel('ROI #')
ax[0,1].set_ylabel('Mean signal (F)')

ax[1,0].imshow(mergeFneu, aspect='auto', interpolation = 'none')
ax[1,0].set_xlabel('Frames from the stitched')
ax[1,0].set_ylabel('ROI #')
ax[1,0].set_title('Fneu')
ax[1,1].plot(np.mean(np.array(mergeFneu), axis=1))
ax[1,1].set_xlabel('ROI #')
ax[1,1].set_ylabel('Mean neurpil (Fneu)')

fig.tight_layout()


#%% Mean activity by the session

meanAct = np.zeros((aspk.shape[0], nSessions))
for si in range(nSessions):
    meanAct[:,si] = np.mean(aspk[:,frameNums[si]:frameNums[si+1]], axis=1)

meanAct = (meanAct - np.expand_dims(np.amin(meanAct, axis=1),axis=1)) / np.expand_dims((np.amax(meanAct, axis=1) - np.amin(meanAct, axis=1)),axis=1)

fig, ax = plt.subplots()
ax.imshow(meanAct, aspect='auto')

ax.set_xlabel('Session #')
ax.set_ylabel('ROI #')
ax.set_title('Normalized session-mean activity')