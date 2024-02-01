# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 21:58:20 2021

@author: shires
"""
import numpy as np
from matplotlib import pyplot as plt
from suite2p.registration import rigid, nonrigid
import os, glob
import napari
from suite2p.io.binary import BinaryFile
from skimage import exposure
import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'
s2pDir = 'D:/TPM/JK/s2p/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
#%% Testing the result
mi = 0
mouse = mice[mi]
pn = 8
s2pPlaneDir = f'{s2pDir}{mouse:03}/plane_{pn}/'
stitchedOpsFn = f'{s2pPlaneDir}stitched_ops.npy'
stitchedDataFn = f'{s2pPlaneDir}data.bin'
stitchedOps = np.load(stitchedOpsFn, allow_pickle=True).item()
napari.view_image(np.array(stitchedOps['regImgs']), name=f'Plane {pn}')
#%%
Ly = stitchedOps['opsList'][0]['Ly']
Lx = stitchedOps['opsList'][0]['Lx']
with BinaryFile(Ly=Ly, Lx=Lx, read_filename=stitchedDataFn) as f:
    nSessions = len(stitchedOps['useSessionNames'])
    nframes = np.array(stitchedOps['nFrames'])
    startFrames = np.concatenate(([0],np.cumsum(nframes)[:-1]))
    endFrames= np.cumsum(nframes)
    meanImgs = []
    for i in range(nSessions):
        meanImgs.append(f.ix(range(startFrames[i],endFrames[i]), is_slice=True).mean(axis=0))

viewer = napari.Viewer()
for i in range(nSessions):
# for i in range(5):    
    viewer.add_image(np.array([stitchedOps['regImgs'][i], meanImgs[i]]), visible=False)

#%%
opsMimgReg = []
for si in range(5):
    opsMimg = stitchedOps['opsList'][si]['meanImg']
    frames = np.expand_dims(opsMimg, axis=0).astype(np.float32)
    
    rigid_y1 = regOps['rigid_offsets_1st'][0][0][si]
    rigid_x1 = regOps['rigid_offsets_1st'][0][1][si]
    nonrigid_y1 = regOps['nonrigid_offsets_1st'][0][0][si,:]
    nonrigid_x1 = regOps['nonrigid_offsets_1st'][0][1][si,:]
    rigid_y2 = regOps['rigid_offsets_2nd'][0][0][si]
    rigid_x2 = regOps['rigid_offsets_2nd'][0][1][si]
    nonrigid_y2 = regOps['nonrigid_offsets_2nd'][0][0][si,:]
    nonrigid_x2 = regOps['nonrigid_offsets_2nd'][0][1][si,:]
    
    # 1st rigid shift
    frames = np.roll(frames, (-rigid_y1, -rigid_x1), axis=(1,2))
    # 1st nonrigid shift
    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=regOps['block_size1'])
    ymax1 = np.tile(nonrigid_y1, (frames.shape[0],1))
    xmax1 = np.tile(nonrigid_x1, (frames.shape[0],1))
    frames = nonrigid.transform_data(
        data=frames,
        nblocks=nblocks,
        xblock=xblock,
        yblock=yblock,
        ymax1=ymax1,
        xmax1=xmax1,
    )
    # 2nd rigid shift
    frames = np.roll(frames, (-rigid_y2, -rigid_x2), axis=(1,2))
    # 2nd nonrigid shift            
    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=regOps['block_size2'])
    ymax1 = np.tile(nonrigid_y2, (frames.shape[0],1))
    xmax1 = np.tile(nonrigid_x2, (frames.shape[0],1))
    frames = nonrigid.transform_data(
        data=frames,
        nblocks=nblocks,
        xblock=xblock,
        yblock=yblock,
        ymax1=ymax1,
        xmax1=xmax1,
    )
    opsMimgReg.append(frames[0,:,:])
#%%
viewer = napari.Viewer()
for i in range(5):
    viewer.add_image(np.array([opsMimgReg[i], stitchedOps['regImgs'][i], meanImgs[i], opsMimgReg[i]]), visible=False)
#%%
with BinaryFile(Ly=Ly, Lx=Lx, read_filename=stitchedDataFn) as f:
    ff = f.ix(range(0,5))
    
#%%
opsMimgList = []
binMimgList = []
binRegList = []
for si in range(5):
    sn = stitchedOps['useSessionNames'][si][4:]
    dataFn = f'{h5planeDir}{sn}/plane0/data.bin'
    opsMimg = stitchedOps['opsList'][si]['meanImg']
    opsMimgList.append(opsMimg)
    
    with BinaryFile(Ly=Ly, Lx=Lx, read_filename=dataFn) as f:
        tempMimg = f.data.mean(axis=0)
        frames = np.expand_dims(tempMimg, axis=0).astype(np.float32)
    
        rigid_y1 = regOps['rigid_offsets_1st'][0][0][si]
        rigid_x1 = regOps['rigid_offsets_1st'][0][1][si]
        nonrigid_y1 = regOps['nonrigid_offsets_1st'][0][0][si,:]
        nonrigid_x1 = regOps['nonrigid_offsets_1st'][0][1][si,:]
        rigid_y2 = regOps['rigid_offsets_2nd'][0][0][si]
        rigid_x2 = regOps['rigid_offsets_2nd'][0][1][si]
        nonrigid_y2 = regOps['nonrigid_offsets_2nd'][0][0][si,:]
        nonrigid_x2 = regOps['nonrigid_offsets_2nd'][0][1][si,:]
        
        # 1st rigid shift
        frames = np.roll(frames, (-rigid_y1, -rigid_x1), axis=(1,2))
        # 1st nonrigid shift
        yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=regOps['block_size1'])
        ymax1 = np.tile(nonrigid_y1, (frames.shape[0],1))
        xmax1 = np.tile(nonrigid_x1, (frames.shape[0],1))
        frames = nonrigid.transform_data(
            data=frames,
            nblocks=nblocks,
            xblock=xblock,
            yblock=yblock,
            ymax1=ymax1,
            xmax1=xmax1,
        )
        # 2nd rigid shift
        frames = np.roll(frames, (-rigid_y2, -rigid_x2), axis=(1,2))
        # 2nd nonrigid shift            
        yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=regOps['block_size2'])
        ymax1 = np.tile(nonrigid_y2, (frames.shape[0],1))
        xmax1 = np.tile(nonrigid_x2, (frames.shape[0],1))
        frames = nonrigid.transform_data(
            data=frames,
            nblocks=nblocks,
            xblock=xblock,
            yblock=yblock,
            ymax1=ymax1,
            xmax1=xmax1,
        )
        binRegList.append(frames[0,:,:])
        binMimgList.append(tempMimg)
#%%
viewer = napari.Viewer()
for i in range(5):
    viewer.add_image(np.array([opsMimgReg[i], stitchedOps['regImgs'][i], meanImgs[i], binRegList[i]]), visible=False)
viewer = napari.Viewer()
for i in range(5):
    viewer.add_image(np.array([opsMimgList[i], binMimgList[i]]), visible=False)

'''
Each ops meanImg is same as averaging all frames from the corresponding bin file.
Registration using each ops meanImg is same as registration using all-frame average of each corresponding bin file (of course).
However, these are different from each frame of each bin file registered and then averaged.
Except for the very first one.
Why???
'''

#%%
nFrameList = []
for i, sn in enumerate(useSessionNames[:5]):
    sname = sn[4:]
    readFn = f'{h5planeDir}{sname}/plane0/data.bin'
    with BinaryFile(Ly=Ly, Lx=Lx, read_filename=readFn, write_filename=writeFn, append=True) as f:
        nFrameList.append(f.shape[0])
#%%
allMeanImg = []
stitchedDataFn = f'{s2pPlaneDir}data.bin'
numFrames = np.concatenate(([0],np.cumsum(np.array(nFrameList))))
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = stitchedDataFn) as f:
    data = f.data
    for i in range(5):
        allMeanImg.append(data[numFrames[i]:numFrames[i+1],:,:].mean(axis=0))
#%%
viewer = napari.Viewer()
for i in range(5):
    viewer.add_image(np.array([allMeanImg[i], opsMimgReg[i], stitchedOps['regImgs'][i], meanImgs[i], opsMimgReg[i]]), visible=False)