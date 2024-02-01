# -*- coding: utf-8 -*-
"""
Created on Fri May 28 06:51:06 2021

@author: jkim
"""
# import numpy as np
# baseDir = 'D:/TPM/JK/h5/'
# mice = [25,27,30,36,37,38,39,41,52,53,54,56]
# mi = 0
# mouse = mice[mi]
# pi = 6
# # planeDir = f'{baseDir}{mouse:03}/plane_{pi}/all_session/plane0/'
# params = range(21,30)
# numCells = np.zeros(np.shape(params))
# numNotcells = np.zeros(np.shape(params))
# for i, numParam in enumerate(params):
#     paramDir = f'{baseDir}{mouse:03}/plane_{pi}/all_session/plane0/param{numParam}/'
#     iscell = np.load(f'{paramDir}/iscell.npy')
#     # ops = np.load(f'{paramDir}/ops.npy', allow_pickle=True).item()
#     numCells[i] = np.sum(iscell[:,0])
#     numNotcells[i] = np.shape(iscell)[0]-numCells[i]
    
# #%%
# baseDir = 'D:/TPM/JK/h5/'
# mouse = 37
# planes = range(1,9)
# params = range(21,30)
# numCells = np.zeros(shape = (len(planes), len(params)))
# numNotcells = np.zeros(shape = (len(planes), len(params)))
# for i, pi in enumerate(planes):
#     for j, numParam in enumerate(params):
#         paramDir = f'{baseDir}{mouse:03}/plane_{pi}/all_session/plane0/param{numParam}/'
#         iscell = np.load(f'{paramDir}/iscell.npy')
#         # ops = np.load(f'{paramDir}/ops.npy', allow_pickle=True).item()
#         numCells[i,j] = np.sum(iscell[:,0])
#         numNotcells[i,j] = np.shape(iscell)[0]-numCells[i,j]
# #%%
# print([(p-20)/10 for p in params])

#%%
from suite2p.io.binary import BinaryFile
import matplotlib.pyplot as plt
import numpy as np
baseDir = 'D:/TPM/JK/h5/'
mice = [25,27,30,36,37,38,39,41,52,53,54,56]
mi = 0
mouse = mice[mi]
pi = 6

opsfn = f'{baseDir}{mouse:03}/plane_{pi}/all_session/plane0/ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()
Ly = ops['Ly']
Lx = ops['Lx']
perFileMeanImg = []

# Combining files from the same session
# Multiple sbx from same training sessions were already combined when transferred to h5
# Spontaneous and piezo sessions were not.
# For mi < 8: spontaneous sessions were saved in each h5 file
# For mi >= 8: spontaneous sessions were divided into 20 files starting from the second spont session
# For all mice: pizeo sessions were divided.
if mi < 8:
    slist = []
    snameList = []
    for f5name in ops['h5list']:
        fstr1 = f5name.split(f'plane_{pi}\\')
        fstr2 = fstr1[1].split('_plane_')[0].split('_')
        snameList.append(f'{fstr2[0]}_{fstr2[1]}_{fstr2[2]}')        
        if int(fstr2[1]) < 9000:
            slist.append(f'{fstr2[0]}_{fstr2[1]}_{fstr2[2]}')
        else: # in case of piezo sessions
            sstr = f'{fstr2[0]}_{fstr2[1]}_{fstr2[2][0]}'
            if sstr not in slist: # make session name list a unique list
                slist.append(sstr)
    h5ind = []
    nframes = []
    for sn in slist:
        tempInd = [i for i, fn in enumerate(snameList) if sn in fn]
        h5ind.append(tempInd)
        nframes.append(sum(ops['nframes_per_folder'][tempInd]))

#%%
# Some sessions are separated (piezo sessions)
# So, I need to explicitly list the frames for each file
# To gather them correctly later (into a single session)
cumsumNframes = np.insert(np.cumsum(ops['nframes_per_folder']),0,0)
frameInds = []
for i in range(len(cumsumNframes)-1):
    frameInds.append([*range(cumsumNframes[i],cumsumNframes[i+1])])

#%%
frameIndsPerSession = []
for i in range(len(h5ind)):
    if len(h5ind[i]) < 1:
        raise(f'No h5 index at {i}')
    elif len(h5ind[i]) == 1:
        frameIndsPerSession.append(frameInds[h5ind[i][0]])
    else:
        tempInds = []
        for j in range(len(h5ind[i])):
            tempInds = tempInds + [k for k in frameInds[h5ind[i][j]]]
        frameIndsPerSession.append(tempInds)


#%%
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = ops['reg_file']) as f:
    for inds in frameIndsPerSession:
        perFileMeanImg.append(f.ix(indices=inds).astype(np.float32).mean(axis=0))

#%%
f, ax = plt.subplots(2,1)
ax[0].imshow(perFileMeanImg[3], cmap = 'gray')
ax[0].axes.get_xaxis().set_visible(False)
ax[0].axes.get_yaxis().set_visible(False)
ax[1].imshow(perFileMeanImg[25], cmap = 'gray')
ax[1].axes.get_xaxis().set_visible(False)
ax[1].axes.get_yaxis().set_visible(False)

#%%
import cv2
temp = perFileMeanImg[3].round()
temp = (temp-np.amin(temp)) / (np.amax(temp) - np.amin(temp)) * (2**16-1)
cv2.imwrite('test.png', temp.astype(np.uint16))

#%%
# Run cellpose

#%%
fn = 'C:/Users/shires/Dropbox/Works/Projects/2020 Neural stretching in S1/Analysis/codes/test_seg.npy'
seg = np.load(fn, allow_pickle = True).item()

#%%
f,ax= plt.subplots()
ax.imshow(seg['flows'][1][0,:,:])

#%%
probImg = seg['flows'][1][0,:,:]
cv2.imwrite('testProb.png', probImg)

#%%
np.save('JK037_per_file_meanImg', perFileMeanImg)