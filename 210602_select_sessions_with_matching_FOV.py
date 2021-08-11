# -*- coding: utf-8 -*-
"""
Based on 210602_select_sessions.py (per_session_mimg.py created)
Upon plotting and visual inspection, select sessions with matched FOV
    from each imaged volume.
Record corresponding estimated depths from these selected sessions.
Save the result in /Data directory

2021/06/07 JK
"""
#%% Collect data from all mice and planes
# Plot phase correlation and mean image correlation after registration
# Show mean images from the low correlation values
# Select sessions that match in the whole volume (4 planes)
# Record estimated depth from that volume

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import napari
import ffmpeg
import cv2

# CLAHE each mean images
def clahe_each(img):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img)) * (2**16-1)
    newimg = exposure.equalize_adapthist(newimg.astype(np.uint16))
    return newimg

h5Dir = 'D:/TPM/JK/h5/'
mice = [25,27,30,36,37,38,39,41,52,53,54,56]
testSessionName = [[4,19], [3,10,15], [3,21], [1,17], [7], [2], [1,22,23], [3], [3,21,22,23,25], [3], [3], [3,4]] # 7-angle test sessions

#%% Plot 
mi = 1 # from 0 to 11
mouse = mice[mi]
plane = 1 # from 1 to 8
planeDir = f'{h5Dir}{mouse:03}/plane_{plane}/all_session/plane0/'

ops = np.load(f'{planeDir}ops.npy', allow_pickle = True).item()
cropFactor = 0.05 # 5% from the top, 5% from the bottom
xcrop = int(np.floor((ops['xrange'][1] - ops['xrange'][0])*cropFactor))
ycrop = int(np.floor((ops['yrange'][1] - ops['yrange'][0])*cropFactor))
xrange = [ops['xrange'][0]+xcrop, ops['xrange'][1]-xcrop]
yrange = [ops['yrange'][0]+ycrop, ops['yrange'][1]-ycrop]

persession = np.load(f'{planeDir}per_session_mimg.npy', allow_pickle = True).item()
mimgs = persession['mimgs']
numSession = len(mimgs)
newmimgs = np.zeros((numSession,yrange[1]-yrange[0],xrange[1]-xrange[0]))
for i in range(numSession):
    newmimgs[i,:,:] = clahe_each(mimgs[i][yrange[0]:yrange[1],xrange[0]:xrange[1]])

imgcorr = np.ones((numSession, numSession))
for i in range(numSession-1):
    img1 = newmimgs[i,:,:]
    for j in range(i+1, numSession):
        img2 = newmimgs[j,:,:]
        imgcorr[i,j] = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
        imgcorr[j,i] = imgcorr[i,j]

meanCorr = [(np.sum(imgcorr[i,:])-1)/(numSession-1)  for i in range(numSession)]

nrCorrxy = [np.mean(ops['corrXY1'][persession['frameIndsPerSession'][i]]) for i in range(len(persession['frameIndsPerSession']))]

testSessionInd = [i for i, sname in enumerate(persession['sessionList']) if int(sname.split('_')[1]) in testSessionName[mi]]

f, ax = plt.subplots()
ax.plot(meanCorr, 'bo-')
ax.plot(testSessionInd, [meanCorr[i] for i in testSessionInd], 'ro')
ax.xaxis.set_ticks(np.arange(0,numSession,3))
ax.set_ylabel('Mean image correlation')
ax.yaxis.label.set_color('b')
ax.tick_params(axis='y', colors='b')
axr = ax.twinx()
axr.plot(nrCorrxy, 'yo-')
axr.plot(testSessionInd, [nrCorrxy[i] for i in testSessionInd], 'ro')
axr.set_ylabel('Mean peak phase correlation with the reference image')
axr.yaxis.label.set_color('y')
axr.tick_params(axis='y', colors='y')
ax.set_xlabel('Session #')
ax.set_title(f'JK{mouse:03} plane {plane}')
f.tight_layout()

sortedNrInd = np.argsort(nrCorrxy) # Nr: non-rigid phase correaltion value
sortedIcInd = np.argsort(meanCorr) # Ic: image correlation
print('Low mean phase correlation:', sortedNrInd)
print('Low mean registered image correlation:', sortedIcInd)
print('Test session index:', testSessionInd)
#%% Visual inspection
inds = [2,7,19,12,18,11]
# inds = [2,17,9,6,10,8]
# inds = [3,23,14,13,7,20]
f3, ax3 = plt.subplots(2,3, figsize = (13.33,7.5))
viewer = napari.Viewer()
for i, ind in enumerate(inds):
    x = i%3
    y = i//3
    ax3[y,x].imshow(newmimgs[ind,:,:])
    ax3[y,x].get_xaxis().set_visible(False)
    ax3[y,x].get_yaxis().set_visible(False)
    sessionName = persession['sessionList'][ind]
    ax3[y,x].set_title(f'Session #{ind}\n{sessionName}\nMean image corr = {meanCorr[ind]:.3f}\nMean peak phase corr = {nrCorrxy[ind]:.3f}')
    
    viewer.add_image(newmimgs[ind,:,:], name=f'Session #{ind}')
f3.tight_layout()

#%% Visual inspection using napari
(
    ffmpeg
    .input('')
)


#%% Depth estimation
edepth = [256,256,256,254,256,255,254,256,253,255,247,255,255,254,250,250,247,248,253,252,251,247,253,252,253]
print(f'Mean = {np.mean(edepth):.2f}, Std = {np.std(edepth):.2f}')