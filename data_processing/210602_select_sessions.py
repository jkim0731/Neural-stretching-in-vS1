# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 08:58:15 2021

@author: jkim

2021/06/02
Purpose: Select the best matching sessions from all sessions
Method: From the all sessions-registered files, 
    take the mean images from each session
    and calculate correlations between mean images.
    
"""

#%% Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import math


#%% Test from an example - JK025 plane 6
h5Dir = 'D:/TPM/JK/h5/'
mouse = 25
plane = 6
planeDir = f'{h5Dir}{mouse:03}/plane_{plane}/all_session/plane0/'
ops = np.load(f'{planeDir}ops.npy', allow_pickle = True).item()
cropFactor = 0.05 # 5% from the top, 5% from the bottom
xcrop = int(np.floor((ops['xrange'][1] - ops['xrange'][0])*cropFactor))
ycrop = int(np.floor((ops['yrange'][1] - ops['yrange'][0])*cropFactor))
xrange = [ops['xrange'][0]+xcrop, ops['xrange'][1]-xcrop]
yrange = [ops['yrange'][0]+ycrop, ops['yrange'][1]-ycrop]

baseDir = 'C:\\Users\\shires\\Dropbox\\Works\\Projects\\2020 Neural stretching in S1\\Analysis\\codes\\'
perSession = np.load(f'{baseDir}JK025_plane6_per_session.npy', allow_pickle = True).item()
mimgs = perSession['mimgs']

# CLAHE each mean images
def clahe_each(img):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img)) * (2**16-1)
    newimg = exposure.equalize_adapthist(newimg.astype(np.uint16))
    return newimg

numSession = mimgs.shape[0]
newmimgs = np.zeros((numSession,yrange[1]-yrange[0],xrange[1]-xrange[0]))
for i in range(numSession):
    newmimgs[i,:,:] = clahe_each(mimgs[i,yrange[0]:yrange[1],xrange[0]:xrange[1]])

#%%
imgcorr = np.ones((numSession, numSession)) 
for i in range(numSession-1):
    img1 = newmimgs[i,:,:]
    for j in range(i+1, numSession):
        img2 = newmimgs[j,:,:]
        imgcorr[i,j] = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
        imgcorr[j,i] = imgcorr[i,j]

#%%
f, ax = plt.subplots()
im = ax.imshow(imgcorr)
ax.set_title('Correlation between sessions')
ax.set_xlabel('Session #')
ax.set_ylabel('Session #')
plt.colorbar(im)
#%%
f, ax = plt.subplots()
meanCorr = [(np.sum(imgcorr[i,:])-1)/(numSession-1)  for i in range(numSession)]
ax.plot(meanCorr)
ax.set_title('Average correlation')
ax.set_xlabel('Session #')
#%% Visual inspection
inds = [0,1,2,3,4,5]
f3, ax3 = plt.subplots(2,3, figsize = (13.33,7.5))
for i, ind in enumerate(inds):
    x = i%3
    y = i//3
    ax3[y,x].imshow(newmimgs[i,:,:])
    ax3[y,x].get_xaxis().set_visible(False)
    ax3[y,x].get_yaxis().set_visible(False)
    sessionName = perSession['sessionList'][ind]
    ax3[y,x].set_title(f'Session #{ind}\n{sessionName}\nMean corr = {meanCorr[ind]:.3f}')
f.tight_layout()

#%% Visual inspection
inds = [3,6,8,9,16,24]
f3, ax3 = plt.subplots(2,3, figsize = (13.33,7.5))
for i, ind in enumerate(inds):
    x = i%3
    y = i//3
    ax3[y,x].imshow(newmimgs[ind,:,:])
    ax3[y,x].get_xaxis().set_visible(False)
    ax3[y,x].get_yaxis().set_visible(False)
    sessionName = perSession['sessionList'][ind]
    ax3[y,x].set_title(f'Session #{ind}\n{sessionName}\nMean corr = {meanCorr[ind]:.3f}')
f.tight_layout()

#%% Visual inspection
inds = [3,6,17,21,25,26]
f3, ax3 = plt.subplots(2,3, figsize = (13.33,7.5))
for i, ind in enumerate(inds):
    x = i%3
    y = i//3
    ax3[y,x].imshow(newmimgs[ind,:,:])
    ax3[y,x].get_xaxis().set_visible(False)
    ax3[y,x].get_yaxis().set_visible(False)
    sessionName = perSession['sessionList'][ind]
    ax3[y,x].set_title(f'Session #{ind}\n{sessionName}\nMean corr = {meanCorr[ind]:.3f}')
f.tight_layout()

#%% Compare with depth estimation
# from './Data/Data curation_neural stretching.xlsx'
eDepth = [256,256,256,254,256,255,254,256,253,255,247,255,255,254,250,250,250,248,253,251,253,252,247,252,277,247,268]
f4, ax4 = plt.subplots()
ax4.plot(meanCorr, 'bo-')
ax4.set_ylabel('Mean correlation')
ax4.yaxis.label.set_color('b')
ax4.tick_params(axis='y', colors='b')
ax4r = ax4.twinx()
ax4r.plot(eDepth, 'yo-')
ax4r.set_ylabel(r'Estimated depth ($\mu$m)')
ax4r.yaxis.label.set_color('y')
ax4r.tick_params(axis='y', colors='y')
ax4.set_xlabel('Session #')

#%% Compare with estimated depth deviation
eDepthDev = [np.abs(ed - np.mean(eDepth)) for ed in eDepth]
f5, ax5 = plt.subplots(1,2)
ax5[0].plot(meanCorr, 'bo-')
ax5[0].set_ylabel('Mean correlation')
ax5[0].yaxis.label.set_color('b')
ax5[0].tick_params(axis='y', colors='b')
ax5r = ax5[0].twinx()
ax5r.plot(eDepthDev, 'yo-')
ax5r.set_ylabel('Estimated depth deviation\n(absolute difference from the mean; $\mu$m)')
ax5r.yaxis.label.set_color('y')
ax5r.tick_params(axis='y', colors='y')
ax5[0].set_xlabel('Session #')

corrVal = np.corrcoef(meanCorr, eDepthDev)[0,1]
ax5[1].plot(meanCorr, eDepthDev, 'k.')
ax5[1].set_xlabel('Mean correlation')
ax5[1].set_ylabel(r'Estimated depth deviation ($\mu$m)')
ax5[1].set_title(f'r = {corrVal:.3f}')

f5.tight_layout()

#%% Compare with registration evaluation
# corrXY (peak of phase correlation between frame and reference image at each timepoint)
# corrxy = [np.mean(ops['corrXY'][perSession['frameIndsPerSession'][i]]) for i in range(len(perSession['frameIndsPerSession']))]
nrCorrxy = [np.mean(ops['corrXY1'][perSession['frameIndsPerSession'][i]]) for i in range(len(perSession['frameIndsPerSession']))]
f6, ax6 = plt.subplots(1,2)
ax6[0].plot(meanCorr, 'bo-')
ax6[0].set_ylabel('Mean image correlation')
ax6[0].yaxis.label.set_color('b')
ax6[0].tick_params(axis='y', colors='b')
ax6r = ax6[0].twinx()
ax6r.plot(nrCorrxy, 'yo-')
ax6r.set_ylabel('Mean peak phase correlation with the reference image')
ax6r.yaxis.label.set_color('y')
ax6r.tick_params(axis='y', colors='y')
ax6[0].set_xlabel('Session #')

corrVal = np.corrcoef(meanCorr, nrCorrxy)[0,1]
ax6[1].plot(meanCorr, nrCorrxy, 'k.')
ax6[1].set_xlabel('Mean image correlation')
ax6[1].set_ylabel('Mean peak phase correlation with the reference image')
ax6[1].set_title(f'r = {corrVal:.3f}')

f6.tight_layout()

#%% Visual inspection
inds = [3,6,11,16,24,26]
f3, ax3 = plt.subplots(2,3, figsize = (13.33,7.5))
for i, ind in enumerate(inds):
    x = i%3
    y = i//3
    ax3[y,x].imshow(newmimgs[ind,:,:])
    ax3[y,x].get_xaxis().set_visible(False)
    ax3[y,x].get_yaxis().set_visible(False)
    sessionName = perSession['sessionList'][ind]
    ax3[y,x].set_title(f'Session #{ind}\n{sessionName}\nMean image corr = {meanCorr[ind]:.3f}\nMean peak phase corr = {nrCorrxy[ind]:.3f}')
f3.tight_layout()


#%% From plane 5 (same volume)

from per_session_mimg import per_session_mimg
import time
h5Dir = 'D:/TPM/JK/h5/'
mouse = 25
plane = 5
planeDir = f'{h5Dir}{mouse:03}/plane_{plane}/all_session/plane0/'
tic = time.perf_counter()
per_session_mimg(planeDir, mouse, plane)
toc = time.perf_counter()
print(f'Took {np.round((toc-tic)/60)} minutes.')

#%%
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

f, ax = plt.subplots()
im = ax.imshow(imgcorr)
ax.set_title('Correlation between sessions')
ax.set_xlabel('Session #')
ax.set_ylabel('Session #')
plt.colorbar(im)
f, ax = plt.subplots()
meanCorr = [(np.sum(imgcorr[i,:])-1)/(numSession-1)  for i in range(numSession)]
ax.plot(meanCorr)
ax.set_title('Average correlation')
ax.set_xlabel('Session #')

#%%
eDepth = [256,256,256,254,256,255,254,256,253,255,247,255,255,254,250,250,250,248,253,251,253,252,247,252,277,247,268]
eDepthDev = [np.abs(ed - np.mean(eDepth)) for ed in eDepth]
# corrxy = [np.mean(ops['corrXY'][perSession['frameIndsPerSession'][i]]) for i in range(len(perSession['frameIndsPerSession']))]
nrCorrxy = [np.mean(ops['corrXY1'][perSession['frameIndsPerSession'][i]]) for i in range(len(perSession['frameIndsPerSession']))]

f5, ax5 = plt.subplots(1,2)
ax5[0].plot(meanCorr, 'bo-')
ax5[0].set_ylabel('Mean correlation')
ax5[0].yaxis.label.set_color('b')
ax5[0].tick_params(axis='y', colors='b')
ax5r = ax5[0].twinx()
ax5r.plot(eDepthDev, 'yo-')
ax5r.set_ylabel('Estimated depth deviation\n(absolute difference from the mean; $\mu$m)')
ax5r.yaxis.label.set_color('y')
ax5r.tick_params(axis='y', colors='y')
ax5[0].set_xlabel('Session #')

corrVal = np.corrcoef(meanCorr, eDepthDev)[0,1]
ax5[1].plot(meanCorr, eDepthDev, 'k.')
ax5[1].set_xlabel('Mean correlation')
ax5[1].set_ylabel(r'Estimated depth deviation ($\mu$m)')
ax5[1].set_title(f'r = {corrVal:.3f}')

f5.tight_layout()

# corrxy = [np.mean(ops['corrXY'][perSession['frameIndsPerSession'][i]]) for i in range(len(perSession['frameIndsPerSession']))]
# f6, ax6 = plt.subplots(1,2)
# ax6[0].plot(meanCorr, 'bo-')
# ax6[0].set_ylabel('Mean image correlation')
# ax6[0].yaxis.label.set_color('b')
# ax6[0].tick_params(axis='y', colors='b')
# ax6r = ax6[0].twinx()
# ax6r.plot(corrxy, 'yo-')
# ax6r.set_ylabel('Mean peak phase correlation with the reference image')
# ax6r.yaxis.label.set_color('y')
# ax6r.tick_params(axis='y', colors='y')
# ax6[0].set_xlabel('Session #')

# corrVal = np.corrcoef(meanCorr, corrxy)[0,1]
# ax6[1].plot(meanCorr, corrxy, 'k.')
# ax6[1].set_xlabel('Mean image correlation')
# ax6[1].set_ylabel('Mean peak phase correlation with the reference image')
# ax6[1].set_title(f'r = {corrVal:.3f}')

# f6.tight_layout()

f7, ax7 = plt.subplots(1,2)
ax7[0].plot(meanCorr, 'bo-')
ax7[0].set_ylabel('Mean image correlation')
ax7[0].yaxis.label.set_color('b')
ax7[0].tick_params(axis='y', colors='b')
ax7r = ax7[0].twinx()
ax7r.plot(nrCorrxy, 'yo-')
ax7r.set_ylabel('Mean peak phase correlation with the reference image')
ax7r.yaxis.label.set_color('y')
ax7r.tick_params(axis='y', colors='y')
ax7[0].set_xlabel('Session #')

corrVal = np.corrcoef(meanCorr, nrCorrxy)[0,1]
ax7[1].plot(meanCorr, nrCorrxy, 'k.')
ax7[1].set_xlabel('Mean image correlation')
ax7[1].set_ylabel('Mean peak phase correlation with the reference image')
ax7[1].set_title(f'r = {corrVal:.3f}')

f7.tight_layout()



#%% Visual inspection
inds = [3,6,8,16,24,26]
f3, ax3 = plt.subplots(2,3, figsize = (13.33,7.5))
for i, ind in enumerate(inds):
    x = i%3
    y = i//3
    ax3[y,x].imshow(newmimgs[ind,:,:])
    ax3[y,x].get_xaxis().set_visible(False)
    ax3[y,x].get_yaxis().set_visible(False)
    sessionName = perSession['sessionList'][ind]
    ax3[y,x].set_title(f'Session #{ind}\n{sessionName}\nMean image corr = {meanCorr[ind]:.3f}\nMean peak phase corr = {nrCorrxy[ind]:.3f}')
f3.tight_layout()

#%% Visual inspection
inds = [0,3,9,11,14,25]
f3, ax3 = plt.subplots(2,3, figsize = (13.33,7.5))
for i, ind in enumerate(inds):
    x = i%3
    y = i//3
    ax3[y,x].imshow(newmimgs[ind,:,:])
    ax3[y,x].get_xaxis().set_visible(False)
    ax3[y,x].get_yaxis().set_visible(False)
    sessionName = perSession['sessionList'][ind]
    ax3[y,x].set_title(f'Session #{ind}\n{sessionName}\nMean image corr = {meanCorr[ind]:.3f}\nMean peak phase corr = {nrCorrxy[ind]:.3f}')
f3.tight_layout()








#%% Run other sessions
from per_session_mimg import per_session_mimg
import time
import numpy as np
# h5Dir = 'D:/TPM/JK/h5/'
# # mouse = 25
# # for plane in [1,2,3,4,7,8]:
# #     planeDir = f'{h5Dir}{mouse:03}/plane_{plane}/all_session/plane0/'
# #     tic = time.perf_counter()
# #     per_session_mimg(planeDir, mouse, plane)
# #     toc = time.perf_counter()
# #     print(f'JK{mouse:03} plane {plane}: {np.round((toc-tic)/60)} minutes.')
# for mouse in [30,36,37,41]:
#     for plane in range(1,9):
#         planeDir = f'{h5Dir}{mouse:03}/plane_{plane}/all_session/plane0/'
#         tic = time.perf_counter()
#         per_session_mimg(planeDir, mouse, plane)
#         toc = time.perf_counter()
#         print(f'JK{mouse:03} plane {plane}: {np.round((toc-tic)/60)} minutes.')


# h5Dir = 'J:/'
# for mouse in [52,53,54,56]:
#     for plane in range(1,9):
#         planeDir = f'{h5Dir}{mouse:03}/plane_{plane}/all_session/plane0/'
#         tic = time.perf_counter()
#         per_session_mimg(planeDir, mouse, plane)
#         toc = time.perf_counter()
#         print(f'JK{mouse:03} plane {plane}: {np.round((toc-tic)/60)} minutes.')
        
# h5Dir = 'I:/'
# mouse= 38
# for plane in range(1,9):
#     planeDir = f'{h5Dir}{mouse:03}/plane_{plane}/all_session/plane0/'
#     tic = time.perf_counter()
#     per_session_mimg(planeDir, mouse, plane)
#     toc = time.perf_counter()
#     print(f'JK{mouse:03} plane {plane}: {np.round((toc-tic)/60)} minutes.')
    
# h5Dir = 'H:/'
# mouse = 27
# for plane in range(1,9):
#     planeDir = f'{h5Dir}{mouse:03}/plane_{plane}/all_session/plane0/'
#     tic = time.perf_counter()
#     per_session_mimg(planeDir, mouse, plane)
#     toc = time.perf_counter()
#     print(f'JK{mouse:03} plane {plane}: {np.round((toc-tic)/60)} minutes.')

h5Dir = 'D:/TPM/JK/h5/'
mouse= 25
for plane in [6]:
    planeDir = f'{h5Dir}{mouse:03}/plane_{plane}/all_session/plane0/'
    tic = time.perf_counter()
    per_session_mimg(planeDir, mouse, plane)
    toc = time.perf_counter()
    print(f'JK{mouse:03} plane {plane}: {np.round((toc-tic)/60)} minutes.')
    
