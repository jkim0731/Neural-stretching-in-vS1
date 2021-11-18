"""
Find the best way to match between reference image and z-stack image.
From 211012_zstack_depth_estimation.py

This time, use a single plane (from averaged z-stack), and first try find out
the best registration methods (with the reference mean image)

For JK027 plane 4
Found out best-matching depth by eye (~145)
First, to include all matching FOV, register z-stack from top-down (not bottom-up like before)
Then, try best method to match 11-plane averaged z-stack image with the reference mean image
TurboReg, simpleITK, suite2p, using raw, clahe (adjust kernel), and cellpose image

2021/10/18 JK
"""

#%% BS
import scipy.io
import numpy as np
import pandas as pd
import napari
import os, glob
from skimage import exposure
from suite2p.io.binary import BinaryFile
import matplotlib.pyplot as plt
from scipy import interpolate
import time

from pystackreg import StackReg
# import SimpleITK as sitk

import gc
gc.enable()

def clahe_each(img: np.float64, kernel_size = None, clip_limit = 0.01, nbins = 2**16):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    newimg = exposure.equalize_adapthist(newimg, kernel_size = kernel_size, clip_limit = clip_limit, nbins=nbins)    
    return newimg

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

def get_session_names(planeDir, mouse, planeNum):
    tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{planeNum}.h5')
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

def img_norm(img):
    if len(img.shape) == 2:
        return (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    elif len(img.shape) == 3:
        rimg = np.zeros(img.shape)
        for i in range(img.shape[0]):
            rimg[i,:,:] = (img[i,:,:] - np.amin(img[i,:,:])) / (np.amax(img[i,:,:]) - np.amin(img[i,:,:]))
        return rimg
    else:
        raise('Input image should have either 2 or 3 dims')

h5Dir = 'D:/TPM/JK/h5/'
zstackDir = 'D:/TPM/JK/h5/zstacks/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]




# #%% Sample preparation
# #%% Select mouse and plane
# mi = 3
# mouse = mice[mi]
# pn = 1
# refSession = refSessions[mi]

# # estDepthInd = 170 # from the top (reverse order of .mat file)
# sessionNames = get_session_names(f'{h5Dir}{mouse:03}/plane_{pn}/', mouse, pn)

# #%% load non-registered mean images
# sessionNames = get_session_names(f'{h5Dir}{mouse:03}/plane_{pn}/', mouse, pn)
# trainingSessionNames = [sn[4:] for sn in sessionNames if len(sn)==7]
# meanImgList = []
# for sn in trainingSessionNames:
#     ops = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/{sn}/plane0/ops.npy', allow_pickle=True).item()
#     meanImgList.append(ops['meanImg'])

# napari.view_image(np.array(meanImgList))

# #%% Trim top ringing portion
# topRingingPix = 80
# meanImgTrimList = [mimg[topRingingPix:,:] for mimg in meanImgList]

# napari.view_image(np.array(meanImgTrimList))

# #%% Register mean images using StackReg

# srRigid = StackReg(StackReg.RIGID_BODY)

# mimgRegList = []
# fixed = meanImgTrimList[0].copy()
# for img in meanImgTrimList:
#     moving = img.copy()
#     moved = srRigid.register_transform(fixed, moving)
#     mimgRegList.append(moved)
# napari.view_image(np.array(mimgRegList))



# #%% Load z-stack and register using StackReg, in reverse order
# srRigid = StackReg(StackReg.RIGID_BODY)

# zregFnList = glob.glob(f'{zstackDir}zstack_{mouse:03}_*.mat')
# zregFn = zregFnList[0]
# mat = scipy.io.loadmat(zregFn)

# zstack = np.moveaxis(mat['zstack'], -1, 0)
# zstack = np.flip(zstack, axis=0)
# nplane = zstack.shape[0]

# zstackReg = srRigid.register_transform_stack(zstack, axis=0, reference='previous')

# napari.view_image(zstackReg)

# #%% Average z-stack
# avgNumPlanes = 11
# zstackAvg = np.zeros(zstackReg.shape)
# nplanes = zstackReg.shape[0]
# for i in range(nplanes):
#     startPlaneNum = max(0, i - avgNumPlanes//2)
#     endPlaneNum = min(nplanes, i + avgNumPlanes//2)
#     zstackAvg[i,:,:] = zstackReg[startPlaneNum:endPlaneNum, :, :].mean(axis=0)

# napari.view_image(zstackAvg)

# #%% Select the best depth
# bestDepthInd = 170

# refImg = zstackAvg[bestDepthInd,:,:]



# #%%
# # zstackClahe = np.zeros(zstack.shape)
# # kernelFrac = 1/3
# # Ly = zstack.shape[1]
# # Lx = zstack.shape[2]
# # kernely = int(Ly * kernelFrac)
# # kernelx = int(Lx * kernelFrac)
# # for i in range(nplane):
# #     zstackClahe[i,:,:] = clahe_each(zstack[i,:,:], kernel_size = (kernely, kernelx))

# # zstackClaheRegTmat = srRigid.register_stack(zstackClahe, axis=0, reference = 'previous')

# # zstackClaheReg = srRigid.transform_stack(zstack, tmats = zstackClaheRegTmat)
# # #%%
# # avgNumPlanes = 11
# # zstackClaheAvg = np.zeros(zstackClaheReg.shape)
# # nplanes = zstackClaheReg.shape[0]
# # for i in range(nplanes):
# #     startPlaneNum = max(0, i - avgNumPlanes//2)
# #     endPlaneNum = min(nplanes, i + avgNumPlanes//2)
# #     zstackClaheAvg[i,:,:] = zstackClaheReg[startPlaneNum:endPlaneNum, :, :].mean(axis=0)

# # napari.view_image(zstackClaheAvg)

# '''
# Making Clahe before z-stack registration does NOT help at all. 
# '''








# #%% Registration
# #%% Reg-(1) Using StackReg
# refSessionInd = [i for i, sn in enumerate(sessionNames) if int(sn.split('_')[1]) == refSession][0]
# movImg = mimgRegList[refSessionInd]

# refImgStackReg = refImg[187:187+movImg.shape[0], 96:96+movImg.shape[1]]

# srScaled = StackReg(StackReg.SCALED_ROTATION)
# srAffine = StackReg(StackReg.AFFINE)
# srBilinear = StackReg(StackReg.BILINEAR)


# outRigid = srRigid.register_transform(refImgStackReg, movImg)
# outScaled = srScaled.register_transform(refImgStackReg, movImg)
# outAffine = srAffine.register_transform(refImgStackReg, movImg)
# outBilinear = srBilinear.register_transform(refImgStackReg, movImg)


# viewer = napari.Viewer()
# refImgRGB = np.moveaxis(img_norm(np.tile(refImgStackReg, (3,1,1))), 0, -1)
# outRigidRGB = np.moveaxis(img_norm(np.tile(outRigid, (3,1,1))), 0, -1)
# blendedRigid = imblend(img_norm(refImgStackReg), img_norm(outRigid))

# outScaledRGB = np.moveaxis(img_norm(np.tile(outScaled, (3,1,1))), 0, -1)
# blendedScaled = imblend(img_norm(refImgStackReg), img_norm(outScaled))

# outAffineRGB = np.moveaxis(img_norm(np.tile(outAffine, (3,1,1))), 0, -1)
# blendedAffine = imblend(img_norm(refImgStackReg), img_norm(outAffine))

# outBilinearRGB = np.moveaxis(img_norm(np.tile(outBilinear, (3,1,1))), 0, -1)
# blendedBilinear = imblend(img_norm(refImgStackReg), img_norm(outBilinear))

# viewer.add_image(np.array([refImgRGB, outRigidRGB, blendedRigid]), rgb='True', name='rigid')
# viewer.add_image(np.array([refImgRGB, outScaledRGB, blendedScaled]), rgb='True', name='scaled_rotation')
# viewer.add_image(np.array([refImgRGB, outAffineRGB, blendedAffine]), rgb='True', name='affine')
# viewer.add_image(np.array([refImgRGB, outBilinearRGB, blendedBilinear]), rgb='True', name='bilinear')



# #%% Reg-(1) Using StackReg, Using clahe
# kernelFrac = 1/3
# Ly = zstack.shape[1]
# Lx = zstack.shape[2]
# kernely = int(Ly * kernelFrac)
# kernelx = int(Lx * kernelFrac)

# movImg = mimgRegList[refSessionInd]
# movImgClahe = clahe_each(movImg, kernel_size = (kernely, kernelx))
# refImgStackReg = refImg[187:187+movImg.shape[0], 96:96+movImg.shape[1]]
# refImgClahe = clahe_each(refImgStackReg, kernel_size = (kernely, kernelx))

# srScaled = StackReg(StackReg.SCALED_ROTATION)
# srAffine = StackReg(StackReg.AFFINE)
# srBilinear = StackReg(StackReg.BILINEAR)


# tmatRigid = srRigid.register(refImgClahe, movImgClahe)
# outRigid = srRigid.transform(movImg, tmatRigid)
# tmatScaled = srScaled.register(refImgClahe, movImgClahe)
# outScaled = srScaled.transform(movImg, tmatScaled)
# tmatAffine = srAffine.register(refImgClahe, movImgClahe)
# outAffine = srAffine.transform(movImg, tmatAffine)
# tmatBilinear = srBilinear.register(refImgClahe, movImgClahe)
# outBilinear = srBilinear.transform(movImg, tmatBilinear)



# viewer = napari.Viewer()
# refImgRGB = np.moveaxis(img_norm(np.tile(refImgStackReg, (3,1,1))), 0, -1)
# outRigidRGB = np.moveaxis(img_norm(np.tile(outRigid, (3,1,1))), 0, -1)
# blendedRigid = imblend(img_norm(refImgStackReg), img_norm(outRigid))

# outScaledRGB = np.moveaxis(img_norm(np.tile(outScaled, (3,1,1))), 0, -1)
# blendedScaled = imblend(img_norm(refImgStackReg), img_norm(outScaled))

# outAffineRGB = np.moveaxis(img_norm(np.tile(outAffine, (3,1,1))), 0, -1)
# blendedAffine = imblend(img_norm(refImgStackReg), img_norm(outAffine))

# outBilinearRGB = np.moveaxis(img_norm(np.tile(outBilinear, (3,1,1))), 0, -1)
# blendedBilinear = imblend(img_norm(refImgStackReg), img_norm(outBilinear))

# viewer.add_image(np.array([refImgRGB, outRigidRGB, blendedRigid]), rgb='True', name='rigid')
# viewer.add_image(np.array([refImgRGB, outScaledRGB, blendedScaled]), rgb='True', name='scaled_rotation')
# viewer.add_image(np.array([refImgRGB, outAffineRGB, blendedAffine]), rgb='True', name='affine')
# viewer.add_image(np.array([refImgRGB, outBilinearRGB, blendedBilinear]), rgb='True', name='bilinear')


# '''
# Bilinear works quite well.
# Now, test depth estimation using pixel correlation.
# '''

# #%% Depth estimation using bilinear registration
# # with raw zstack or averaged zstack

# pixCorr = np.zeros((nplanes,2)) # [:,0] raw zstack, [:,1] avg zstack
# for i in range(nplanes):
#     pixCorr[i,0] = np.corrcoef(outBilinear.flatten(), zstackReg[i,187:187+movImg.shape[0], 96:96+movImg.shape[1]].flatten())[0,1]
#     pixCorr[i,1] = np.corrcoef(outBilinear.flatten(), zstackAvg[i,187:187+movImg.shape[0], 96:96+movImg.shape[1]].flatten())[0,1]

# fig, ax = plt.subplots()
# ax.plot(pixCorr[:,0])
# ax.plot(pixCorr[:,1])

# '''
# Pix correlation does not match with visual inspection.
# '''


# #%% Depth estimation using bilinear registration
# # with raw zstack or averaged zstack, contrast-adjusted using clahe
# kernelFrac = 1/10
# Ly = zstack.shape[1]
# Lx = zstack.shape[2]
# kernely = int(Ly * kernelFrac)
# kernelx = int(Lx * kernelFrac)
# outClahe = clahe_each(outBilinear, kernel_size = (kernely, kernelx))

# pixCorrClahe = np.zeros((nplanes,2)) # [:,0] raw zstack, [:,1] avg zstack
# for i in range(nplanes):
#     regClahe = clahe_each(zstackReg[i,187:187+movImg.shape[0], 96:96+movImg.shape[1]], kernel_size = (kernely, kernelx))
#     pixCorrClahe[i,0] = np.corrcoef(outClahe.flatten(), regClahe.flatten())[0,1]
#     avgClahe = clahe_each(zstackAvg[i,187:187+movImg.shape[0], 96:96+movImg.shape[1]], kernel_size = (kernely, kernelx))
#     pixCorrClahe[i,1] = np.corrcoef(outClahe.flatten(), avgClahe.flatten())[0,1]

# fig, ax = plt.subplots()
# ax.plot(pixCorrClahe[:,0])
# ax.plot(pixCorrClahe[:,1])

# '''
# Clahe does not help. 
# Main mismatch is sustained increased correlation after the estimated depth, 170.
# Peaks at ~220 and decreases a little bit. Before the peak it is monotonically increasing.
# Why do lower planes show high correlation values?

# Reducing kernel fraction improves this. 1/10 shows single peak.

# Let's ses if this can distinguish between seeminly different planes
# '''

# #%% Depth estimation from two different mean images
# kernelFrac = 1/10
# Ly = zstack.shape[1]
# Lx = zstack.shape[2]
# kernely = int(Ly * kernelFrac)
# kernelx = int(Lx * kernelFrac)

# mimgList = mimgRegList[2:4]
# # For now, since these two are aligned, just use the same transformation matrix.
# outList = []
# outList.append(clahe_each(srBilinear.transform(mimgList[0], tmatBilinear), kernel_size = (kernely, kernelx)))
# outList.append(clahe_each(srBilinear.transform(mimgList[1], tmatBilinear), kernel_size = (kernely, kernelx)))

# # Show the results grouped by if averaged.
# pixCorrRaw = np.zeros((nplanes,2))
# pixCorrAvg = np.zeros((nplanes,2))
# for i in range(nplanes):
#     rawClahe = clahe_each(zstackReg[i,187:187+movImg.shape[0], 96:96+movImg.shape[1]], kernel_size = (kernely, kernelx))
#     avgClahe = clahe_each(zstackAvg[i,187:187+movImg.shape[0], 96:96+movImg.shape[1]], kernel_size = (kernely, kernelx))

#     pixCorrRaw[i,0] = np.corrcoef(outList[0].flatten(), rawClahe.flatten())[0,1]
#     pixCorrRaw[i,1] = np.corrcoef(outList[1].flatten(), rawClahe.flatten())[0,1]

#     pixCorrAvg[i,0] = np.corrcoef(outList[0].flatten(), avgClahe.flatten())[0,1]
#     pixCorrAvg[i,1] = np.corrcoef(outList[1].flatten(), avgClahe.flatten())[0,1]
# #%%
# fig, ax = plt.subplots(1,2)
# ax[0].plot(pixCorrRaw[:,0], label='session 3')
# ax[0].plot(pixCorrRaw[:,1], label='session 4')
# ax[0].legend()
# ax[0].set_xlabel('Z-stack plane #')
# ax[0].set_ylabel('Pixel intensity')
# ax[0].set_title('Raw zstack clahe with 1/10 kernel')

# ax[1].plot(pixCorrAvg[:,0])
# ax[1].plot(pixCorrAvg[:,1])
# ax[1].set_title('Averaged zstack clahe with 1/10 kernel')

# fig.suptitle(f'JK{mouse:03} plane {pn}')
# fig.tight_layout()

# '''
# 8-10 um difference
# Now, try this from single session z-drift.

# Beginning to record in 211018 Z-drift during imaging.pptx
# '''



# #%% Example single session z-drift
# #%% Read session
# sname = '001'

# pn = 4
# topMargin = 80
# bottomMargin = 15
# leftMargin = 10
# rightMargin = 10

# division = 6
# sessionOps = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/{sname}/plane0/ops.npy', allow_pickle=True).item()
# Ly = sessionOps['Ly']
# Lx = sessionOps['Lx']
# nframes = sessionOps['nframes']
# sessionImgsDiv = []
# with BinaryFile(Ly, Lx, read_filename=f'{h5Dir}{mouse:03}/plane_{pn}/{sname}/plane0/data.bin') as f:
#     data = f.data
#     for i in range(division):
#         tempStartFrame = (i*nframes) // division
#         tempEndFrame = ((i+1)*nframes) // division
#         sessionImgsDiv.append(data[tempStartFrame:tempEndFrame,topMargin:-bottomMargin,leftMargin:-rightMargin].mean(axis=0))
# napari.view_image(np.array(sessionImgsDiv))

# mimg = sessionOps['meanImg'][topMargin:-bottomMargin,leftMargin:-rightMargin]
# napari.view_image(mimg)

# #%% Check registration
# kernelFrac = 1/10
# Ly = mimg.shape[0]
# Lx = mimg.shape[1]
# kernely = int(Ly * kernelFrac)
# kernelx = int(Lx * kernelFrac)

# estDepthInd = 172
# estYpoint = 195
# estXpoint = 105
# zref = zstackAvg[estDepthInd, estYpoint:estYpoint+mimg.shape[0], estXpoint:estXpoint+mimg.shape[1]]
# zrefClahe = clahe_each(zref, kernel_size = (kernely, kernelx))
# mimgClahe = clahe_each(mimg, kernel_size = (kernely, kernelx))

# srBilinear = StackReg(StackReg.BILINEAR)
# tmatBilinear = srBilinear.register(zrefClahe, mimgClahe)
# outBilinear = srBilinear.transform(mimg, tmatBilinear)

# refImgRGB = np.moveaxis(img_norm(np.tile(zref, (3,1,1))), 0, -1)
# outBiRGB = np.moveaxis(img_norm(np.tile(outBilinear, (3,1,1))), 0, -1)
# blendedBi = imblend(img_norm(zref), img_norm(outBilinear))

# napari.view_image(np.array([refImgRGB, outBiRGB, blendedBi]))

# #%% Apply to image bins
# kernelFrac = 1/10
# Ly = mimg.shape[0]
# Lx = mimg.shape[1]
# kernely = int(Ly * kernelFrac)
# kernelx = int(Lx * kernelFrac)

# outList = []
# for i in range(division):
#     outList.append(clahe_each(srBilinear.transform(sessionImgsDiv[i], tmatBilinear), kernel_size = (kernely, kernelx)))

# # Show the results grouped by if averaged.
# pixCorrRaw = np.zeros((nplanes,division))
# pixCorrAvg = np.zeros((nplanes,division))
# for pi in range(nplanes):
#     rawClahe = clahe_each(zstackReg[pi,estYpoint:estYpoint+mimg.shape[0], estXpoint:estXpoint+mimg.shape[1]],
#                           kernel_size = (kernely, kernelx))
#     avgClahe = clahe_each(zstackAvg[pi,estYpoint:estYpoint+mimg.shape[0], estXpoint:estXpoint+mimg.shape[1]],
#                           kernel_size = (kernely, kernelx))

#     for di in range(division):
#         pixCorrRaw[pi,di] = np.corrcoef(outList[di].flatten(), rawClahe.flatten())[0,1]
#         pixCorrAvg[pi,di] = np.corrcoef(outList[di].flatten(), avgClahe.flatten())[0,1]

# fig, ax = plt.subplots(1,2)
# for di in range(division):
#     ax[0].plot(pixCorrRaw[:,di], label=f'bin {di}')
# ax[0].legend()
# ax[0].set_xlabel('Z-stack plane #')
# ax[0].set_ylabel('Pixel intensity')
# ax[0].set_title('Raw zstack clahe with 1/10 kernel')

# for di in range(division):
#     ax[1].plot(pixCorrAvg[:,di])
# ax[1].set_title('Averaged zstack clahe with 1/10 kernel')

# fig.suptitle(f'JK{mouse:03} session {sname} plane {pn}')
# fig.tight_layout()

# #%% Normalized correlation values to compare peaks
# fig, ax = plt.subplots(1,2)
# for di in range(division):
#     ax[0].plot(pixCorrRaw[:,di]/np.amax(pixCorrRaw[:,di]), label=f'bin {di}')
# ax[0].legend()
# ax[0].set_xlabel('Z-stack plane #')
# ax[0].set_ylabel('Normalized pixel intensity')
# ax[0].set_title('Raw zstack clahe with 1/10 kernel')

# for di in range(division):
#     ax[1].plot(pixCorrAvg[:,di]/np.amax(pixCorrAvg[:,di]))
# ax[1].set_title('Averaged zstack clahe with 1/10 kernel')

# fig.suptitle(f'JK{mouse:03} session {sname} plane {pn}')
# fig.tight_layout()




# #%%



# '''
# Now that I've found out that StackReg bilinear with clahe (with kernel size adjustment and averaging) works,
# let's look at the pattern and ses which parameters affect the most, at least in this single example.
# Then, test in other planes.
# Then, test in other mice.
# Quantify the mode of direction (unidirectional? upward or downward?), and average drift velocity.


# Parameters to test:
#     (1) CLAHE kernel size (fraction) for registration & pixel correlation calculation
#     (2) Average # of zstack planes.
#         Does estimated depth affect a lot?
    
# '''

# #%% (1) CLAHE kernel size (fraction) for registration & pixel correlation calculation
# # Take a single example from sessionImgsDiv. Test with the rest later.
# # Test both registration and


# #%% (2) Average # of zstack planes.
# # Calculate pixel correlations after matching to each planes around the estimated depth.
# # Compare this with applying the same transformation to the estimated depth to other surrounding depths.
# # Test 20 planes up and down (total 41 planes) from the estimated depth.
# # Test averaging 3, 5, 11, and 21
# # Using the first quarter mean image of the example reference (JK027 session 3 plane 4)
# # Estimated depth, x, and y points are selected above

# testImg = sessionImgsDiv[0]
# Ly = testImg.shape[0]
# Lx = testImg.shape[1]
# estDepthInd = 172
# estYpoint = 195
# estXpoint = 105

# srBi = StackReg(StackReg.BILINEAR)

# numTestPlane = 20 # up and down, total numTestPlane*2+1 planes
# numTotalPlane = numTestPlane*2+1

# # List of testants
# regKernelFracList = [1/i for i in range(3,9)]
# numRK = len(regKernelFracList)
# corrKernelFracList = [1/i for i in range(3,9)]
# numCK = len(corrKernelFracList)
# numAvgList = [3,5,11,21]
# numNA = len(numAvgList)
# corrEachReg = np.zeros((numRK, numCK, numNA, numTotalPlane))

# regImgAllList = []
# time0 = time.time()
# for regKi in range(numRK):
#     regKernelFrac = regKernelFracList[regKi]
#     regKernelSize = (int(Ly*regKernelFrac), int(Lx*regKernelFrac))
#     testImgReg = clahe_each(testImg, kernel_size = regKernelSize)
#     for corrKi in range(numCK):
#         corrKernelFrac = corrKernelFracList[corrKi]
#         corrKernelSize = (int(Ly*corrKernelFrac), int(Lx*corrKernelFrac))
#         corrImgReg = clahe_each(testImg, kernel_size = corrKernelSize)
#         for avgNi in range(numNA):
#             time1 = time.time()
#             print(f'Running regKernel {regKi}/{numRK-1}, corrKernel {corrKi}/{numCK-1}, {avgNi}/{numNA-1}')
#             numAvg = numAvgList[avgNi]
            
#             zstackAvgTmp = np.zeros((numTotalPlane,zstackReg.shape[1],zstackReg.shape[2]))
#             nplanes = zstackReg.shape[0]
#             for i in range(numTotalPlane):
#                 startPlaneNum = estDepthInd - numTestPlane -(numAvg//2) + i
#                 endPlaneNum = startPlaneNum + numAvg
#                 zstackAvgTmp[i,:,:] = zstackReg[startPlaneNum:endPlaneNum, :, :].mean(axis=0)
            
#             # corrSingleReg = np.zeros(numTotalPlane)
#             # refImgReg = clahe_each(zstackAvgTmp[numTestPlane, estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx],
#             #                            kernel_size = regKernelSize)
#             # tformSingle = srBi.register(refImgReg, testImgReg)
            
#             # corrEachReg = np.zeros(numTotalPlane)
#             if corrKi == 0:
#                 regImgList = []
#             for i in range(numTotalPlane):
#                 refImgReg = clahe_each(zstackAvgTmp[i, estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx],
#                                        kernel_size = regKernelSize)
#                 tform = srBi.register(refImgReg, testImgReg)
#                 out = srBi.transform(testImg, tmat=tform)
#                 outEachCorr = clahe_each(out, kernel_size = corrKernelSize)
#                 refImgCorr = clahe_each(zstackAvgTmp[i, estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx],
#                                        kernel_size = corrKernelSize)
#                 corrEachReg[regKi, corrKi, avgNi, i] = np.corrcoef(outEachCorr.flatten(), refImgCorr.flatten())[0,1]
            
#                 # outSingleCorr = clahe_each(srBi.transform(testImg, tmat = tformSingle),
#                 #                               kernel_size = corrKernelSize)
#                 # corrSingleReg[i] = np.corrcoef(outSingleCorr.flatten(), refImgCorr.flatten())[0,1]
#                 if corrKi == 0:
#                     regImgList.append(out)
#             time2 = time.time()
#             avgNumLoopSpentMin = int(time2-time1)//60
#             avgNumLoopSpentSec = int(time2-time1)%60
#             print(f'{avgNumLoopSpentMin} min {avgNumLoopSpentSec} sec')
#     regImgAllList.append(regImgList)
# regLoopSpentMin = int(time2-time0)//60
# regLoopSpentSec = int(time2-time0)%60
# print(f'Total {regLoopSpentMin} min {regLoopSpentSec} sec has passed.')
# '''
# Observed that using the same transformation to the estimated depth to other plane matching
# results in reduced correlation value (reduced matching)
# Register to each plane. Within 40 um the registration was not wildly off.
# '''


# #%%
# cmap = plt.get_cmap('Blues')(range(256))[::int(256/numCK), :3]
# for avgNi in range(numNA):
#     numAvg = numAvgList[avgNi]
#     fig, ax = plt.subplots(2,3, figsize=(13,7))
#     for regKi in range(numRK):
#         regKernelFrac = int(np.round(1/regKernelFracList[regKi]))
#         ayi = regKi // 3
#         axi = regKi % 3
#         for corrKi in range(numCK):
#             corrKernelFrac = int(np.round(1/corrKernelFracList[corrKi]))
#             ax[ayi, axi].plot(range(-numTestPlane, numTestPlane+1), corrEachReg[regKi, corrKi, avgNi, :], '-', color=cmap[corrKi,:],
#                               label=f'Corr kernel frac: 1/{corrKernelFrac}')
#         if regKi == 0:
#             ax[ayi, axi].legend()
#         if regKi == 3:
#             ax[ayi, axi].set_ylabel('Pixel correation', fontsize=15)
#         if regKi == 4:
#             ax[ayi, axi].set_xlabel('Relative z-position', fontsize=15)
#         ax[ayi, axi].set_title(f'Reg kernel factor: 1/{regKernelFrac}', fontsize=15)
#         ax[ayi, axi].set_ylim(0.3, 0.6)
#         ax[ayi, axi].set_yticks([n/10 for n in range(3,7)])
#     fig.suptitle(f'Z-stack averaging {numAvg}', fontsize=20)
#     # fig.tight_layout()
# #%%

# # napari.view_image(np.array(regImgList))
# # fig, ax = plt.subplots()
# # ax.plot(range(estDepthInd-numTestPlane, estDepthInd+numTestPlane+1), corrEachReg, 'k-', label='Reg each')
# # ax.plot(range(estDepthInd-numTestPlane, estDepthInd+numTestPlane+1), corrSingleReg, 'b-', label='Reg-to-rep')
# # ax.legend()



# #%% Example of different kernel size
# kernelFracList = [1/i for i in range(3,8)]
# fig, ax = plt.subplots(3,2)
# ax[0,0].imshow(testImg, cmap='gray')
# ax[0,0].axis('off')
# for i, kf in enumerate(kernelFracList):
#     kernelSize = (int(testImg.shape[0]*kf), int(testImg.shape[1]*kf))
#     ayi = (i+1) // 2
#     axi = (i+1) % 2
#     ax[ayi, axi].imshow(clahe_each(testImg, kernel_size=kernelSize), cmap='gray')
#     ax[ayi, axi].axis('off')
#     kfrac = int(np.round(1/kf))
#     ax[ayi, axi].set_title(f'Kernel fraction 1/{kfrac}')

# '''
# It seems smaller CLAHE kernel is better for correlation calculation (larger slope past the peak)
# and somewhere-in-the-middle-sized CLAHE kernel is better for registration.
# Z-stack averaging might not be necessary?

# Just found out that kernel size would be better if it's square.
# Re-do it.
# '''

# #%%
# testImg = sessionImgsDiv[0]
# Ly = testImg.shape[0]
# Lx = testImg.shape[1]
# estDepthInd = 177
# estYpoint = 195
# estXpoint = 105

# srBi = StackReg(StackReg.BILINEAR)

# numTestPlane = 20 # up and down, total numTestPlane*2+1 planes
# numTotalPlane = numTestPlane*2+1

# # List of testants
# regKernelSizeList = [i*10 for i in range(4,11,2)]
# numRK = len(regKernelSizeList)
# corrKernelSizeList = [i*10 for i in range(4,11,2)]
# numCK = len(corrKernelSizeList)
# numAvgList = [1,3,5,11]
# numNA = len(numAvgList)
# corrEachReg = np.zeros((numRK, numCK, numNA, numTotalPlane))

# regImgAllList = []
# time0 = time.time()
# for regKi in range(numRK):
#     regKernelSize = (regKernelSizeList[regKi], regKernelSizeList[regKi])
#     testImgReg = clahe_each(testImg, kernel_size = regKernelSize)
#     for corrKi in range(numCK):
#         corrKernelSize = (corrKernelSizeList[corrKi], corrKernelSizeList[corrKi])
#         corrImgReg = clahe_each(testImg, kernel_size = corrKernelSize)
#         for avgNi in range(numNA):
#             time1 = time.time()
#             print(f'Running regKernel {regKi}/{numRK-1}, corrKernel {corrKi}/{numCK-1}, {avgNi}/{numNA-1}')
#             numAvg = numAvgList[avgNi]
            
#             zstackAvgTmp = np.zeros((numTotalPlane,zstackReg.shape[1],zstackReg.shape[2]))
#             nplanes = zstackReg.shape[0]
#             for i in range(numTotalPlane):
#                 startPlaneNum = estDepthInd - numTestPlane - (numAvg//2) + i
#                 endPlaneNum = startPlaneNum + numAvg
#                 zstackAvgTmp[i,:,:] = zstackReg[startPlaneNum:endPlaneNum, :, :].mean(axis=0)
            
#             # corrSingleReg = np.zeros(numTotalPlane)
#             # refImgReg = clahe_each(zstackAvgTmp[numTestPlane, estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx],
#             #                            kernel_size = regKernelSize)
#             # tformSingle = srBi.register(refImgReg, testImgReg)
            
#             # corrEachReg = np.zeros(numTotalPlane)
#             if corrKi == 0:
#                 regImgList = []
#             for i in range(numTotalPlane):
#                 refImgReg = clahe_each(zstackAvgTmp[i, estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx],
#                                        kernel_size = regKernelSize)
#                 tform = srBi.register(refImgReg, testImgReg)
#                 out = srBi.transform(testImg, tmat=tform)
#                 outEachCorr = clahe_each(out, kernel_size = corrKernelSize)
#                 refImgCorr = clahe_each(zstackAvgTmp[i, estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx],
#                                        kernel_size = corrKernelSize)
#                 corrEachReg[regKi, corrKi, avgNi, i] = np.corrcoef(outEachCorr.flatten(), refImgCorr.flatten())[0,1]
            
#                 # outSingleCorr = clahe_each(srBi.transform(testImg, tmat = tformSingle),
#                 #                               kernel_size = corrKernelSize)
#                 # corrSingleReg[i] = np.corrcoef(outSingleCorr.flatten(), refImgCorr.flatten())[0,1]
#                 if corrKi == 0:
#                     regImgList.append(out)
#             time2 = time.time()
#             avgNumLoopSpentMin = int(time2-time1)//60
#             avgNumLoopSpentSec = int(time2-time1)%60
#             print(f'{avgNumLoopSpentMin} min {avgNumLoopSpentSec} sec')
#     regImgAllList.append(regImgList)
# regLoopSpentMin = int(time2-time0)//60
# regLoopSpentSec = int(time2-time0)%60
# print(f'Total {regLoopSpentMin} min {regLoopSpentSec} sec has passed.')
# #%%
# cmap = plt.get_cmap('viridis_r')(range(256))[::int(256/numCK), :3]
# for avgNi in range(numNA):
#     numAvg = numAvgList[avgNi]
#     fig, ax = plt.subplots(2,2, figsize=(13,7))
#     for regKi in range(numRK):
#         regKernelSize = regKernelSizeList[regKi]
#         ayi = regKi // 2
#         axi = regKi % 2
#         for corrKi in range(numCK):
#             corrKernelSize = corrKernelSizeList[corrKi]
#             ax[ayi, axi].plot(range(-numTestPlane, numTestPlane+1), corrEachReg[regKi, corrKi, avgNi, :], '-', color=cmap[corrKi,:],
#                               label=f'Corr kernel size: ({corrKernelSize},{corrKernelSize})')
#         if regKi == 0:
#             ax[ayi, axi].legend()
#         if regKi == 2:
#             ax[ayi, axi].set_ylabel('Pixel correation', fontsize=15)
#             ax[ayi, axi].set_xlabel('Relative z-position', fontsize=15)
#         ax[ayi, axi].set_title(f'Reg kernel size: ({regKernelSize},{regKernelSize})', fontsize=15)
#         ax[ayi, axi].set_ylim(0.3, 0.6)
#         ax[ayi, axi].set_yticks([n/10 for n in range(3,7)])
#     fig.suptitle(f'Z-stack averaging {numAvg}', fontsize=20)


# #%% Compare across images with best paramters
# bestRegKernelSize = 100
# bestCorrKernelSize = 40
# bestAvgNum = 3

# #%% Compare peaks within the session
# time0 = time.time()
# corrSessionPlane = np.zeros((division, numTotalPlane))
# bestRKi = np.where(np.array(regKernelSizeList)==bestRegKernelSize)[0][0]
# bestCKi = np.where(np.array(corrKernelSizeList)==bestCorrKernelSize)[0][0]
# bestANi = np.where(np.array(numAvgList)==bestAvgNum)[0][0]

# refZstackReg = np.zeros((numTotalPlane,zstackReg.shape[1],zstackReg.shape[2]))
# refZstackCorr = np.zeros((numTotalPlane,zstackReg.shape[1],zstackReg.shape[2]))
# nplanes = zstackReg.shape[0]
# for pi in range(numTotalPlane):
#     startPlaneNum = estDepthInd - numTestPlane - (bestAvgNum // 2) + pi
#     endPlaneNum = startPlaneNum + bestAvgNum
#     tempZimg = zstackReg[startPlaneNum:endPlaneNum, :, :].mean(axis=0)
#     refZstackReg[pi,:,:] = clahe_each(tempZimg, kernel_size = bestRegKernelSize)
#     refZstackCorr[pi,:,:] = clahe_each(tempZimg, kernel_size = bestCorrKernelSize)
# time1 = time.time()
# zstackMin = int(time1-time0) // 60
# zstackSec = int(time1-time0) % 60
# print(f'{zstackMin} min {zstackSec} sec passed for z-stack registration and contrast adjustment.')

# corrSessionPlane[0,:] = corrEachReg[bestRKi, bestCKi, bestANi,:]

# for di in range(1,division):
#     time1 = time.time()
#     print(f'Running {di}/{division-1}')
#     testImg = sessionImgsDiv[di]
#     Ly = testImg.shape[0]
#     Lx = testImg.shape[1]
#     testImgReg = clahe_each(testImg, kernel_size = bestRegKernelSize)
#     corrImgReg = clahe_each(testImg, kernel_size = bestCorrKernelSize)
#     for pi in range(numTotalPlane):
#         refImgReg = refZstackReg[pi,estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx]
#         refImgCorr = refZstackCorr[pi,estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx]

#         tform = srBi.register(refImgReg, testImgReg)
#         out = srBi.transform(testImg, tmat=tform)
#         outCorr = clahe_each(out, kernel_size=bestCorrKernelSize)

#         corrSessionPlane[di,pi] = np.corrcoef(outCorr.flatten(), refImgCorr.flatten())[0,1]
#     time2 = time.time()
#     divLoopMin = int(time2-time1)//60
#     divLoopSec = int(time2-time1)%60
#     print(f'Division loop {di} done in {divLoopMin} min {divLoopSec} sec.')

# totMin = int(time2 - time0) // 60
# totSec = int(time2 - time0) % 60
# print(f'Total {totMin} min {totSec} sec passed.')
# #%%
# cmap = plt.get_cmap('plasma_r')(range(256))[::int(256/(division)), :3]
# corrPeaks = np.zeros(division)
# smoothing = 3
# fig, ax = plt.subplots(1,2)
# for di in range(division):
#     ax[0].plot(range(-numTestPlane,numTestPlane+1), corrSessionPlane[di,:], '-', color=cmap[di,:], label=f'Time bin {di}')
#     tck = interpolate.splrep(np.arange(-numTestPlane, numTestPlane+1), corrSessionPlane[di,:], s=smoothing)
#     xnew = np.arange(-numTestPlane, numTestPlane+1, 0.5)
#     ynew = interpolate.splev(xnew,tck,der=0)
#     corrPeaks[di] = np.argmax(ynew)
#     ax[1].plot(range(-numTestPlane*2,numTestPlane*2+2), ynew/np.amax(ynew), '-', color=cmap[di,:])
# ax[0].legend()
# ax[0].set_xlabel('Relative z-position', fontsize=15)
# ax[0].set_ylabel('Pixel correation', fontsize=15)
# ax[1].set_xlabel('Relative z-position in $\mu$m', fontsize=15)
# ax[1].set_ylabel('Normalized correation', fontsize=15)
# fig.suptitle(f'JK{mouse} S{sname} plane{pn}')
# zdriftMicron = corrPeaks - corrPeaks[0]

# #%% Save the result
# d = {'plane': [pn], 'session': [sname], 'estDepthInd': [estDepthInd], 'estYpoint': [estYpoint], 'estXpoint': [estXpoint],
#      'numTotalPlane': [numTotalPlane], 'bestRegKernelSize': [bestRegKernelSize], 'bestCorrKernelSize': [bestCorrKernelSize], 'bestAvgNum': [bestAvgNum],
#      'topMargin': [topMargin], 'bottomMargin': [bottomMargin], 'leftMargin': [leftMargin], 'rightMargin': [rightMargin],
#      'division': [division], 'corrSessionPlane': [corrSessionPlane], 'smoothing': [smoothing], 'corrPeaks': [corrPeaks], 'zdriftMicron': [zdriftMicron]}
# saveFn = f'{h5Dir}{mouse:03}/zdrift.csv'
# if os.path.isfile(saveFn):
#     df_zdrift = pd.read_csv(saveFn)
#     df_zdrift = df_zdrift.append(d)
# else:
#     df_zdrift = pd.DataFrame(data=d)
# df_zdrift.to_csv(saveFn)

#%%


# napari.view_image(refZstackReg)
'''
Small kernel size for registration led to lower correlation (possibly bad registration). 80 and 100 seems good. Higher the better?
Small kernel size for correlation calculation works the best (40). 60 gave higher correlation value, but less steep reduction in deeper planes (right tail).

Averaging 3 planes, CLAHE kernel size (100, 100) for registration, and (40, 40) for correlation calculation.
Test in other planes.

Include z-drift calculation.
'''





#%% Test in other session, other planes, other mice
#%% Read plane and register zstack
mi = 0
mouse = mice[mi]
refSession = refSessions[mi]
imagingFreq = freq[mi]


# Load z-stack and register using StackReg, in reverse order
srRigid = StackReg(StackReg.RIGID_BODY)

zregFnList = glob.glob(f'{zstackDir}zstack_{mouse:03}_*.mat')
zregFn = zregFnList[0]
mat = scipy.io.loadmat(zregFn)

zstack = np.moveaxis(mat['zstack'], -1, 0)
zstack = np.flip(zstack, axis=0)
nplane = zstack.shape[0]

zstackReg = srRigid.register_transform_stack(zstack[:,:,100:716], axis=0, reference='previous')

napari.view_image(zstackReg)

#%% Choose a session and calculate # of bins (division), read binary files
pn = 3
sessionNames = get_session_names(f'{h5Dir}{mouse:03}/plane_{pn}/', mouse, pn)
si = 3
sname = sessionNames[si][4:]

trialsFnList = glob.glob(f'{h5Dir}{mouse:03}/{mouse:03}_{sname}_*.trials')
if len(trialsFnList) == 1:
    trials = scipy.io.loadmat(trialsFnList[0])
    frameToUse = trials['frame_to_use'][0,pn-1]
else:
    nfiles = len(trialsFnList)
    print(f'Importing {nfiles} files...')
    frameToUse = []
    for tfn in trialsFnList:
        trials = scipy.io.loadmat(tfn)
        if len(frameToUse)==0:
            frameToUse = trials['frame_to_use'][0,pn-1]
        else:
            tempFtu = trials['frame_to_use'][0,pn-1] + np.amax(frameToUse)
            frameToUse = np.hstack((frameToUse, tempFtu))

tminSpent = round((frameToUse[0,-1] - frameToUse[0,0]) / (imagingFreq*4) / 60)

print(f'JK{mouse:03} S_{sname} plane {pn} spent about {tminSpent} min.')
division = round(tminSpent/10)
print(f'Running with {division} divisions')

# Read binary files

sessionOps = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/{sname}/plane0/ops.npy', allow_pickle=True).item()
Ly = sessionOps['Ly']
Lx = sessionOps['Lx']
nframes = sessionOps['nframes']
sessionImgsDivTemp = []
with BinaryFile(Ly, Lx, read_filename=f'{h5Dir}{mouse:03}/plane_{pn}/{sname}/plane0/data.bin') as f:
    data = f.data
    for i in range(division):
        tempStartFrame = (i*nframes) // division
        tempEndFrame = ((i+1)*nframes) // division
        sessionImgsDivTemp.append(data[tempStartFrame:tempEndFrame,:,:].mean(axis=0))
napari.view_image(np.array(sessionImgsDivTemp))


#%% Remove margins and match with z-stack by eyes

topMargin = 28
bottomMargin = 381
leftMargin = 21
rightMargin = 681
sessionImgsDiv = [sidt[topMargin:bottomMargin,leftMargin:rightMargin] for sidt in sessionImgsDivTemp]

mimg = sessionOps['meanImg'][topMargin:bottomMargin,leftMargin:rightMargin]
napari.view_image(mimg)

'''
Visual matching
'''

#%% Check registration

estDepthInd = 210
estYpoint = 199
estXpoint = 0

testImg = sessionImgsDiv[0]
Ly = min(testImg.shape[0], zstackReg.shape[1]-estYpoint)
Lx = min(testImg.shape[1], zstackReg.shape[2]-estXpoint)
testImg = testImg[:Ly, :Lx]

srBi = StackReg(StackReg.BILINEAR)
srRigid = StackReg(StackReg.RIGID_BODY)

numTestPlane = 20 # up and down, total numTestPlane*2+1 planes
numTotalPlane = numTestPlane*2+1

# List of testants
srList = [srBi, srRigid]
# regKernelSizeList = [i*10 for i in range(4,11,2)]
regKernelSizeList = [100, 300]
numRK = len(regKernelSizeList)
# corrKernelSizeList = [i*10 for i in range(4,11,2)]
corrKernelSizeList = [40]
numCK = len(corrKernelSizeList)
# numAvgList = [1,3,5,11]
numAvgList = [5, 11]
numNA = len(numAvgList)
corrEachReg = np.zeros((numRK, numCK, numNA, numTotalPlane, len(srList)))

regImgAllList = []
time0 = time.time()
for regKi in range(numRK):
    regKernelSize = (regKernelSizeList[regKi], regKernelSizeList[regKi])
    testImgReg = clahe_each(testImg, kernel_size = regKernelSize)
    for corrKi in range(numCK):
        corrKernelSize = (corrKernelSizeList[corrKi], corrKernelSizeList[corrKi])
        corrImgReg = clahe_each(testImg, kernel_size = corrKernelSize)
        for avgNi in range(numNA):
            time1 = time.time()
            print(f'Running regKernel {regKi}/{numRK-1}, corrKernel {corrKi}/{numCK-1}, {avgNi}/{numNA-1}')
            numAvg = numAvgList[avgNi]
            
            zstackAvgTmp = np.zeros((numTotalPlane,zstackReg.shape[1],zstackReg.shape[2]))
            nplanes = zstackReg.shape[0]
            for i in range(numTotalPlane):
                startPlaneNum = estDepthInd - numTestPlane - (numAvg//2) + i
                endPlaneNum = startPlaneNum + numAvg
                zstackAvgTmp[i,:,:] = zstackReg[startPlaneNum:endPlaneNum, :, :].mean(axis=0)
            
            # corrSingleReg = np.zeros(numTotalPlane)
            # refImgReg = clahe_each(zstackAvgTmp[numTestPlane, estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx],
            #                            kernel_size = regKernelSize)
            # tformSingle = srBi.register(refImgReg, testImgReg)
            
            # corrEachReg = np.zeros(numTotalPlane)
            if corrKi == 0:
                regImgList = []
            for i in range(numTotalPlane):
                for sri in range(len(srList)):
                    sr = srList[sri]
                    refImgReg = clahe_each(zstackAvgTmp[i, estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx],
                                           kernel_size = regKernelSize)
                    tform = sr.register(refImgReg, testImgReg)
                    out = sr.transform(testImg, tmat=tform)
    
                    outEachCorr = clahe_each(out, kernel_size = corrKernelSize)
                    refImgCorr = clahe_each(zstackAvgTmp[i, estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx],
                                           kernel_size = corrKernelSize)
                    corrEachReg[regKi, corrKi, avgNi, i, sri] = np.corrcoef(outEachCorr.flatten(), refImgCorr.flatten())[0,1]
                
                    # outSingleCorr = clahe_each(srBi.transform(testImg, tmat = tformSingle),
                    #                               kernel_size = corrKernelSize)
                    # corrSingleReg[i] = np.corrcoef(outSingleCorr.flatten(), refImgCorr.flatten())[0,1]
                    if corrKi == 0:
                        regImgList.append(out)
            time2 = time.time()
            avgNumLoopSpentMin = int(time2-time1)//60
            avgNumLoopSpentSec = int(time2-time1)%60
            print(f'{avgNumLoopSpentMin} min {avgNumLoopSpentSec} sec')
    regImgAllList.append(regImgList)
regLoopSpentMin = int(time2-time0)//60
regLoopSpentSec = int(time2-time0)%60
print(f'Total {regLoopSpentMin} min {regLoopSpentSec} sec has passed.')

cmap = plt.get_cmap('viridis_r')(range(256))[::int(256/numCK), :3]
for avgNi in range(numNA):
    numAvg = numAvgList[avgNi]
    for sri in range(len(srList)):
        fig, ax = plt.subplots(2,2, figsize=(13,7))
        for regKi in range(numRK):
            regKernelSize = regKernelSizeList[regKi]
            ayi = regKi // 2
            axi = regKi % 2
            for corrKi in range(numCK):
                corrKernelSize = corrKernelSizeList[corrKi]
                ax[ayi, axi].plot(range(-numTestPlane, numTestPlane+1), corrEachReg[regKi, corrKi, avgNi, :, sri], '-', color=cmap[corrKi,:],
                                  label=f'Corr kernel size: ({corrKernelSize},{corrKernelSize})')
            if regKi == 0:
                ax[ayi, axi].legend()
            if regKi == 2:
                ax[ayi, axi].set_ylabel('Pixel correation', fontsize=15)
                ax[ayi, axi].set_xlabel('Relative z-position', fontsize=15)
            ax[ayi, axi].set_title(f'Reg kernel size: ({regKernelSize},{regKernelSize})', fontsize=15)
            # ax[ayi, axi].set_ylim(0.3, 0.6)
            # ax[ayi, axi].set_yticks([n/10 for n in range(3,7)])
        fig.suptitle(f'Z-stack averaging {numAvg}, sri = {sri}', fontsize=20)


#%% Compare peaks within the session
#%% Compare across images with best paramters
bestRegKernelSize = 100
bestCorrKernelSize = 40
bestAvgNum = 5
sr = srBi

time0 = time.time()
corrSessionPlane = np.zeros((division, numTotalPlane))

refZstack = np.zeros((numTotalPlane,Ly,Lx))
refZstackReg = np.zeros((numTotalPlane,Ly,Lx))
refZstackCorr = np.zeros((numTotalPlane,Ly,Lx))
nplanes = zstackReg.shape[0]
for pi in range(numTotalPlane):
    startPlaneNum = estDepthInd - numTestPlane - (bestAvgNum // 2) + pi
    endPlaneNum = startPlaneNum + bestAvgNum
    tempZimg = zstackReg[startPlaneNum:endPlaneNum, estYpoint:estYpoint+Ly, estXpoint:estXpoint+Lx].mean(axis=0)
    refZstack[pi,:,:] = tempZimg
    refZstackReg[pi,:,:] = clahe_each(tempZimg, kernel_size = bestRegKernelSize)
    refZstackCorr[pi,:,:] = clahe_each(tempZimg, kernel_size = bestCorrKernelSize)
    
time1 = time.time()
zstackMin = int(time1-time0) // 60
zstackSec = int(time1-time0) % 60
print(f'{zstackMin} min {zstackSec} sec passed for z-stack registration and contrast adjustment.')

tformList = []
for di in range(division):
    time1 = time.time()
    print(f'Running {di}/{division-1}')
    testImg = sessionImgsDiv[di]
    Ly = min(testImg.shape[0], zstackReg.shape[1]-estYpoint)
    Lx = min(testImg.shape[1], zstackReg.shape[2]-estXpoint)
    testImg = testImg[:Ly, :Lx]
    
    testImgReg = clahe_each(testImg, kernel_size = bestRegKernelSize)
    corrImgReg = clahe_each(testImg, kernel_size = bestCorrKernelSize)
    for pi in range(numTotalPlane):
        refImgReg = refZstackReg[pi,:, :]
        refImgCorr = refZstackCorr[pi,:, :]
        
        if di == 0:
            tform = sr.register(refImgReg, testImgReg)
            tformList.append(tform)
        else:
            tform = tformList[pi]
        out = sr.transform(testImg, tmat=tform)
        outCorr = clahe_each(out, kernel_size=bestCorrKernelSize)

        corrSessionPlane[di,pi] = np.corrcoef(outCorr.flatten(), refImgCorr.flatten())[0,1]
    time2 = time.time()
    divLoopMin = int(time2-time1)//60
    divLoopSec = int(time2-time1)%60
    print(f'Division loop {di} done in {divLoopMin} min {divLoopSec} sec.')

totMin = int(time2 - time0) // 60
totSec = int(time2 - time0) % 60
print(f'Total {totMin} min {totSec} sec passed.')

cmap = plt.get_cmap('plasma_r')(range(256))[::int(256/(division)), :3]
# corrPeaks = np.zeros(division)
# smoothing = 3
# fig, ax = plt.subplots(1,2)
# for di in range(division):
#     ax[0].plot(range(-numTestPlane,numTestPlane+1), corrSessionPlane[di,:], '-', color=cmap[di,:], label=f'Time bin {di}')
#     tck = interpolate.splrep(np.arange(-numTestPlane, numTestPlane+1), corrSessionPlane[di,:], s=smoothing)
#     xnew = np.arange(-numTestPlane, numTestPlane+1, 0.5)
#     ynew = interpolate.splev(xnew,tck,der=0)
#     corrPeaks[di] = np.argmax(ynew)
#     ax[1].plot(range(-numTestPlane*2,numTestPlane*2+2), ynew/np.amax(ynew), '-', color=cmap[di,:])
# ax[0].legend()
# ax[0].set_xlabel('Relative z-position', fontsize=15)
# ax[0].set_ylabel('Pixel correation', fontsize=15)
# ax[1].set_xlabel('Relative z-position in $\mu$m', fontsize=15)
# ax[1].set_ylabel('Normalized correation', fontsize=15)
# fig.suptitle(f'JK{mouse} S{sname} plane{pn}')

smoothing = 1
fig, ax = plt.subplots()
for di in range(division):
    ax.plot(range(-numTestPlane,numTestPlane+1), corrSessionPlane[di,:], '-', color=cmap[di,:], label=f'Time bin {di}')
ax.legend()
ax.set_xlabel('Relative z-position', fontsize=15)
ax.set_ylabel('Pixel correation', fontsize=15)
ax.set_title(f'JK{mouse} S{sname} plane{pn}')
corrPeaks = np.argmax(corrSessionPlane, axis=1)*2

zdriftMicron = (corrPeaks - corrPeaks[0]).astype(int)

print(zdriftMicron)
perHrAvg = round(zdriftMicron[-1]/(division-1)*6)
print(f'{perHrAvg} /hr')
#%%
normCorrSessionPlane = corrSessionPlane / np.tile(np.amax(corrSessionPlane, axis=1), (numTotalPlane, 1)).T
figure, ax = plt.subplots()
for di in range(division):
    ax.plot(range(-numTestPlane,numTestPlane+1), normCorrSessionPlane[di,:], '-', color=cmap[di,:], label=f'Time bin {di}')
#%%
napari.view_image(refZstack)
#%% Save the result
d = {'plane': [pn], 'session': [sname], 'estDepthInd': [estDepthInd], 'estYpoint': [estYpoint], 'estXpoint': [estXpoint],
     'numTotalPlane': [numTotalPlane], 'bestRegKernelSize': [bestRegKernelSize], 'bestCorrKernelSize': [bestCorrKernelSize], 'bestAvgNum': [bestAvgNum],
     'topMargin': [topMargin], 'bottomMargin': [bottomMargin], 'leftMargin': [leftMargin], 'rightMargin': [rightMargin],
     'division': [division], 'corrSessionPlane': [corrSessionPlane], 'smoothing': [smoothing], 'corrPeaks': [corrPeaks], 'zdriftMicron': [zdriftMicron]}
saveFn = f'{h5Dir}{mouse:03}/zdrift.csv'
if os.path.isfile(saveFn):
    df_zdrift = pd.read_csv(saveFn)
    df_zdrift = df_zdrift.append(d, ignore_index=True)
else:
    df_zdrift = pd.DataFrame(data=d)
df_zdrift.to_csv(saveFn)

print(f'S_{sname} plane {pn} saved.')

















#%% 2021/10/29
#%% Re-calculating peaks
z027 = pd.read_csv('D:/TPM/JK/h5/zdrift_027.csv')
z036 = pd.read_csv('D:/TPM/JK/h5/zdrift_036.csv')
z052 = pd.read_csv('D:/TPM/JK/h5/zdrift_052.csv')


#%%












#%% Example of clahe
mimg2 = clahe_each(mimg, kernel_size=100)
mimg3 = clahe_each(mimg, kernel_size=40)
napari.view_image(np.array([img_norm(mimg), img_norm(mimg2), img_norm(mimg3)]))

#%%
cmap = plt.get_cmap('plasma_r')(range(256))[::int(256/(10)), :3]
fig, ax = plt.subplots()

for i in range(10):
    ax.plot(range(-20,21), a[i], color=cmap[i,:], label=f'Time bin {i}')
ax.legend()

ax.set_xlabel('Relative z-position', fontsize=15)
ax.set_ylabel('Pixel correation', fontsize=15)





















#%%
#%%
#%% simpleITK registration
#%%
#%%
#%%
fixed = sitk.GetImageFromArray(refImg)
moving = sitk.GetImageFromArray(testImg)


#%% bspline registration
transformDomainMeshSize = [8] * moving.GetDimension()
tx = sitk.BSplineTransformInitializer(fixed,
                                      transformDomainMeshSize)

R = sitk.ImageRegistrationMethod()
R.SetMetricAsCorrelation()

R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                       numberOfIterations=100,
                       maximumNumberOfCorrections=5,
                       maximumNumberOfFunctionEvaluations=1000,
                       costFunctionConvergenceFactor=1e+7)
R.SetInitialTransform(tx, True)
R.SetInterpolator(sitk.sitkLinear)

outTx = R.Execute(fixed, moving)
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(100)
resampler.SetTransform(outTx)

out = resampler.Execute(moving)
resultImg = sitk.GetArrayFromImage(out)

bsplineRefImg = clahe_each(refImg)
bsplineRegImg = clahe_each(resultImg)
bsplineBlend = imblend(bsplineRefImg, bsplineRegImg)

#%% demons registration
demons = sitk.DemonsRegistrationFilter()
demons.SetNumberOfIterations(1000)
demons.SetStandardDeviations(10.0)

displacementField = demons.Execute(fixed, moving)

outTx = sitk.DisplacementFieldTransform(displacementField)

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(100)
resampler.SetTransform(outTx)

out = resampler.Execute(moving)
# simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
# simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
# cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
resultImg = sitk.GetArrayFromImage(out)
# plt.imshow(resultImg, cmap='gray')
# plt.axis('off')
demonRefImg = clahe_each(refImg)
demonRegImg = clahe_each(resultImg)
demonBlend = imblend(demonRefImg, demonRegImg)
