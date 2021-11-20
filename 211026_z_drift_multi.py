
"""
Automatic z-drift calculation across sessions
Pre-determining estimated depth at each plane, and then run the whole mouse
Estimate best depth at each session and at each 10-min window

To do this, first register mean images across sessions

Use a single parameter, Avg 5 / Reg Kernel 100 / Corr Kernel 40
Save correlation values for later confirmation.
For a session with weird pattern, change kernels

Results should contain matched depth (in absolute values) for each 10-min window
from each session and plane

2021/10/26 JK
"""


import scipy.io
import numpy as np
import pandas as pd
import napari
import os, glob
from skimage import exposure
from suite2p.io.binary import BinaryFile
from suite2p.registration.register import enhanced_mean_image
from suite2p.registration import rigid, nonrigid
import matplotlib.pyplot as plt
from scipy import interpolate
import time
from pystackreg import StackReg
import gc
gc.enable()

def clahe_each(img: np.float64, kernel_size = None, clip_limit = 0.01, nbins = 2**16):
    newimg = img.copy()
    if len(newimg.shape) == 2:
        newimg = np.expand_dims(newimg, axis=0)
    for i in range(newimg.shape[0]):
        newimg[i,:,:] = (newimg[i,:,:] - np.amin(newimg[i,:,:])) / (np.amax(newimg[i,:,:]) - np.amin(newimg[i,:,:]))
    newimg = np.squeeze(exposure.equalize_adapthist(newimg, kernel_size = kernel_size, clip_limit = clip_limit, nbins=nbins))
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
        
def corr_match_z_clahed(imgs4reg, imgs4corr, zstack4reg, zstack4corr, stackReg = None):
    if imgs4reg.shape != imgs4corr.shape:
        raise('Input images for registration and correlation should have matched dimensions.')
    if imgs4reg.ndim == 2:
        imgs4reg = np.expand_dims(imgs4reg, axis=0)
        imgs4corr = np.expand_dims(imgs4corr, axis=0)
    elif imgs4reg.ndim != 3:
        raise('Input images should be either 2 or 3 dimensions.')
        
    if (zstack4reg.ndim != 3) or (zstack4corr.ndim != 3):
        raise('Zstack image should have 3 dimensions.')
    
    if stackReg is None:
        stackReg = StackReg(StackReg.BILINEAR)
        print('stackReg not given. Processing with bilinear registration.')
    else:
        print('Using given stackReg.')
    
    nplaneImg = imgs4reg.shape[0]
    nplaneZstack = zstack4reg.shape[0]
    corrVals = np.zeros((nplaneImg, nplaneZstack))
    tformList = []
    for zpi in range(nplaneZstack):
        refImgReg = zstack4reg[zpi,:,:]
        tempReg = np.array([refImgReg, *imgs4reg])
        tform = stackReg.register_stack(tempReg, axis=0, reference='first')
        tformList.append(tform[1:,:,:])
        tempCorr = np.array([refImgReg, *imgs4corr])
        out = stackReg.transform_stack(tempCorr)
        out = out[1:,:,:]
        refImgCorr = zstack4corr[zpi,:,:]
        for ipi in range(nplaneImg):
            corrVals[ipi,zpi] = np.corrcoef(out[ipi,:,:].flatten(), refImgCorr.flatten())[0,1]
    return corrVals, tformList
    
def corr_match_to_best_z(imgs4reg, imgs4corr, zstack4reg, zstack4corr, stackReg = None):
    if imgs4reg.shape != imgs4corr.shape:
        raise('Input images for registration and correlation should have matched dimensions.')
    if imgs4reg.ndim == 2:
        imgs4reg = np.expand_dims(imgs4reg, axis=0)
        imgs4corr = np.expand_dims(imgs4corr, axis=0)
    elif imgs4reg.ndim != 3:
        raise('Input images should be either 2 or 3 dimensions.')
        
    if (zstack4reg.ndim != 3) or (zstack4corr.ndim != 3):
        raise('Zstack image should have 3 dimensions.')
    
    if stackReg is None:
        stackReg = StackReg(StackReg.BILINEAR)
        print('stackReg not given. Processing with bilinear registration.')
    else:
        print('Using given stackReg.')

    nplaneImg = imgs4reg.shape[0]
    nplaneZstack = zstack4reg.shape[0]
    corrValsFirst = np.zeros((nplaneImg, nplaneZstack))
    tformList = []
    for zpi in range(nplaneZstack):
        refImgReg = zstack4reg[zpi,:,:]
        tempReg = np.array([refImgReg, *imgs4reg])
        tform = stackReg.register_stack(tempReg, axis=0, reference='first')
        tformList.append(tform[1:,:,:])
        tempCorr = np.array([refImgReg, *imgs4corr])
        out = stackReg.transform_stack(tempCorr)
        out = out[1:,:,:]
        refImgCorr = zstack4corr[zpi,:,:]
        for ipi in range(nplaneImg):
            corrValsFirst[ipi,zpi] = np.corrcoef(out[ipi,:,:].flatten(), refImgCorr.flatten())[0,1]
    bestTformList = []
    corrVals = np.zeros((nplaneImg, nplaneZstack))
    zcorrFlatten = np.reshape(zstack4corrTrim, (nplaneZstack,-1))
    for ipi in range(nplaneImg):
        maxzi = np.argmax(corrValsFirst[ipi,:])
        maxTform = tformList[maxzi][ipi,:,:]
        bestTformList.append(maxTform)
        outTemp = stackReg.transform(imgs4corr[ipi,:,:], tmat=maxTform)
        outFlatten = np.expand_dims(outTemp.flatten(),axis=0)
        corrVals[ipi,:] = np.corrcoef(outFlatten, zcorrFlatten)[0,1:]
    return corrVals, bestTformList, corrValsFirst, tformList

def s2p_register(mimgList, op, rigid_offsets, nonrigid_offsets = None):
    # Register using rigid (and nonrigid, optionally) offsets
    
    # Rigid registration
    frames = np.array(mimgList).astype(np.float32)
    ymax, xmax = rigid_offsets[0][0][0], rigid_offsets[0][1][0]
    for frame in frames:
        frame[:] = rigid.shift_frame(frame=frame, dy=ymax, dx=xmax)
        
    # Nonrigid registration
    if nonrigid_offsets is not None:
        Ly, Lx = frames.shape[1], frames.shape[2]
        yblock, xblock, nblocks, _, _= nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=op['block_size'])
        
        ymax1, xmax1 = nonrigid_offsets[0][0], nonrigid_offsets[0][1]
        nframes = frames.shape[0]
        ymax1 = np.tile(ymax1, (nframes,1))
        xmax1 = np.tile(xmax1, (nframes,1))
        frames = nonrigid.transform_data(
            data=frames,
            nblocks=nblocks,
            xblock=xblock,
            yblock=yblock,
            ymax1=ymax1,
            xmax1=xmax1,
        )
    return frames

def s2p_nonrigid_registration(mimgList, refImg, op):
    ### ------------- compute registration masks ----------------- ###
    Ly, Lx = refImg.shape
    maskMul, maskOffset = rigid.compute_masks(
        refImg=refImg,
        maskSlope= 3 * op['smooth_sigma'],
    )
    cfRefImg = rigid.phasecorr_reference(
        refImg=refImg,
        smooth_sigma=op['smooth_sigma'],
        # pad_fft=ops['pad_fft'], # False by default
    )

    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=op['block_size'])
    
    maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(
        refImg0=refImg,
        maskSlope=3 * op['smooth_sigma'], # slope of taper mask at the edges
        smooth_sigma=op['smooth_sigma'],
        yblock=yblock,
        xblock=xblock,
        # pad_fft=ops['pad_fft'], # False by default
    )

    ### ------------- register binary to reference image ------------ ###
    
    mean_img = np.zeros((Ly, Lx))
    rigid_offsets, nonrigid_offsets = [], []

    frames = np.array(mimgList).astype(np.float32)
    fsmooth = frames.copy().astype(np.float32)

    # rigid registration
    ymax, xmax, cmax = rigid.phasecorr(
        data=rigid.apply_masks(data=fsmooth, maskMul=maskMul, maskOffset=maskOffset),
        cfRefImg=cfRefImg,
        maxregshift=op['maxregshift'],
        smooth_sigma_time=op['smooth_sigma_time'],
    )
    rigid_offsets.append([ymax, xmax, cmax])

    for frame, dy, dx in zip(frames, ymax, xmax):
        frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)

    # non-rigid registration
    # need to also shift smoothed data (if smoothing used)
    fsmooth = frames.copy()
        
    ymax1, xmax1, cmax1 = nonrigid.phasecorr(
        data=fsmooth,
        maskMul=maskMulNR.squeeze(),
        maskOffset=maskOffsetNR.squeeze(),
        cfRefImg=cfRefImgNR.squeeze(),
        snr_thresh=op['snr_thresh'],
        NRsm=NRsm,
        xblock=xblock,
        yblock=yblock,
        maxregshiftNR=op['maxregshiftNR'],
    )

    frames = nonrigid.transform_data(
        data=frames,
        nblocks=nblocks,
        xblock=xblock,
        yblock=yblock,
        ymax1=ymax1,
        xmax1=xmax1,
    )

    nonrigid_offsets.append([ymax1, xmax1, cmax1])
    
    return frames, rigid_offsets, nonrigid_offsets


h5Dir = 'D:/TPM/JK/h5/'
zstackDir = 'D:/TPM/JK/h5/zstacks/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]


numAvg = 11
regKernel = 100
corrKernel = 50
timeBin = 10 # in min

#%% Load z-stack and register using rigid body
# And load absolute depth calculation
mi = 0
mouse = mice[mi]
imagingFreq = freq[mi]
refSession = refSessions[mi]

srRigid = StackReg(StackReg.RIGID_BODY)

zregFnList = glob.glob(f'{zstackDir}zstack_{mouse:03}_*.mat')
zregFn = zregFnList[0]
mat = scipy.io.loadmat(zregFn)

zstack = np.moveaxis(mat['zstack'], -1, 0)
zstack = np.flip(zstack, axis=0)
nplane = zstack.shape[0]

if mi < 8:
    zstackReg = srRigid.register_transform_stack(zstack, axis=0, reference='previous')
else:
    zstackReg = srRigid.register_transform_stack(zstack[:,:,90:710], axis=0, reference='previous')
zstackAvg = np.zeros(zstackReg.shape)
zstack4reg = np.zeros(zstackReg.shape)
zstack4corr = np.zeros(zstackReg.shape)
nplane = zstackReg.shape[0]
for pi in range(nplane):
    zstackAvg[pi,:,:] = np.mean(zstackReg[max(0, pi-numAvg//2):min(nplane, pi+numAvg//2)], axis=0)
    zstack4reg[pi,:,:] = clahe_each(zstackAvg[pi,:,:], kernel_size = regKernel)
    zstack4corr[pi,:,:] = clahe_each(zstackAvg[pi,:,:], kernel_size = corrKernel)
napari.view_image(zstackAvg)

zInfoList = glob.glob(f'{zstackDir}zstackReg_{mouse:03}_*.mat')
zInfoFn = zInfoList[0]

zinfo = scipy.io.loadmat(zInfoFn)
absDepthVal = np.flip(zinfo['zstackDepths'])

#%% Trimming and finding matching x,y,z from the z-stack (per plane)
# Register across sessions
# Using StackReg Bilinear
# Visual confirmation afterwards (especially for anesthetized sessions)
# Select the reference session, and 


#%% First, look at all mean images and select trimming parameters
# Also, save time-binned images in each session
pn = 5
sessionNames = get_session_names(f'{h5Dir}{mouse:03}/plane_{pn}/', mouse, pn)

mimgList = []
tminSpentList = []
nframesList = []
numBinList = []
sessionImgsDivList = []
numSessions = len(sessionNames)

tempPn = pn if pn < 5 else pn-4
for i, sn in enumerate(sessionNames):
    print(f'Processing session #{i}/{numSessions}')
    sname = sn[4:]
    sessionOps = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/{sname}/plane0/ops.npy', allow_pickle=True).item()
    mimgList.append(sessionOps['meanImg'])
    if len(sname.split('_'))==1:
        trialsFnList = glob.glob(f'{h5Dir}{mouse:03}/{mouse:03}_{sname}_*.trials')
    else:
        trialsFnList = glob.glob(f'{h5Dir}{mouse:03}/{mouse:03}_{sname}*.trials')
    if len(trialsFnList) == 1:
        trials = scipy.io.loadmat(trialsFnList[0])
        if len(sname) == 3:
            frameToUse = trials['frame_to_use'][0, pn-1]
        elif len(sname) >= 6:
            frameToUse = trials['frame_to_use'][tempPn-1]
            if len(frameToUse) == 1:
                frameToUse = frameToUse[0]
    else:
        nfiles = len(trialsFnList)
        print(f'Importing {nfiles} files...')
        frameToUse = []
        for tfn in trialsFnList:
            trials = scipy.io.loadmat(tfn)

            if len(sname) == 3:
                try:
                    temp = trials['frame_to_use'][0, pn-1]
                except:
                    temp = trials['frame_to_use'][pn-1]
            elif len(sname) >= 6:
                try:
                    temp = trials['frame_to_use'][0, tempPn-1]
                except:
                    temp = trials['frame_to_use'][tempPn-1]

            if len(frameToUse)==0:
                frameToUse = temp
            else:
                tempFtu = temp + np.amax(frameToUse)
                frameToUse = np.hstack((frameToUse, tempFtu))
    frameToUse = np.squeeze(frameToUse)
    tminSpent = round((frameToUse[-1] - frameToUse[0]) / (imagingFreq*4) / 60)
    tminSpentList.append(tminSpent)
    numBin = round(tminSpent/timeBin)
    numBinList.append(numBin)
    nframes = sessionOps['nframes']
    nframesList.append(nframes)
    
    # Save 10-min window averaged images
    Ly = sessionOps['Ly']
    Lx = sessionOps['Lx']
    sessionImgsDivTemp = []
    with BinaryFile(Ly, Lx, read_filename=f'{h5Dir}{mouse:03}/plane_{pn}/{sname}/plane0/data.bin') as f:
        data = f.data
        for i in range(numBin):
            tempStartFrame = (i*nframes) // numBin
            tempEndFrame = ((i+1)*nframes) // numBin
            sessionImgsDivTemp.append(data[tempStartFrame:tempEndFrame,:,:].mean(axis=0))
    sessionImgsDivList.append(np.array(sessionImgsDivTemp))

napari.view_image(np.array(mimgList))

#%% Set trimming parameters and check the result
# Rigid and bilinear registration works similarly at this point
# Rigid seems more reasonable

topMargin = 11 # Could be larger for the bottom-most plane (4 and 8)
bottomMargin = 363
leftMargin = 2
rightMargin = 683

trimMimg = np.array([img[topMargin:bottomMargin, leftMargin:rightMargin] for img in mimgList])

refSi = np.where(np.array([sn.split('_')[1] == f'{refSession:03}' for sn in sessionNames]))[0][0]
refSessionMimg = trimMimg[refSi,:,:]
trimMimgArr = np.array([refSessionMimg, *trimMimg])
trimMimgClahe = np.array([clahe_each(img, 100) for img in trimMimgArr])

# srBi = StackReg(StackReg.BILINEAR)
srRigid = StackReg(StackReg.RIGID_BODY)
tformSessionRigid = srRigid.register_stack(trimMimgClahe, axis=0, reference='first') # for later use
outSessionRigid = srRigid.transform_stack(trimMimgArr, tmats = tformSessionRigid)

# tformSessionBi = srBi.register_stack(trimMimgClahe, axis=0, reference='first') # for later use
# outSessionBi = srBi.transform_stack(trimMimgClahe)

outSessionRigid = outSessionRigid[1:,:,:]
tformSessionRigid = tformSessionRigid[1:,:,:]
# outSessionBi = outSessionBi[1:,:,:]
trimMimg = trimMimg[1:,:,:]

viewer = napari.Viewer()
viewer.add_image(trimMimg, name='before reg')
# viewer.add_image(outSessionBi, name='after bilinear reg')
viewer.add_image(outSessionRigid, name='after rigid reg')
viewer.add_image(refSessionMimg, name='ref session')

outSession = outSessionRigid.copy()
#%% Find x,y,z position of the reference sessions
estZ = 96
estY = 62
estX = 76





'''
Try finding the best parameter (number of averaging, kernel size for registration and correlation calculation)
'''
# #%% Find best-matched z positions
# # Compare between each session matching, vs ref session matching and applying to the rest
# # Also, compare between different # of planes for averaging.
# # With the same kernels.
# #%% First, use reference image to find the best parameters
# numAvgList= [5, 10, 20]
# kernelList = [50, 100, 200, 300]
# zthickness = 50
# zrange = range(max(0,estZ-zthickness//2), min(nplane, estZ+zthickness//2))
# srBi = StackReg(StackReg.BILINEAR)
# useY = min(refSessionMimg.shape[0], zstackReg.shape[1]-estY)
# useX = min(refSessionMimg.shape[1], zstackReg.shape[2]-estX)
# zstackTrim = zstackReg[:,estY:estY+useY, estX:estX+useX].copy()
# refSessionMimg = outSession[refSi,:,:]
# refTrim = refSessionMimg[:useY, :useX].copy()

# zstackAvgList = []
# zstackClaheList = []
# nplane = zstackTrim.shape[0]
# for numAvg in numAvgList:
#     print(f'Averaging with {numAvg}')
#     t1 = time.time()
#     zstackAvgTemp = np.zeros(zstackTrim.shape)
#     for pi in range(nplane):
#         zstackAvgTemp[pi,:,:] = np.mean(zstackTrim[max(0, pi-numAvg//2):min(nplane, pi+numAvg//2)], axis=0)
#     zstackAvgList.append(zstackAvgTemp)


# for ni in range(len(numAvgList)):
#     zstackAvgTemp = zstackAvgList[ni][zrange,:,:]
#     zstackClaheTemp = np.zeros((len(kernelList), *zstackAvgTemp.shape))
#     for ki in range(len(kernelList)):
#         kernelSize = kernelList[ki]
#         zstackClaheTemp[ki,:,:,:] = clahe_each(zstackAvgTemp, kernel_size = kernelSize)
#     zstackClaheList.append(zstackClaheTemp)
# print(f'Zstack averaging done.')

# refTrimClaheList = []
# for ki in range(len(kernelList)):
#     kernelSize = kernelList[ki]
#     refTrimClaheList.append(clahe_each(refTrim, kernel_size = kernelSize))

# numKer = len(kernelList)
# corrValsList = []
# lenNav = len(numAvgList)
# for ni in range(lenNav):
#     print(f'Processing numAvg #{ni}/{lenNav}')
#     for ri in range(numKer):
#         z4reg = np.squeeze(zstackClaheList[ni][ri,:,:,:])
#         ref4reg = np.squeeze(refTrimClaheList[ri].copy())
#         tmats = []
#         for ci in range(numKer):
#             corrValsTemp = np.zeros(zthickness)
#             z4corr = np.squeeze(zstackClaheList[ni][ci,:,:,:])
#             ref4corr = np.squeeze(refTrimClaheList[ci].copy())
#             for pi in range(zthickness):
#                 srBi.register(ref4reg, z4reg[pi,:,:])
#                 tempImg = srBi.transform(ref4corr)
#                 corrValsTemp[pi] = np.corrcoef(tempImg.flatten(), z4corr[pi,:,:].flatten())[0,1]
#             corrValsList.append(corrValsTemp)
# #%%            
# for ni in range(lenNav):
#     numAvg = numAvgList[ni]
#     fig, ax = plt.subplots(2, int(np.ceil(len(kernelList)/2)), figsize=(15,10))
#     for ri in range(numKer):
#         regKer = kernelList[ri]
#         yi = ri//2
#         xi = ri%2
#         for ci in range(numKer):
#             corrKer = kernelList[ci]
#             ax[yi,xi].plot(corrValsList[ni*numKer*numKer+ri*numKer+ci], label=f'corr kernel {corrKer}')
#         if yi==0 and xi==0:
#             ax[yi,xi].legend()
#         if yi==1 and xi==0:
#             ax[yi,xi].set_ylabel('Correlation')
#             ax[yi,xi].set_xlabel('Session index')
#         ax[yi,xi].set_title(f'reg kernel {regKer}')
#     fig.suptitle(f'Averaging {numAvg} planes')
#     fig.tight_layout()



'''
Change of plan.
It seems plane 4 in some of the sessions are not just able to fit.
The goal now is to find out drift difference between plane 1 and plane 4.
(Hope there is not...)
Just fix # of average to 11, reg kernel size 100, and corr kernel size 40.
Visit each session, draw the curve, and select the sessions with good curve.
In these sessions, compute drift in plane 1. 
Compare the drift.

'''


#%%
zthickness = 50
zrange = range(max(0,estZ-zthickness//2), min(nplane, estZ+zthickness//2))

imgs4reg = np.zeros(outSession.shape)
imgs4corr = np.zeros(outSession.shape)
nsession = len(sessionNames)
for si in range(nsession):
    imgs4reg[si,:,:] = clahe_each(outSession[si,:,:], kernel_size = regKernel)
    imgs4corr[si,:,:] = clahe_each(outSession[si,:,:], kernel_size = corrKernel)

useY = min(imgs4reg.shape[1], zstack4reg.shape[1]-estY)
useX = min(imgs4reg.shape[2], zstack4reg.shape[2]-estX)
imgs4regTrim = imgs4reg[:,:useY, :useX].copy()
imgs4corrTrim = imgs4corr[:,:useY, :useX].copy()

zstack4regTrim = zstack4reg[zrange,estY:estY+useY, estX:estX+useX].copy()
zstack4corrTrim = zstack4corr[zrange,estY:estY+useY, estX:estX+useX].copy()
srBi = StackReg(StackReg.BILINEAR)
bestCorrVals, bestTforms, corrVals, tforms = corr_match_to_best_z(imgs4reg=imgs4regTrim, imgs4corr=imgs4corrTrim, 
                                zstack4reg=zstack4regTrim, zstack4corr=zstack4corrTrim, stackReg = srBi)
# corrVals, tforms  = corr_match_to_best_z_suite2p(imgs4reg=imgs4regTrim, imgs4corr=imgs4corrTrim, 
#                                zstack4reg=zstack4regTrim, zstack4corr=zstack4corrTrim, op = op)

maxCorrVals = np.amax(corrVals, axis=1)
bestZsession = np.argmax(bestCorrVals, axis=1) + max(0, estZ-zthickness//2)

fig, ax = plt.subplots()
ax.plot(np.amax(bestCorrVals, axis=1), label='max corr')
ax.plot(maxCorrVals, label='each corr')
ax.legend()
ax.set_title(f'JK{mouse:03} plane {pn}\nBest correlation values')
ax.set_ylabel('Correlation')
ax.set_xlabel('Session index')

fig, ax = plt.subplots()
ax.plot([0, nsession], [estZ, estZ], '--', colors=(0.6, 0.6, 0.6))
ax.plot(bestZsession)
ax.set_title(f'JK{mouse:03} plane {pn}\nBest registered depth')
ax.set_ylabel(r'Best-matched depth ($\mu$m)')
ax.set_xlabel('Session index')

#%% Look at the curves in each sesion
if nsession <19:
    numRow = 3
elif nsession <25:
    numRow = 4
else:
    numRow = 5
numCol = nsession//numRow + int(np.ceil(numRow/(nsession%numRow)))

fig, ax = plt.subplots(numRow,numCol)
for si in range(nsession):
    pxi = si%numCol
    pyi = si//numCol
    ax[pyi,pxi].plot(corrVals[si,:])
    ax[pyi,pxi].set_title(f'Session #{si}')

fig.suptitle(f'JK{mouse:03} plane {pn}')
#%%
fig.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.9, hspace=0.5)



'''
Proceed to within-session z-drift 
only if registration works well.
For now, (comparing between plane 1 and plane 4)
select good session and proceed.
'''
#%% Within-session z-drift
selectedSi = [i for i in range(nsession)]

maxCorrList = []
corrDriftList = []
zdriftList = []
zflatten = np.array([img.flatten() for img in zstack4corrTrim])
for si in selectedSi:
    corrSession = np.zeros((numBinList[si],zthickness))
    maxCorrSession = np.zeros(numBinList[si])
    zdrift = np.zeros(numBinList[si])
    # if numBinList[si] > 3: # only for sessions longer than 30 min
    sessionImgs = [srRigid.transform(img[topMargin:bottomMargin, leftMargin:rightMargin],
                                       tmat = tformSessionRigid[si,:,:]) for img in sessionImgsDivList[si]]
    sessionImgArrTmp = np.array([outSession[si,:,:], *sessionImgs])
    sessionImgArrReged = srRigid.register_transform_stack(sessionImgArrTmp, reference='first')
    sessionImgArr = sessionImgArrReged[1:,:useY,:useX]
    for bi in range(numBinList[si]):
        binImg = sessionImgArr[bi,:,:].copy()
        binImgReged = srBi.transform(binImg, tmat = bestTforms[si])
        binImgCorr = clahe_each(binImgReged, kernel_size = corrKernel)
        corrValTmp = np.corrcoef(binImgCorr.flatten(), zflatten)[0,1:]
        corrSession[bi,:] = corrValTmp
    maxCorrSession = np.amax(corrSession, axis=1)
    zdrift = np.argmax(corrSession, axis=1)
    maxCorrList.append(maxCorrSession)
    corrDriftList.append(corrSession)
    zdriftList.append(zdrift)



#%%
#% Total drift across sessions
fig, ax = plt.subplots()
maxLength = max(numBinList)
for i, si in enumerate(selectedSi):
    zdrift = absDepthVal[zdriftList[i].astype(int) + max(0, estZ-zthickness//2)]
    numBin = numBinList[si]
    if numBin > 1:
        ax.plot([si+bn/10 for bn in range(numBin)], zdrift, 'k-', linewidth = 5)
    elif numBin == 1:
        ax.plot(si, zdrift[0], 'k.', markersize = 10)
    else:
        continue
ax.set_xlabel('Session index')
ax.set_ylabel('Estimated depth ($\mu$m)')
ax.set_title(f'JK{mouse:03} plane {pn}')

#%% Look at max correlation values in each session
if nsession <19:
    numRow = 3
elif nsession <25:
    numRow = 4
else:
    numRow = 5
    
if nsession % numRow > 0:
    numCol = nsession//numRow + 1
else:
    numCol = nsession//numRow
fig, ax = plt.subplots(numRow,numCol)
for i, si in enumerate(selectedSi):
    pxi = si%numCol
    pyi = si//numCol
    for bi in range(numBinList[si]):
        ax[pyi,pxi].plot(corrDriftList[i][bi,:])
    ax[pyi,pxi].set_title(f'Session #{si}')
fig.suptitle(f'JK{mouse:03} plane {pn}')
#%%
fig.tight_layout(pad=0.2)
plt.subplots_adjust(top=0.9, hspace=0.6)

#%% Save the result

zdriftResult = {}
zdriftResult['info'] = {}
zdriftResult['info']['mouse'] = mouse
zdriftResult['info']['plane'] = pn
zdriftResult['info']['nsession'] = nsession
zdriftResult['info']['sessionNames'] = sessionNames
zdriftResult['info']['numAvg'] = numAvg
zdriftResult['info']['regKernel'] = regKernel
zdriftResult['info']['corrKernel'] = corrKernel
zdriftResult['info']['topMargin'] = topMargin
zdriftResult['info']['bottomMargin'] = bottomMargin
zdriftResult['info']['leftMargin'] = leftMargin
zdriftResult['info']['rightMargin'] = rightMargin
zdriftResult['info']['estX'] = estX
zdriftResult['info']['estY'] = estY
zdriftResult['info']['estZ'] = estZ
zdriftResult['info']['zthickness'] = zthickness
zdriftResult['info']['selectedSi'] = selectedSi

zdriftResult['zstackAvg'] = zstackAvg
zdriftResult['sessionImgsDivList'] = sessionImgsDivList

zdriftResult['session'] = {}
zdriftResult['session']['bestTforms'] = bestTforms
zdriftResult['session']['bestCorrVals'] = bestCorrVals
zdriftResult['session']['corrVals'] = corrVals
zdriftResult['session']['tforms'] = tforms

zdriftResult['corrDriftList'] = corrDriftList
zdriftResult['maxCorrList'] = maxCorrList
zdriftResult['zdriftList'] = zdriftList

savefn = f'{h5Dir}JK{mouse:03}_zdrift_plane{pn}.npy'
np.save(savefn, zdriftResult, allow_pickle = True)














'''
Bilinear registration does not work well with plane 4 (for JK025 at least)
'''
#%% Check plane-to-plane correlation values
# See if these get higher in lower planes

offset = 10
fr2frCorr = np.zeros((nplane-offset, numNAL))
fig, ax = plt.subplots()
for ina in range(numNAL):
    zstack4corrTemp = zstack4corrList[ina][:,20:-20,20:-20]
    Ly = zstack4corrTemp.shape[1]
    Lx = zstack4corrTemp.shape[2]
    zFlatten = np.zeros((nplane, Ly*Lx))
    for pi in range(nplane):
        zFlatten[pi,:] = zstack4corrTemp[pi,:,:].flatten()
    fr2frCorr[:,ina] = np.corrcoef(zFlatten).diagonal(offset=offset)
    na = numAvgList[ina]
    ax.plot(fr2frCorr[:,ina], label=f'# of Average: {na}')
ax.legend()
ax.set_ylabel('Correlation')
ax.set_title(f'Offset {offset}')

    
    
'''
Plane-to-plane correlation decreases as you go deeper.
Higher correlation with higher # of averages (of course).
'''



'''
Kernel sizes should be calculated in each plane.
How about using suite2p instead? (after rigid registration using stackreg)
'''
#%% Use suite2p registration
# first, try without clahe

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['snr_thresh'] = 1.2
op['block_size'] = [128, 128]
op['maxregshiftNR'] = min(op['block_size'])//10




#%% Before going into full session and numAvgs, 
# Try for one good session (Reference session)
# with 20 numAvg
numAvg = 20
useY = min(refSessionMimg.shape[0], zstackReg.shape[1]-estY)
useX = min(refSessionMimg.shape[1], zstackReg.shape[2]-estX)
refSessionMimg = outSession[refSi,:,:]

refTrim = refSessionMimg[:useY, :useX].copy()
# ref4corrTrim = clahe_each(refTrim, kernel_size = corrKernel)

zthickness = 50
zrange = range(max(0,estZ-zthickness//2), min(nplane, estZ+zthickness//2))


kernelList = [50,100,200]
# regKernelList = [50,100,200]
# corrKernelList = [50,100,200]
zstackAvgTmp = np.zeros(zstackReg.shape)
for pi in range(nplane):
    zstackAvgTmp[pi,:,:] = np.mean(zstackReg[max(0, pi-numAvg//2):min(nplane, pi+numAvg//2)], axis=0)
zstackTrim = zstackAvgTmp[zrange, estY:estY+useY, estX:estX+useX]

zstackList = []
refTrimList = []
for kernelSize in kernelList:
    zstackTmp = np.zeros(zstackTrim.shape)
    for pi in range(zthickness):
        zstackTmp[pi,:,:] = clahe_each(zstackTrim[pi,:,:], kernel_size = kernelSize)
    zstackList.append(zstackTmp)
    refTrimList.append(clahe_each(refTrim, kernel_size = kernelSize))
#%%
framesList = []
corrValsList = []
for rki in range(len(kernelList)):
    z4reg = zstackList[rki]
    ref4reg = refTrimList[rki].copy()
    for cki in range(len(kernelList)):
        z4corr = zstackList[cki]
        ref4corr = refTrimList[cki]
        frames = []
        corrVals = np.zeros(zthickness)
        for pi in range(zthickness):
            frame, rigid_offsets, nonrigid_offsets = s2p_nonrigid_registration([ref4reg], z4reg[pi,:,:], op)
            frames.append(frame)
            frame4corr = s2p_register([ref4corr], op, rigid_offsets, nonrigid_offsets = nonrigid_offsets )
            
            corrVals[pi] = np.corrcoef(frame4corr.flatten(), z4corr[pi,:,:].flatten())[0,1]
        corrValsList.append(corrVals)

#%%

fig, ax = plt.subplots(1,len(kernelList), figsize=(5*len(kernelList), 5))
for rki in range(len(kernelList)):
    regKernel = kernelList[rki]
    for cki in range(len(kernelList)):
        corrKernel = kernelList[cki]
        ax[rki].plot(corrValsList[len(kernelList)*rki+cki], label=f'corr kernel {corrKernel}')
    if rki==0:
        ax[rki].legend()
        ax[rki].set_ylabel('Correlation')
        ax[rki].set_xlabel('Relative plane depth')
    ax[rki].set_title(f'reg kernel {regKernel}')
# zstackTemp = zstackAvgList[ina]
# # zstack4regTemp = zstack4regList[ina]
# zstack4corrTemp = zstack4corrList[ina]
# zstackTrim = zstackTemp[zrange,estY:estY+useY, estX:estX+useX].copy()
# # zstack4regTrim = zstack4regTemp[zrange,estY:estY+useY, estX:estX+useX].copy()
# zstack4corrTrim = zstack4corrTemp[zrange,estY:estY+useY, estX:estX+useX].copy()
# z4corrFlat = np.vstack([zimg.flatten() for zimg in zstack4corrTrim])

# roffsetList = []
# nroffsetList = []
# framesList = []
# corrVals = np.zeros(zthickness)    
# for pi, refImg in enumerate(zstackTrim):
#     frames, rigid_offsets, nonrigid_offsets = s2p_nonrigid_registration([refTrim], refImg, op)
#     roffsetList.append(rigid_offsets)
#     nroffsetList.append(nonrigid_offsets)
#     framesList.append(frames)
#     frame4corr = clahe_each(frames, kernel_size = corrKernel)
#     z4corr = clahe_each(refImg, kernel_size = corrKernel)
#     corrVals[pi] = np.corrcoef(frame4corr.flatten(), z4corr.flatten())[0,1]
#%%
maxCorrInd = np.argmax(corrVals)
maxRoffset = roffsetList[maxCorrInd]
maxNRoffset = nroffsetList[maxCorrInd]
regImg = framesList[maxCorrInd]
regImg4corr = clahe_each(regImg, kernel_size = corrKernel)
regCorrVals = np.zeros(zthickness)
for pi, refImg in enumerate(zstackTrim):


#%%

napari.view_image(np.array(framesList))

#%%

numAvgList= [5, 10, 20]


# zstackAvgList = []
# zstack4regList = []
# zstack4corrList = []
# nplane = zstackReg.shape[0]

# for numAvg in numAvgList:
#     print(f'Averaging with {numAvg}')
#     t1 = time.time()
#     zstackAvgTemp = np.zeros(zstackReg.shape)
#     zstack4regTemp = np.zeros(zstackReg.shape)
#     zstack4corrTemp = np.zeros(zstackReg.shape)
#     for pi in range(nplane):
#         zstackAvgTemp[pi,:,:] = np.mean(zstackReg[max(0, pi-numAvg//2):min(nplane, pi+numAvg//2)], axis=0)
#         zstack4regTemp[pi,:,:] = clahe_each(zstackAvgTemp[pi,:,:], kernel_size = regKernel)
#         zstack4corrTemp[pi,:,:] = clahe_each(zstackAvgTemp[pi,:,:], kernel_size = corrKernel)
#     zstackAvgList.append(zstackAvgTemp)
#     zstack4regList.append(zstack4regTemp)
#     zstack4corrList.append(zstack4corrTemp)
# print(f'Zstack averaging done.')

zthickness = 50

# imgs4reg = np.zeros(outSession.shape)
# imgs4corr = np.zeros(outSession.shape)
# nsession = len(sessionNames)
# for si in range(nsession):
#     imgs4reg[si,:,:] = clahe_each(outSession[si,:,:], kernel_size = regKernel)
#     imgs4corr[si,:,:] = clahe_each(outSession[si,:,:], kernel_size = corrKernel)

useY = min(imgs4reg.shape[1], zstack4reg.shape[1]-estY)
useX = min(imgs4reg.shape[2], zstack4reg.shape[2]-estX)
imgsTrim = outSession[:,:useY, :useX].copy()
# imgs4regTrim = imgs4reg[:,:useY, :useX].copy()
imgs4corrTrim = imgs4corr[:,:useY, :useX].copy()
zrange = range(max(0,estZ-zthickness//2), min(nplane, estZ+zthickness//2))

corrValsList = []
tformsList = []
bestZsessionList = []
numNAL = len(numAvgList)
zstackTemp = zstackAvgList[ina]
# zstack4regTemp = zstack4regList[ina]
zstack4corrTemp = zstack4corrList[ina]
zstackTrim = zstackTemp[zrange,estY:estY+useY, estX:estX+useX].copy()
# zstack4regTrim = zstack4regTemp[zrange,estY:estY+useY, estX:estX+useX].copy()
zstack4corrTrim = zstack4corrTemp[zrange,estY:estY+useY, estX:estX+useX].copy()
z4corrFlat = np.vstack([zimg.flatten() for zimg in zstack4corrTrim])
for ina in range(numNAL):
    print(f'Procesing registration for depth {ina}/{numNAL-1}')
    t1 = time.time()

    # corrVals, tforms  = corr_match_to_best_z(imgs4reg=imgs4regTrim, imgs4corr=imgs4corrTrim, 
    #                                zstack4reg=zstack4regTrim, zstack4corr=zstack4corrTrim, stackReg = srBi)
    
    # corrVals, tforms  = corr_match_to_best_z(imgs4reg=imgs4regTrim, imgs4corr=imgs4corrTrim, 
    #                                zstack4reg=zstack4regTrim, zstack4corr=zstack4corrTrim, stackReg = srRigid)
    roffsetList = []
    nroffsetList = []
    framesList = []
    for refImg in zstackTrim:
        frames, rigid_offsets, nonrigid_offsets = s2p_nonrigid_registration(imgs4corrTrim, refImg, op)
        roffsetList.append(rigid_offsets)
        nroffsetList.append(nonrigid_offsets)
        framesList.append(frames)
    
    corrVals = np.zeros((nsession,zthickness))
    for si in range(nsession):
        tempImgs4corr = [clahe_each(frlist[si,:,:], kernel_size = corrKernel) for frlist in framesList]
        tempFlat = np.vstack([img.flatten() for img in tempImgs4corr])
        tempCorr = np.zeros(zthickness)
        for pi in range(zthickness):
            tempCorr[pi] = np.corrcoef(tempFlat[pi,:], z4corrFlat[pi,:])[0,1]
        bestPi = np.argmax(tempCorr)
        bestRigidOffset = [ros[si] for ros in roffsetList[bestPi][0]][:2]
        bestNonrigidOffset = [ros[si] for ros in nroffsetList[bestPi][0]][:2]
        bestFrame = framesList[bestPi][si,:,:]
        bestFrameCorr = tempImgs4corr[bestPi]
        corrVals[si,:] = np.corrcoef(np.vstack([tempFlat[bestPi,:], z4corrFlat]))[0,1:]
    
    corrValsList.append(corrVals)
    tformsList.append(tforms)
    bestZsession = np.argmax(corrVals, axis=1) + max(0, estZ-zthickness//2)
    bestZsessionList.append(bestZsession)
    t2 = time.time()
    tpass = round((t2-t1)/60)
    print(f'{tpass} min.')

fig, ax = plt.subplots()
for ina in range(len(numAvgList)):
    na = numAvgList[ina]
    bestZsession = bestZsessionList[ina]
    ax.plot(absDepthVal[bestZsession], label=f'avg: {na}')
ax.legend()
ax.set_title(f'JK{mouse:03} plane {pn}\nMatched z across sessions')
ax.set_ylabel('Matched depth ($\mu$m)')
ax.set_xlabel('Session index')

















































#%% Confirm the z matching in each session 
#%%

#%%
si = 0
fig, ax = plt.subplots()
ax.plot(zrange, corrVals[si,:])
ax.set_title(f'session #{si}')




#%% Test z-drift in each session. 
time0 = time.time()
corrValsDriftNA = [] # NA for number of averaging
zdriftAllNA = []
# trimMimgDiv = [imgs[:, topMargin:bottomMargin, leftMargin:rightMargin] for imgs in sessionImgsDivList] # why doesn't this work?

for ina in range(numNAL):
    time1 = time.time()
    zstack4corrTemp = zstack4corrList[ina]
    zstack4corrTrim = zstack4corrTemp[zrange,estY:estY+useY, estX:estX+useX].copy()
    zcorrFlatten = np.reshape(zstack4corrTrim, (zthickness,-1))
    tforms = tformsList[ina]
    corrValsDrift = []
    zdriftAll = []
    for si in range(nsession):
        print(f'Running {si}/{nsession-1}')
        time2 = time.time()
        imgs = sessionImgsDivList[si]
        mimgDivTempReg = [srRigid.transform(img, tmat = tformSessionRigid[si,:,:]) for img in imgs]
        mimgDivTempCorr = [clahe_each(img[topMargin:topMargin+useY, leftMargin:leftMargin+useX], kernel_size = corrKernel) 
                           for img in mimgDivTempReg]
        tformTemp = tforms[si]
        outDivTempCorr = [srBi.transform(img, tmat=tformTemp).flatten() for img in mimgDivTempCorr]
        
        corrValsTemp = []
        numDivTemp = len(outDivTempCorr)
        bestZtemp = np.zeros(numDivTemp)
        for di in range(numDivTemp):
            cv = np.corrcoef(outDivTempCorr[di], zcorrFlatten)[0,1:]
            corrValsTemp.append(cv)
            bestZtemp[di] = np.argmax(cv) + max(0, estZ-zthickness//2)
        corrValsDrift.append(corrValsTemp)
        zdriftAll.append(bestZtemp)
        
        time3 = time.time()
        elapsedMin = int(time3-time2)//60
        elapsedSec = int(time3-time2)%60
        print(f'Session {si} done in {elapsedMin} min {elapsedSec} s.')
    
    corrValsDriftNA.append(corrValsDrift)
    zdriftAllNA.append(zdriftAll)
    totMin = int(time3 - time1) // 60
    totSec = int(time3 - time1) % 60
    print(f'Single average {totMin} min {totSec} s passed.')

totMin = int(time3 - time0) // 60
totSec = int(time3 - time0) % 60
print(f'Total {totMin} min {totSec} s passed.')


#%%
fig, ax = plt.subplots()
for ina in range(numNAL):
    na = numAvgList[ina]
    tempZ = []
    for si in range(nsession):
        tempZ = np.hstack((tempZ, zdriftAll[si].astype(int))).astype(int)
    ax.plot(absDepthVal[tempZ], label=f'numAvg = {na}')
ax.legend()
ax.set_title('z-drift')

#%% Randomly select some sessions and compare correlation values between avg numbers
# First, take a look at them, and then, just compare peak
# From session mimg, and also from 10 min mimgs
siList = [0,3,6,9,12,15]
fig, ax = plt.subplots(2,3)
for i in range(6):
    si = siList[i]
    pxi = i%3
    pyi = i//3
    for ani in range(numNAL):
        na = numAvgList[ani]
        ax[pyi,pxi].plot(corrValsList[ani][si,:], label=f'numAvg = {na}')
    ax[pyi,pxi].set_title(f'Session #{si}')
    if i == 0:
        ax[pxi, pyi].legend()
fig.tight_layout()









