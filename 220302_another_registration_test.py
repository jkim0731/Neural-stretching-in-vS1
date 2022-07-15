# -*- coding: utf-8 -*-
"""
Registration test.
Compare with previous registration using suite2p 2-step non-rigid registration.
Test in JK025 planes 5-8 and JK039 and JK052.

Try other parameters for suite2p 2-step non-rigid registration (s2nr).
Try combination (StackReg first, and then s2nr)
Try using ROI map instead.

Use pixel-to-pixel correlation (with edge removed) for comparison.

2022/03/02 JK
"""

import numpy as np
from matplotlib import pyplot as plt
from suite2p.registration import rigid, nonrigid, utils, register

from pystackreg import StackReg
import os, glob
import napari
from suite2p.io.binary import BinaryFile
from skimage import exposure
import gc
gc.enable()

# h5Dir = 'D:/TPM/JK/h5/'
h5Dir = 'D:/'

mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]


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

def s2p_2step_nr(mimgList, refMimg, op):
    op['block_size'] = [op['block_size_list'][0], op['block_size_list'][0]]
    mimgList1step, roff1, nroff1 = s2p_nonrigid_registration(mimgList, refMimg, op)
    op['block_size'] = [op['block_size_list'][1], op['block_size_list'][1]]
    mimgList2step, roff2, nroff2 = s2p_nonrigid_registration(mimgList1step, refMimg, op)
    return mimgList2step, roff1, roff2, nroff1, nroff2

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
    
    # mean_img = np.zeros((Ly, Lx))
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
        maxregshiftNR=op['block_size'][0]//10,
    )

    frames2 = nonrigid.transform_data(
        data=frames,
        nblocks=nblocks,
        xblock=xblock,
        yblock=yblock,
        ymax1=ymax1,
        xmax1=xmax1,
    )

    nonrigid_offsets.append([ymax1, xmax1, cmax1])
    
    return frames2, rigid_offsets, nonrigid_offsets

def get_session_names(baseDir, mouse, planeNum):
    tempFnList = glob.glob(f'{baseDir}{mouse:03}_*_plane_{planeNum}.h5')
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

def phase_corr(fixed, moving, transLim = 0):
    # apply np.roll(moving, (ymax, xmax), axis=(0,1)) to match moving to fixed
    # or, np.roll(fixed, (-ymax, -xmax), axis=(0,1))
    if fixed.shape != moving.shape:
        raise('Dimensions must match')
    R = np.fft.fft2(fixed) * np.fft.fft2(moving).conj()
    R /= np.absolute(R)
    r = np.absolute(np.fft.ifft2(R))
    if transLim > 0:
        r = np.block([[r[-transLim:,-transLim:], r[-transLim:, :transLim+1]],
                     [r[:transLim+1,-transLim:], r[:transLim+1, :transLim+1]]]
            )
        ymax, xmax = np.unravel_index(np.argmax(r), (transLim*2 + 1, transLim*2 + 1))
        ymax, xmax = ymax - transLim, xmax - transLim
        cmax = np.amax(r)
        center = r[transLim, transLim]
    else:
        ymax, xmax = np.unravel_index(np.argmax(r), r.shape)
        cmax = np.amax(r)
        center = r[0, 0]
    return ymax, xmax, cmax, center, r

def gather_images(h5Dir, mouse, pn, sn):
    opsfn = f'{h5Dir}{mouse:03}/plane_{pn}/{sn:03}/plane0/ops.npy'
    ops = np.load(opsfn, allow_pickle=True).item()
    mimg = ops['meanImg']
    if 'meanImgE' in ops.keys():
        emimg = ops['meanImgE']
    else:
        ops = register.enhanced_mean_image(ops)
        emimg = ops['meanImgE']
        np.save(opsfn, ops)
    
    stat = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/{sn:03}/plane0/stat.npy', allow_pickle=True)
    iscell = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/{sn:03}/plane0/iscell.npy', allow_pickle=True)
    celli = np.where(iscell[:,0])[0]
    numCell = len(celli)
    cellMap = np.zeros(mimg.shape, 'bool')
    for ci in celli:
        for pixi in range(len(stat[ci]['ypix'])):
            xi = stat[ci]['xpix']
            yi = stat[ci]['ypix']
            cellMap[yi,xi] = 1
    return mimg, emimg, cellMap

def clahe_each(img: np.float64, kernel_size = None, clip_limit = 0.01, nbins = 2**16):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    newimg = exposure.equalize_adapthist(newimg, kernel_size = kernel_size, clip_limit = clip_limit, nbins=nbins)    
    return newimg

def clahe_multi(img, kernel_size = None, clip_limit = 0.01, nbins = 2**16):
    if len(img.shape)!= 3:
        raise('Dimension should be 3.')
    else:
        newImg = img.copy()
        for i in range(newImg.shape[0]):
            newImg[i,:,:] = clahe_each(newImg[i,:,:], kernel_size, clip_limit, nbins)
    return newImg











#%%
#%% (1) Rigid body first, and then s2p nr. Either 1-step or 2-step.

#%%
#%% Selecting sessions
mi = 6
mouse = mice[mi]
refSn = refSessions[mi]

vi = 5 # volume index, either 1 or 5

# Load z-drift data
zdrift = np.load(f"{h5Dir}JK{mouse:03}_zdrift_plane{vi}.npy", allow_pickle=True).item()

# Select training sessions only
# Re-order sessions if necessary
siArr = np.where([len(sn.split('_'))==2 for sn in zdrift['info']['sessionNames']])[0]
snums = np.array([int(sn.split('_')[1]) for sn in zdrift['info']['sessionNames'] if len(sn.split('_'))==2])
siSorted = siArr[np.argsort(snums)]

# Set depths (relative value)
# selDepthsRV = [27,37] # JK052 upper
# selDepthsRV = [17,27] # JK052 lower
# selDepthsRV = [7,17] # JK025 upper
# selDepthsRV = [18,28] # JK025 lower
# selDepthsRV = [20,30] # JK027 upper
# selDepthsRV = [25,35] # JK027 lower
# selDepthsRV = [17,27] # JK030 upper
# selDepthsRV = [22,32] # JK030 lower
# selDepthsRV = [16,26] # JK036 upper
# selDepthsRV = [12,22] # JK036 lower
# selDepthsRV = [22,32] # JK039 upper
selDepthsRV = [17,27] # JK039 lower

# Select sessions
selectedSi = np.array([si for si in siSorted if \
              sum(np.logical_and(zdrift['zdriftList'][si]>=selDepthsRV[0], zdrift['zdriftList'][si]<=selDepthsRV[1])) >=3 ])
selectedSnums = [int(sname.split('_')[1]) for sname in np.array(zdrift['info']['sessionNames'])[selectedSi]]

# Manually removed session indice
# manRmvSi = np.array([7,14]) # the index is from selectedSi, not from all the sessions
# manRmvSi = np.array([]) # JK052 upper
# manRmvSi = np.array([]) # JK052 lower
# manRmvSi = np.array([7,14]) # JK025 upper
# manRmvSi = np.array([]) # JK025 lower
# manRmvSi = np.array(range(7,12)) # JK027 upper
# manRmvSi = np.array([]) # JK027 lower
# manRmvSi = np.array([]) # JK030 upper
# manRmvSi = np.array([19]) # JK030 lower
# manRmvSi = np.array([17]) # JK036 upper
# manRmvSi = np.array([19]) # JK036 lower
# manRmvSi = np.array([]) # JK039 upper
manRmvSi = np.array([]) # JK039 lower

if len(manRmvSi)>0:
    selectedSi = np.delete(selectedSi, manRmvSi)
    selectedSnums = np.delete(selectedSnums, manRmvSi)









#%%
#%%
#%% Gather ref images
# # for pn in range(vi,vi+4):
# pn = vi+0 # +0, +1, +2, +3

# planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
# regFn = f'{planeDir}s2p_nr_reg.npy'
# reg = np.load(regFn, allow_pickle=True).item()

# numSession = len(selectedSnums)

# refMimg, refEmimg, refCellmap = gather_images(h5Dir, mouse, pn, refSn)


# #% Run registration
# leftBuffer = 30
# rightBuffer = 30 if mouse < 50 else 100
# bottomBuffer = 10
# topBuffer = 50

# resultX = refMimg.shape[1] - (leftBuffer + rightBuffer)
# resultY = refMimg.shape[0] - (topBuffer + bottomBuffer)

# regResultMimg = np.zeros((numSession, resultY, resultX))
# regResultEmimg = np.zeros_like(regResultMimg)
# regResultCellmap = np.zeros_like(regResultMimg)
# regImgs = reg['regImgs'][selectedSi,topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]

# srRigid = StackReg(StackReg.RIGID_BODY)
# op = {'smooth_sigma': 1.15, 'maxregshift': 0.3, 'smooth_sigma_time': 0, 'snr_thresh': 1.2, 'block_size_list': [128,32]}

# refMimg2 = refMimg[topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
# refEMimg2 = refEmimg[topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
# refCellmap2 = refCellmap[topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
# refMimgClip2 = np.clip(refMimg2, np.percentile(refMimg2,1), np.percentile(refMimg2,99))

# for si, sn in enumerate(selectedSnums):
# # si = 0
# # sn = 1

#     tempMimg, tempEmimg, tempCellmap = gather_images(h5Dir, mouse, pn, sn)
    
#     tempMimg2 = refMimg[topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
#     tempEMimg2 = refEmimg[topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
#     tempCellmap2 = refCellmap[topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
#     tempMimgClip2 = np.clip(tempMimg2, np.percentile(tempMimg2,1), np.percentile(tempMimg2,99))
    
#     # Running using Mimg
#     rotMimg = srRigid.register_transform(refMimg, tempMimg)
#     rotMimg2 = rotMimg[topBuffer:-bottomBuffer, leftBuffer:-rightBuffer]
#     regResultMimg[si,:,:] = s2p_2step_nr([rotMimg2], refMimg2, op)[0]
    
#     # Running using MimgClip
#     rotMimg = srRigid.register_transform(refMimg, tempMimg)
#     rotMimg2 = rotMimg[topBuffer:-bottomBuffer, leftBuffer:-rightBuffer]
#     regResultMimg[si,:,:] = s2p_2step_nr([rotMimg2], refMimgClip2, op)[0]
    
    
#     # Running using Emimg
#     tform = srRigid.register(refEmimg, tempEmimg)
#     rotEmimg = srRigid.transform(tempEmimg, tmat=tform)
#     rotEmimg2 = rotEmimg[topBuffer:-bottomBuffer, leftBuffer:-rightBuffer]
#     _, roff1, roff2, nroff1, nroff2 = s2p_2step_nr([rotEmimg2], refEMimg2, op)
    
#     rigid_y1 = roff1[0][0][0]
#     rigid_x1 = roff1[0][1][0]
#     nonrigid_y1 = nroff1[0][0]
#     nonrigid_x1 = nroff1[0][1]
    
#     rigid_y2 = roff2[0][0][0]
#     rigid_x2 = roff2[0][1][0]
#     nonrigid_y2 = nroff2[0][0]
#     nonrigid_x2 = nroff2[0][1]
    
#     rotMimg = srRigid.transform(tempMimg, tmat=tform)
#     rotMimg2 = rotMimg[topBuffer:-bottomBuffer, leftBuffer:-rightBuffer]
#     regResultEmimg[si,:,:] = twostep_register(rotMimg2, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, [op['block_size_list'][0],op['block_size_list'][0]], 
#                       rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, [op['block_size_list'][1], op['block_size_list'][1]])
        
#     # Running using cellmap   
#     tform = srRigid.register(refCellmap, tempCellmap)
#     rotCellmap = srRigid.transform(tempCellmap, tmat=tform)
#     rotCellmap2 = rotCellmap[topBuffer:-bottomBuffer, leftBuffer:-rightBuffer]
#     _, roff1, roff2, nroff1, nroff2 = s2p_2step_nr([rotCellmap2], refCellmap2, op)
    
#     rigid_y1 = roff1[0][0][0]
#     rigid_x1 = roff1[0][1][0]
#     nonrigid_y1 = nroff1[0][0]
#     nonrigid_x1 = nroff1[0][1]
    
#     rigid_y2 = roff2[0][0][0]
#     rigid_x2 = roff2[0][1][0]
#     nonrigid_y2 = nroff2[0][0]
#     nonrigid_x2 = nroff2[0][1]
    
#     rotMimg = srRigid.transform(tempMimg, tmat=tform)
#     rotMimg2 = rotMimg[topBuffer:-bottomBuffer, leftBuffer:-rightBuffer]
#     regResultCellmap[si,:,:] = twostep_register(rotMimg2, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, [op['block_size_list'][0],op['block_size_list'][0]], 
#                       rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, [op['block_size_list'][1], op['block_size_list'][1]])
        
# # Show the result
# viewer = napari.Viewer()
# viewer.add_image(regImgs, name='s2p 2step nr')
# viewer.add_image(regResultMimg, name='After rotation, Using mimg')
# viewer.add_image(regResultEmimg, name='After rotation, Using Emimg')
# viewer.add_image(regResultCellmap, name='After rotation, Using cellmap')

# #%% Plot pixel value correlation (to the reference image)



# #%% Compare bewteen two s2p 2step nr
# # For JK052, where the second registration is made by setting reference session as the later one. S021


# pn = vi+1

# leftBuffer = 30
# rightBuffer = 30 if mouse < 50 else 100
# bottomBuffer = 10
# topBuffer = 50

# planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
# reg1Fn = f'{planeDir}s2p_nr_reg_bu1.npy'
# reg1 = np.load(reg1Fn, allow_pickle=True).item()
# regImgs1 = reg1['regImgs'][selectedSi,topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]

# reg2Fn = f'{planeDir}s2p_nr_reg.npy'
# reg2 = np.load(reg2Fn, allow_pickle=True).item()
# regImgs2 = reg2['regImgs'][selectedSi,topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]

# viewer = napari.Viewer()
# viewer.add_image(regImgs1, name='s2p 2step nr 1')
# viewer.add_image(regImgs2, name='s2p 2step nr 2')


# fix, ax = plt.subplots()
# ax.plot(np.array(reg1['corrVals'])[selectedSi], label='old')
# ax.plot(np.array(reg2['corrVals'])[selectedSi], label='new')

# ax.legend()
# ax.set_xlabel('Session index (Selected only)')
# ax.set_ylabel('Correlation to the reference (after CLAHE)')
# ax.set_title(f'JK{mouse:03} plane {pn}')

# #%% Quantify this with correlation matrix.
# # CLAHE before calculating correlation.
# numSession = len(selectedSnums)
# claheImgs1 = clahe_multi(regImgs1)
# claheImgs2 = clahe_multi(regImgs2)

# corr1 = np.corrcoef(np.reshape(claheImgs1, (numSession, -1)))
# corr2 = np.corrcoef(np.reshape(claheImgs2, (numSession, -1)))

# fig, ax = plt.subplots(1,3, figsize=(10,3))
# im = ax[0].imshow(corr1, vmin=0.75, vmax=1)
# ax[0].set_title('Old registration')
# plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
# im = ax[1].imshow(corr2, vmin=0.75, vmax=1)
# ax[1].set_title('New registration')
# plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
# im = ax[2].imshow(corr2-corr1)
# plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
# ax[2].set_title('New - Old (difference)')
# fig.suptitle(f'JK{mouse:03} plane {pn}')
# fig.subplots_adjust(wspace=0.5, hspace=0.001)




#%% (2) Serial registration
# Register each mean image to the previous one (which again was registered one step before)
# Start with 1 preivous image. Some sessions are off (even after depth matching), so consider increasing # of previous images
# Use StackReg, CLAHE, and s2p 2-step nr

# #%% (2) collecting images

# pn = vi+3 # +0, +1, +2, +3

# print(f'Collecting images from plane {pn}')

# leftBuffer = 30
# rightBuffer = 30 if mouse < 50 else 100
# bottomBuffer = 10
# topBuffer = 50

# planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
# # regFn = f'{planeDir}s2p_nr_reg_bu1.npy'
# regFn = f'{planeDir}s2p_nr_reg.npy'
# reg = np.load(regFn, allow_pickle=True).item()
# regImgs = reg['regImgs'][selectedSi,topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]

# numSelected = len(selectedSi)
# mimgs = np.zeros_like(regImgs)
# mimgClahe = np.zeros_like(regImgs)
# for si, sn in enumerate(selectedSnums):
#     opsFn = f'{planeDir}{sn:03}/plane0/ops.npy'
#     ops = np.load(opsFn, allow_pickle=True).item()
#     mimgs[si,:,:] = ops['meanImg'][topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
#     mimgClahe[si,:,:] = clahe_each(mimgs[si,:,:])
    
# viewer = napari.Viewer()
# viewer.add_image(regImgs, name='s2p reg')
# viewer.add_image(mimgs, name='mean images')
# #%% (2) registration 
# # prevN = 1 # number of previous images to refer

# # srBi = StackReg(StackReg.BILINEAR)
# # # srRigid = StackReg(StackReg.RIGID_BODY)
# # srAffine = StackReg(StackReg.AFFINE)
# # regBi = np.zeros_like(mimgs)
# # regAffine = np.zeros_like(mimgs)
# # # regRigid = np.zeros_like(mimgs)
# # for si in range(numSelected):
# #     if si == 0:
# #         regBi[si,:,:] = mimgs[si,:,:]
# #         regAffine[si,:,:] = mimgs[si,:,:]
# #         # regRigid[si,:,:] = mimgs[si,:,:]
# #     else:
# #         previStart = max(0,si-prevN)
        
# #         refBi = np.mean(regBi[previStart:si,:,:], axis=0)
# #         regBi[si,:,:] = srBi.register_transform(refBi, mimgs[si,:,:])
        
# #         refAffine = np.mean(regAffine[previStart:si,:,:], axis=0)
# #         regAffine[si,:,:] = srAffine.register_transform(refAffine, mimgs[si,:,:])
        
# #         # refRigid = np.mean(regRigid[previStart:si,:,:], axis=0)
# #         # regRigid[si,:,:] = srRigid.register_transform(refRigid, mimgs[si,:,:])

# # viewer = napari.Viewer()
# # viewer.add_image(regImgs, name=f's2p plane {pn}')
# # # viewer.add_image(regRigid, name=f'serial Rigid {prevN}')
# # viewer.add_image(regAffine, name=f'serial Affine -{prevN} images')
# # viewer.add_image(regBi, name=f'serial Bilinear -{prevN} images')

# #%% (2) Using clahe
# '''
# Using CLAHE is better for bilinear. (it prevents weird wild transition)
# There is a little difference between bilinear and affine. 

# These registrations are far better than s2p nr registered to a reference session.
# '''
# prevN = 1

# srBi = StackReg(StackReg.BILINEAR)
# # srRigid = StackReg(StackReg.RIGID_BODY)
# srAffine = StackReg(StackReg.AFFINE)
# regBi = np.zeros_like(mimgs)
# regAffine = np.zeros_like(mimgs)
# # regRigid = np.zeros_like(mimgs)

# for si in range(numSelected):
#     if si == 0:
#         regBi[si,:,:] = mimgs[si,:,:]
#         regAffine[si,:,:] = mimgs[si,:,:]
#         # regRigid[si,:,:] = mimgs[si,:,:]
#     else:
#         previStart = max(0,si-prevN)
        
#         refBi = clahe_each(np.mean(regBi[previStart:si,:,:], axis=0))
#         tform = srBi.register(refBi, mimgClahe[si,:,:])
#         regBi[si,:,:] = srBi.transform(mimgs[si,:,:], tmat=tform)
        
#         refAffine = clahe_each(np.mean(regAffine[previStart:si,:,:], axis=0))
#         tform = srAffine.register(refAffine, mimgClahe[si,:,:])
#         regAffine[si,:,:] = srAffine.transform(mimgs[si,:,:], tmat=tform)
        
#         # refRigid = clahe_each(np.mean(regRigid[previStart:si,:,:], axis=0))
#         # srRigid.register(refRigid, mimgClahe[si,:,:])
#         # regRigid[si,:,:] = srRigid.transform(mimgs[si,:,:])

# viewer = napari.Viewer()
# viewer.add_image(regImgs, name=f's2p plane {pn}')
# # viewer.add_image(regRigid, name=f'serial Rigid clahe {prevN}')
# viewer.add_image(regAffine, name=f'serial Affine -{prevN} images')
# viewer.add_image(regBi, name=f'serial Bilinear -{prevN} images')


# #%% (2) Can tforms be saved?
# ''' yes.
# '''
# srBi = StackReg(StackReg.BILINEAR)
# srAffine = StackReg(StackReg.AFFINE)
# regBi = np.zeros_like(mimgs)
# regAffine = np.zeros_like(mimgs)

# tformsBi = []
# tformsAffine = []

# for si in range(numSelected):
#     if si == 0:
#         regBi[si,:,:] = mimgs[si,:,:]
#         regAffine[si,:,:] = mimgs[si,:,:]
#     else:
#         previStart = max(0,si-prevN)
        
#         refBi = clahe_each(np.mean(regBi[previStart:si,:,:], axis=0))
#         tform = srBi.register(refBi, mimgClahe[si,:,:])
#         tformsBi.append(tform)
#         regBi[si,:,:] = srBi.transform(mimgs[si,:,:], tmat=tform)
        
#         refAffine = clahe_each(np.mean(regAffine[previStart:si,:,:], axis=0))
#         tform = srAffine.register(refAffine, mimgClahe[si,:,:])
#         tformsAffine.append(tform)
#         regAffine[si,:,:] = srAffine.transform(mimgs[si,:,:], tmat=tform)

# srBi2 = StackReg(StackReg.BILINEAR)
# srAffine2 = StackReg(StackReg.AFFINE)
# regBi2 = np.zeros_like(mimgs)
# regAffine2 = np.zeros_like(mimgs)
# for si in range(numSelected):
#     if si == 0:
#         regBi2[si,:,:] = mimgs[si,:,:]
#         regAffine2[si,:,:] = mimgs[si,:,:]
#     else:
#         regBi2[si,:,:] = srBi2.transform(mimgs[si,:,:], tmat=tformsBi[si-1])
#         regAffine2[si,:,:] = srAffine2.transform(mimgs[si,:,:], tmat=tformsAffine[si-1])

# viewer = napari.Viewer()
# viewer.add_image(regImgs, name='s2p')
# viewer.add_image(regAffine2, name=f'serial Affine clahe {prevN}')
# viewer.add_image(regBi2, name=f'serial Bilinear clahe {prevN}')

# fig, ax = plt.subplots()
# ax.plot(np.sum(abs(regAffine-regAffine2), axis=(1,2)), label='affine')
# ax.plot(np.sum(abs(regBi-regBi2), axis=(1,2)), label='bilinear')
# ax.legend()

# #%% (2) s2p 2-step nonrigid serial registration
# # using mimg, collected from ops files
# op = {'smooth_sigma': 1.15, 'maxregshift': 0.3, 'smooth_sigma_time': 0, 'snr_thresh': 1.2, 'block_size_list': [128,32]}
# regS2p = np.zeros_like(mimgs)
# for si in range(numSelected):
#     if si == 0:
#         regS2p[si,:,:] = mimgs[si,:,:]
#     else:
#         previStart = max(0,si-prevN)
        
#         refImg = np.mean(regBi[previStart:si,:,:], axis=0)
#         regS2p[si,:,:] = s2p_2step_nr([mimgs[si,:,:]], refImg, op)[0]
        
# #%%
# viewer = napari.Viewer()
# viewer.add_image(regImgs, name=f's2p plane {pn}')
# viewer.add_image(regS2p, name=f'serial s2p -{prevN} images')
# viewer.add_image(regAffine, name=f'serial Affine -{prevN} images')
# viewer.add_image(regBi, name=f'serial Bilinear -{prevN} images')

# '''
# Serial s2p is definitely better than s2p to ref session,
# but seems worse than StackReg.
# Need to quantify this.

# '''













#%%
#%%
#%% (2) Serial registration using StackReg and suite2p
# include re-calculating old registration
pn = vi+3 # +0, +1, +2, +3
prevN = 3
op = {'smooth_sigma': 1.15, 'maxregshift': 0.3, 'smooth_sigma_time': 0, 'snr_thresh': 1.2, 'block_size_list': [128,32]}

print(f'Collecting images from plane {pn}\n using {prevN} previous images.')

leftBuffer = 30
rightBuffer = 30 if mouse < 50 else 100
bottomBuffer = 10
topBuffer = 50

planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
# regFn = f'{planeDir}s2p_nr_reg_bu1.npy' # for JK052. Need to get rid of _bu1.py file
regFn = f'{planeDir}s2p_nr_reg.npy'
reg = np.load(regFn, allow_pickle=True).item()
regImgs = reg['regImgs'][selectedSi,topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
# re-calculate!

numSelected = len(selectedSi)
mimgs = np.zeros_like(regImgs)
mimgClahe = np.zeros_like(regImgs)
refOld = []
for si, sn in enumerate(selectedSnums):
    opsFn = f'{planeDir}{sn:03}/plane0/ops.npy'
    ops = np.load(opsFn, allow_pickle=True).item()
    mimgs[si,:,:] = ops['meanImg'][topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
    mimgClahe[si,:,:] = clahe_each(mimgs[si,:,:])
    if sn == refSn:
        refOld = ops['meanImg'][topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
if len(refOld) == 0:
    raise('Reference session image not defined (for old reg method).')

srBi = StackReg(StackReg.BILINEAR)
# srRigid = StackReg(StackReg.RIGID_BODY)
srAffine = StackReg(StackReg.AFFINE)
regBi = np.zeros_like(mimgs)
regAffine = np.zeros_like(mimgs)
# regRigid = np.zeros_like(mimgs)
regS2p = np.zeros_like(mimgs)
regOld = np.zeros_like(mimgs)

tformsBi = []
tformsAffine = []

roff1 = []
roff2 = []
nroff1 = []
nroff2 = []

for si in range(numSelected):
    if si == 0:
        regBi[si,:,:] = mimgs[si,:,:]
        regAffine[si,:,:] = mimgs[si,:,:]
        # regRigid[si,:,:] = mimgs[si,:,:]
        
        regS2p[si,:,:] = mimgs[si,:,:]

    else:
        previStart = max(0,si-prevN)
        
        refBi = clahe_each(np.mean(regBi[previStart:si,:,:], axis=0))
        tform = srBi.register(refBi, mimgClahe[si,:,:])
        regBi[si,:,:] = srBi.transform(mimgs[si,:,:], tmat=tform)
        tformsBi.append(tform)
        
        refAffine = clahe_each(np.mean(regAffine[previStart:si,:,:], axis=0))
        tform = srAffine.register(refAffine, mimgClahe[si,:,:])
        regAffine[si,:,:] = srAffine.transform(mimgs[si,:,:], tmat=tform)
        tformsAffine.append(tform)
        # refRigid = clahe_each(np.mean(regRigid[previStart:si,:,:], axis=0))
        # tform = srRigid.register(refRigid, mimgClahe[si,:,:])
        # regRigid[si,:,:] = srRigid.transform(mimgs[si,:,:], tmat=tform)

        refImg = np.mean(regS2p[previStart:si,:,:], axis=0)
        regS2p[si,:,:], roff1tmp, roff2tmp, nroff1tmp, nroff2tmp = s2p_2step_nr([mimgs[si,:,:]], refImg, op)
        roff1.append(roff1tmp[0])
        roff2.append(roff2tmp[0])
        nroff1.append(nroff1tmp[0])
        nroff2.append(nroff2tmp[0])

regOld, roff1Old, roff2Old, nroff1Old, nroff2Old= s2p_2step_nr(mimgs, refOld, op)

viewer = napari.Viewer()
viewer.add_image(mimgs, name=f'mean images plane {pn}')
viewer.layers[f'mean images plane {pn}'].contrast_limits = (0,np.amax(mimgs)/2)
viewer.add_image(regOld, name=f's2p plane {pn}')
viewer.layers[f's2p plane {pn}'].contrast_limits = (0,np.amax(regOld)/2)
viewer.add_image(regS2p, name=f'serial s2p -{prevN} images')
viewer.layers[f'serial s2p -{prevN} images'].contrast_limits = (0,np.amax(regS2p)/2)
# viewer.add_image(regRigid, name=f'serial Rigid -{prevN} images')
viewer.add_image(regAffine, name=f'serial Affine -{prevN} images')
viewer.layers[f'serial Affine -{prevN} images'].contrast_limits = (0,np.amax(regAffine)/2)
viewer.add_image(regBi, name=f'serial Bilinear -{prevN} images')
viewer.layers[f'serial Bilinear -{prevN} images'].contrast_limits = (0,np.amax(regBi)/2)

#% Show sessions farther away next to each other
indOrder = [0,numSelected-1,1,numSelected-2,2,numSelected-3,3,numSelected-4,4,numSelected-5]
viewer = napari.Viewer()
viewer.add_image(mimgs[indOrder,:,:], name=f'mean images plane {pn}')
viewer.layers[f'mean images plane {pn}'].contrast_limits = (0,np.amax(mimgs)/2)
viewer.add_image(regOld[indOrder,:,:], name=f's2p plane {pn}')
viewer.layers[f's2p plane {pn}'].contrast_limits = (0,np.amax(regOld)/2)
viewer.add_image(regS2p[indOrder,:,:], name=f'serial s2p -{prevN} images')
viewer.layers[f'serial s2p -{prevN} images'].contrast_limits = (0,np.amax(regS2p)/2)
# viewer.add_image(regRigid, name=f'serial Rigid -{prevN} images')
viewer.add_image(regAffine[indOrder,:,:], name=f'serial Affine -{prevN} images')
viewer.layers[f'serial Affine -{prevN} images'].contrast_limits = (0,np.amax(regAffine)/2)
viewer.add_image(regBi[indOrder,:,:], name=f'serial Bilinear -{prevN} images')
viewer.layers[f'serial Bilinear -{prevN} images'].contrast_limits = (0,np.amax(regBi)/2)

#% Show sessions farther away next to each other CLAHE
regBiClahe = clahe_multi(regBi)
regAffClahe = clahe_multi(regAffine)
regS2pClahe = clahe_multi(regS2p)
regOldClahe = clahe_multi(regOld)

indOrder = [0,numSelected-1,1,numSelected-2,2,numSelected-3,3,numSelected-4,4,numSelected-5]
viewer = napari.Viewer()
# viewer.add_image(mimgsClahe[indOrder,:,:], name=f'mean images plane {pn}')
viewer.add_image(regOldClahe[indOrder,:,:], name=f's2p plane {pn}')
viewer.add_image(regS2pClahe[indOrder,:,:], name=f'serial s2p -{prevN} images')
# viewer.add_image(regRigid, name=f'serial Rigid -{prevN} images')
viewer.add_image(regAffClahe[indOrder,:,:], name=f'serial Affine -{prevN} images')
viewer.add_image(regBiClahe[indOrder,:,:], name=f'serial Bilinear -{prevN} images')

'''
Seems StackReg is superior. Maybe because they maintain linear relationship between pixel positions.
Serial suite2p shows errors in patches. Still better than the old suite2p (to ref).
'''


#%% (2) Quantification
# Correlation of the patches, to the reference or to the grand average.
# Mean, minimum, patches of the lowest correlation value.
# Run s2p 2step nr again, because pixle values were trimmed before.

# Or, mean of the correlation matrix. (between all pairs)

# Compare with s2p nr to the reference (existing) and serial s2p nr.

patchLength = 150

Ly = mimgs.shape[1]
Lx = mimgs.shape[2]
patchNumY = Ly//patchLength + 1
patchNumX = Lx//patchLength + 1

overlapY = (patchLength * patchNumY - Ly) // (patchNumY-1)
startY = np.array([(patchLength - overlapY)*yi for yi in range(patchNumY)])
overlapX = (patchLength * patchNumX - Lx) // (patchNumX-1)
startX = np.array([(patchLength - overlapX)*xi for xi in range(patchNumX)])

corr2refAllBi = np.zeros((numSelected, patchNumY, patchNumX))
# corr2refAllClahe = np.zeros((numSelected, patchNumY, patchNumX))
corr2meanAllBi = np.zeros((numSelected, patchNumY, patchNumX))
corrPairsAllBi = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
corr2refBi = np.zeros(numSelected)
corr2meanBi = np.zeros(numSelected)
corrPairsBi = np.zeros((numSelected, numSelected))

corr2refAllAff = np.zeros((numSelected, patchNumY, patchNumX))
corr2meanAllAff = np.zeros((numSelected, patchNumY, patchNumX))
corrPairsAllAff = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
corr2refAff = np.zeros(numSelected)
corr2meanAff = np.zeros(numSelected)
corrPairsAff = np.zeros((numSelected, numSelected))

# serial s2p
corr2refAllS2p = np.zeros((numSelected, patchNumY, patchNumX))
corr2meanAllS2p = np.zeros((numSelected, patchNumY, patchNumX))
corrPairsAllS2p = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
corr2refS2p = np.zeros(numSelected)
corr2meanS2p = np.zeros(numSelected)
corrPairsS2p = np.zeros((numSelected, numSelected))

# old (original) s2p
corr2refAllOld = np.zeros((numSelected, patchNumY, patchNumX))
corr2meanAllOld = np.zeros((numSelected, patchNumY, patchNumX))
corrPairsAllOld = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
corr2refOld = np.zeros(numSelected)
corr2meanOld = np.zeros(numSelected)
corrPairsOld = np.zeros((numSelected, numSelected))

# Make pixels with 0 in at least one session as 0 for all sessions, for StackReg
zeroInd = np.where(np.sum(regBi>0, axis=0) < regBi.shape[0])
regBi0 = regBi.copy()
for i in range(len(zeroInd[0])):
    regBi0[:,zeroInd[0][i], zeroInd[1][i]]=0
    
zeroInd = np.where(np.sum(regAffine>0, axis=0) < regAffine.shape[0])
regAffine0 = regAffine.copy()
for i in range(len(zeroInd[0])):
    regAffine0[:,zeroInd[0][i], zeroInd[1][i]]=0
# regBi0Clahe = clahe_multi(regBi0) # Clahe does not affect correlation values at all

# Make pixels as 0 for buffers calculated from the 1st rigid translation, for serial s2p nr
yoffs = np.array([off[0][0] for off in roff1])
xoffs = np.array([off[1][0] for off in roff1])
yneg = np.where(yoffs<0)[0]
ytop = 0 if len(yneg)==0 else -min(yoffs[yneg])
ypos = np.where(yoffs>0)[0]
ybottom = regS2p.shape[0] if len(ypos)==0 else -max(yoffs[ypos])
xneg = np.where(xoffs<0)[0]
xleft = 0 if len(xneg)==0 else -min(xoffs[xneg])
xpos = np.where(xoffs>0)[0]
xright = regS2p.shape[1] if len(xpos)==0 else -max(xoffs[xpos])

regS2p0 = regS2p.copy()
for si in range(numSelected):
    # tempOffsetY = reg['rigid_offsets_1st'][0][0][si]
    # if tempOffsetY > 0:
    #     regImg0[i,-tempOffsetY:,:] = 0
    # elif tempOffsetY < 0:
    #     regImg0[i,:-tempOffsetY,:] = 0
    
    # tempOffsetX = reg['rigid_offsets_1st'][0][1][si]
    # if tempOffsetX > 0:
    #     regImg0[i,:,-tempOffsetX:] = 0
    # elif tempOffsetY < 0:
    #     regImg0[i,:,:-tempOffsetX] = 0
    
    regS2p0[si,:ytop,:] = 0
    regS2p0[si,ybottom:,:] = 0
    regS2p0[si,:,:xleft] = 0
    regS2p0[si,:,xright:] = 0
    
# Make pixels as 0 for buffers calculated from the 1st rigid translation, for the old s2p nr
yoffs = roff1Old[0][0]
xoffs = roff1Old[0][1]
yneg = np.where(yoffs<0)[0]
ytop = 0 if len(yneg)==0 else -min(yoffs[yneg])
ypos = np.where(yoffs>0)[0]
ybottom = regS2p.shape[0] if len(ypos)==0 else -max(yoffs[ypos])
xneg = np.where(xoffs<0)[0]
xleft = 0 if len(xneg)==0 else -min(xoffs[xneg])
xpos = np.where(xoffs>0)[0]
xright = regS2p.shape[1] if len(xpos)==0 else -max(xoffs[xpos])
regOld0 = regOld.copy()
regOld0[:,:ytop,:] = 0
regOld0[:,ybottom:,:] = 0
regOld0[:,:,:xleft] = 0
regOld0[:,:,xright:] = 0

#%% viewer = napari.Viewer()
# viewer.add_image(regS2p, name='serial s2p')
# viewer.add_image(regS2p0, name='0 padding')
# viewer.add_image(regOld, name='s2p old')
# viewer.add_image(regOld0, name='0 for old')

viewer = napari.Viewer()
viewer.add_image(regOld0, name='0 padding for old s2p')
viewer.add_image(regS2p0, name='0 padding for serial s2p')
viewer.add_image(regAffine0, name='0 padding for serial affine')
viewer.add_image(regBi0, name='0 padding for serial bilinear')

# #% Correlation with the reference session and the mean image of all the sessions
# refSi = np.where(np.array(selectedSnums) == refSn)[0][0]
# refImgMeanBi = np.mean(regBi0,axis=0)
# refImgMeanAff = np.mean(regAffine0,axis=0)
# refImgMeanS2p = np.mean(regS2p0,axis=0)
# refImgMeanOld = np.mean(regOld0,axis=0)
# for si in range(numSelected):
#     corr2refBi[si] = np.corrcoef(regBi0[refSi,:,:].flatten(),
#                                  regBi0[si,:,:].flatten())[0,1]
#     corr2meanBi[si] = np.corrcoef(refImgMeanBi.flatten(),
#                                  regBi0[si,:,:].flatten())[0,1]
    
#     corr2refAff[si] = np.corrcoef(regAffine0[refSi,:,:].flatten(),
#                                  regAffine0[si,:,:].flatten())[0,1]
#     corr2meanAff[si] = np.corrcoef(refImgMeanAff.flatten(),
#                                  regAffine0[si,:,:].flatten())[0,1]
    
#     corr2refS2p[si] = np.corrcoef(regS2p0[refSi,:,:].flatten(),
#                                  regS2p0[si,:,:].flatten())[0,1]
#     corr2meanS2p[si] = np.corrcoef(refImgMeanS2p.flatten(),
#                                  regS2p0[si,:,:].flatten())[0,1]
    
#     corr2refOld[si] = np.corrcoef(regOld0[refSi,:,:].flatten(),
#                                  regOld0[si,:,:].flatten())[0,1]
#     corr2meanOld[si] = np.corrcoef(refImgMeanOld.flatten(),
#                                  regOld0[si,:,:].flatten())[0,1]
    
#     for yi in range(patchNumY):
#         for xi in range(patchNumX):
#             corr2refAllBi[si,yi,xi] = np.corrcoef(regBi0[refSi, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten(), 
#                                              regBi0[si, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten())[0,1]
#             # corr2refAllClahe[si,yi,xi] = np.corrcoef(regBi0Clahe[refSi, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten(), 
#             #                                  regBi0Clahe[si, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten())[0,1]
#             corr2meanAllBi[si,yi,xi] = np.corrcoef(refImgMeanBi[startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten(), 
#                                              regBi0[si, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten())[0,1]

#             corr2refAllAff[si,yi,xi] = np.corrcoef(regAffine0[refSi, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten(), 
#                                              regAffine0[si, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten())[0,1]
#             # corr2refAllClahe[si,yi,xi] = np.corrcoef(regBi0Clahe[refSi, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten(), 
#             #                                  regBi0Clahe[si, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten())[0,1]
#             corr2meanAllAff[si,yi,xi] = np.corrcoef(refImgMeanAff[startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten(), 
#                                              regAffine0[si, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten())[0,1]
            
#             corr2refAllS2p[si,yi,xi] = np.corrcoef(regS2p0[refSi, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten(), 
#                                              regS2p0[si, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten())[0,1]
#             corr2meanAllS2p[si,yi,xi] = np.corrcoef(refImgMeanS2p[startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten(), 
#                                              regS2p0[si, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten())[0,1]
            
#             corr2refAllOld[si,yi,xi] = np.corrcoef(regOld0[refSi, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten(), 
#                                              regOld0[si, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten())[0,1]
#             corr2meanAllOld[si,yi,xi] = np.corrcoef(refImgMeanOld[startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten(), 
#                                              regOld0[si, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)].flatten())[0,1]
 
#%% Correlation between each session
# Including clahe normalization

# regBiClahe = clahe_multi(regBi0)
# regAffClahe = clahe_multi(regAffine0)
# regS2pClahe = clahe_multi(regS2p0)
# regOldClahe = clahe_multi(regOld0)

corrPairsBiClahe = np.zeros_like(corrPairsBi)
corrPairsBiAllClahe = np.zeros_like(corrPairsAllBi)
corrPairsAffClahe = np.zeros_like(corrPairsBi)
corrPairsAffAllClahe = np.zeros_like(corrPairsAllBi)
corrPairsS2pClahe = np.zeros_like(corrPairsBi)
corrPairsS2pAllClahe = np.zeros_like(corrPairsAllBi)
corrPairsOldClahe = np.zeros_like(corrPairsBi)
corrPairsOldAllClahe = np.zeros_like(corrPairsAllBi)
#%
for si1 in range(numSelected-1):
    for si2 in range(si1+1,numSelected):
        corrPairsBi[si1,si2] = np.corrcoef(regBi0[si1,:,:].flatten(),
                                 regBi0[si2,:,:].flatten())[0,1]
        corrPairsAff[si1,si2] = np.corrcoef(regAffine0[si1,:,:].flatten(),
                                 regAffine0[si2,:,:].flatten())[0,1]
        corrPairsS2p[si1,si2] = np.corrcoef(regS2p0[si1,:,:].flatten(),
                                 regS2p0[si2,:,:].flatten())[0,1]
        corrPairsOld[si1,si2] = np.corrcoef(regOld0[si1,:,:].flatten(),
                                 regOld0[si2,:,:].flatten())[0,1]
        
        corrPairsBiClahe[si1,si2] = np.corrcoef(regBiClahe[si1,:,:].flatten(),
                                 regBiClahe[si2,:,:].flatten())[0,1]
        corrPairsAffClahe[si1,si2] = np.corrcoef(regAffClahe[si1,:,:].flatten(),
                                 regAffClahe[si2,:,:].flatten())[0,1]
        corrPairsS2pClahe[si1,si2] = np.corrcoef(regS2pClahe[si1,:,:].flatten(),
                                 regS2pClahe[si2,:,:].flatten())[0,1]
        corrPairsOldClahe[si1,si2] = np.corrcoef(regOldClahe[si1,:,:].flatten(),
                                 regOldClahe[si2,:,:].flatten())[0,1]
        
        
        for yi in range(patchNumY):
            for xi in range(patchNumX):
                y1 = startY[yi]
                y2 = min(startY[yi]+patchLength, Ly)
                x1 = startX[xi]
                x2 = min(startX[xi]+patchLength, Lx)
                corrPairsAllBi[si1, si2, yi, xi] = np.corrcoef(regBi0[si1, y1:y2, x1:x2].flatten(), 
                                             regBi0[si2, y1:y2, x1:x2].flatten())[0,1]
                corrPairsAllAff[si1, si2, yi, xi] = np.corrcoef(regAffine0[si1, y1:y2, x1:x2].flatten(), 
                                             regAffine0[si2, y1:y2, x1:x2].flatten())[0,1]
                corrPairsAllS2p[si1, si2, yi, xi] = np.corrcoef(regS2p0[si1, y1:y2, x1:x2].flatten(), 
                                             regS2p0[si2, y1:y2, x1:x2].flatten())[0,1]
                corrPairsAllOld[si1, si2, yi, xi] = np.corrcoef(regOld0[si1, y1:y2, x1:x2].flatten(), 
                                             regOld0[si2, y1:y2, x1:x2].flatten())[0,1]

                corrPairsBiAllClahe[si1, si2, yi, xi] = np.corrcoef(regBiClahe[si1, y1:y2, x1:x2].flatten(), 
                                             regBiClahe[si2, y1:y2, x1:x2].flatten())[0,1]
                corrPairsAffAllClahe[si1, si2, yi, xi] = np.corrcoef(regAffClahe[si1, y1:y2, x1:x2].flatten(), 
                                             regAffClahe[si2, y1:y2, x1:x2].flatten())[0,1]
                corrPairsS2pAllClahe[si1, si2, yi, xi] = np.corrcoef(regS2pClahe[si1, y1:y2, x1:x2].flatten(), 
                                             regS2pClahe[si2, y1:y2, x1:x2].flatten())[0,1]
                corrPairsOldAllClahe[si1, si2, yi, xi] = np.corrcoef(regOldClahe[si1, y1:y2, x1:x2].flatten(), 
                                             regOldClahe[si2, y1:y2, x1:x2].flatten())[0,1]
#%% Display correlation - Full field correlation
# fig, ax = plt.subplots(2,1)
# ax[0].plot(corr2refBi, label='bilinear')
# ax[0].plot(corr2refAff, label='affine')
# ax[0].plot(corr2refS2p, label='serial s2p')
# ax[0].plot(corr2refOld, label='old s2p')
# ax[0].set_title('Corr to ref session')

# ax[0].legend()

# ax[1].plot(corr2meanBi, label='bilinear')
# ax[1].plot(corr2meanAff, label='affine')
# ax[1].plot(corr2meanS2p, label='serial s2p')
# ax[1].plot(corr2meanOld, label='old s2p')
# ax[1].set_title('Corr to exp mean')
# fig.tight_layout()

#% Display correlation - Pairwise full field correlation
minVal = np.percentile(np.array([val for val in corrPairsOld.flatten() if val!=0]), 1)
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(corrPairsBi, vmin=minVal, vmax=1)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].set_title('Bilinear')
ax[0,1].imshow(corrPairsAff, vmin=minVal, vmax=1)
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)
ax[0,1].set_title('Affine')
ax[1,0].imshow(corrPairsS2p, vmin=minVal, vmax=1)
ax[1,0].set_title('Serial s2p')
im=ax[1,1].imshow(corrPairsOld, vmin=minVal, vmax=1)
ax[1,1].get_yaxis().set_visible(False)
ax[1,1].set_title('Old s2p')
plt.colorbar(im, ax=ax)
fig.suptitle('Full FOV session pairwise correlation')

minVal = np.percentile(np.array([val for val in corrPairsOldClahe.flatten() if val!=0]), 1)
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(corrPairsBiClahe, vmin=minVal, vmax=1)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].set_title('Bilinear')
ax[0,1].imshow(corrPairsAffClahe, vmin=minVal, vmax=1)
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)
ax[0,1].set_title('Affine')
ax[1,0].imshow(corrPairsS2pClahe, vmin=minVal, vmax=1)
ax[1,0].set_title('Serial s2p')
im=ax[1,1].imshow(corrPairsOldClahe, vmin=minVal, vmax=1)
ax[1,1].get_yaxis().set_visible(False)
ax[1,1].set_title('Old s2p')
plt.colorbar(im, ax=ax)
fig.suptitle('Full FOV session pairwise correlation (CLAHE)')

#% Compare values
valBiPair = np.sum(np.triu(corrPairsBi,k=1))/(numSelected*(numSelected-1)/2)
valAffPair = np.sum(np.triu(corrPairsAff,k=1))/(numSelected*(numSelected-1)/2)
valS2pPair = np.sum(np.triu(corrPairsS2p,k=1))/(numSelected*(numSelected-1)/2)
valOldPair = np.sum(np.triu(corrPairsOld,k=1))/(numSelected*(numSelected-1)/2)

valBiPairDiag = np.mean(np.diag(corrPairsBi,k=1))
valAffPairDiag = np.mean(np.diag(corrPairsAff,k=1))
valS2pPairDiag = np.mean(np.diag(corrPairsS2p,k=1))
valOldPairDiag = np.mean(np.diag(corrPairsOld,k=1))


#%%
fig, ax = plt.subplots(1,2)
ax[0].plot(np.diag(corrPairsOld, k=1), label='Old s2p')
ax[0].plot(np.diag(corrPairsS2p, k=1), label='Serial s2p')
ax[0].plot(np.diag(corrPairsAff, k=1), label='Serial Affine')
ax[0].plot(np.diag(corrPairsBi, k=1), label='Serial Bilinear')
ax[0].set_title('Serial correlation')
ax[0].legend()

ax[1].plot(np.diag(corrPairsOldClahe, k=1), label='Old s2p')
ax[1].plot(np.diag(corrPairsS2pClahe, k=1), label='Serial s2p')
ax[1].plot(np.diag(corrPairsAffClahe, k=1), label='Serial Affine')
ax[1].plot(np.diag(corrPairsBiClahe, k=1), label='Serial Bilinear')
ax[1].set_title('Serial correlation (CLAHE)')
ax[1].legend()


#%% Show sessions farther away next to each other
indOrder = [0,numSelected-1,1,numSelected-2,2,numSelected-3,3,numSelected-4,4,numSelected-5]
viewer = napari.Viewer()
viewer.add_image(mimgs[indOrder,:,:], name=f'mean images plane {pn}')
viewer.add_image(regOld[indOrder,:,:], name=f's2p plane {pn}')
viewer.add_image(regS2p[indOrder,:,:], name=f'serial s2p -{prevN} images')
# viewer.add_image(regRigid, name=f'serial Rigid -{prevN} images')
viewer.add_image(regAffine[indOrder,:,:], name=f'serial Affine -{prevN} images')
viewer.add_image(regBi[indOrder,:,:], name=f'serial Bilinear -{prevN} images')

#%% Show sessions farther away next to each other CLAHE
indOrder = [0,numSelected-1,1,numSelected-2,2,numSelected-3,3,numSelected-4,4,numSelected-5]
viewer = napari.Viewer()
# viewer.add_image(mimgsClahe[indOrder,:,:], name=f'mean images plane {pn}')
viewer.add_image(regOldClahe[indOrder,:,:], name=f's2p plane {pn}')
viewer.add_image(regS2pClahe[indOrder,:,:], name=f'serial s2p -{prevN} images')
# viewer.add_image(regRigid, name=f'serial Rigid -{prevN} images')
viewer.add_image(regAffClahe[indOrder,:,:], name=f'serial Affine -{prevN} images')
viewer.add_image(regBiClahe[indOrder,:,:], name=f'serial Bilinear -{prevN} images')


#%% Calculate pixel value difference
# both absolute and relative (normalized, using CLAHE) values
# re-calculate old s2p, since it used clipped image

diffAbsPairsAllBi = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
diffRelPairsAllBi = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
diffAbsPairsBi = np.zeros((numSelected, numSelected))
diffRelPairsBi = np.zeros((numSelected, numSelected))

diffAbsPairsAllAff = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
diffRelPairsAllAff = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
diffAbsPairsAff = np.zeros((numSelected, numSelected))
diffRelPairsAff = np.zeros((numSelected, numSelected))

# serial s2p
diffAbsPairsAllS2p = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
diffRelPairsAllS2p = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
diffAbsPairsS2p = np.zeros((numSelected, numSelected))
diffRelPairsS2p = np.zeros((numSelected, numSelected))

# old (original) s2p
diffAbsPairsAllOld = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
diffRelPairsAllOld = np.zeros((numSelected, numSelected, patchNumY, patchNumX))
diffAbsPairsOld = np.zeros((numSelected, numSelected))
diffRelPairsOld = np.zeros((numSelected, numSelected))

claheBi = clahe_multi(regBi0)
claheAff = clahe_multi(regAffine0)
claheS2p = clahe_multi(regS2p0)
claheOld = clahe_multi(regOld0)

pixNumBi = regBi0.shape[1]*regBi0.shape[2] - len(np.where(regBi0[0,:,:].flatten()==0)[0])
pixNumAff = regAffine0.shape[1]*regAffine0.shape[2] - len(np.where(regAffine0[0,:,:].flatten()==0)[0])
pixNumS2p = regS2p0.shape[1]*regS2p0.shape[2] - len(np.where(regS2p0[0,:,:].flatten()==0)[0])
pixNumOld = regOld0.shape[1]*regOld0.shape[2] - len(np.where(regOld0[0,:,:].flatten()==0)[0])
for si1 in range(numSelected-1):
    for si2 in range(si1+1, numSelected):
        diffAbsPairsBi[si1,si2] = np.sum(np.abs(regBi0[si1,:,:] - regBi0[si2,:,:]))/pixNumBi
        diffRelPairsBi[si1,si2] = np.sum(np.abs(claheBi[si1,:,:] - claheBi[si2,:,:]))/pixNumBi
        
        diffAbsPairsAff[si1,si2] = np.sum(np.abs(regAffine0[si1,:,:] - regAffine0[si2,:,:]))/pixNumAff
        diffRelPairsAff[si1,si2] = np.sum(np.abs(claheAff[si1,:,:] - claheAff[si2,:,:]))/pixNumAff
        
        diffAbsPairsS2p[si1,si2] = np.sum(np.abs(regS2p0[si1,:,:] - regS2p0[si2,:,:]))/pixNumS2p
        diffRelPairsS2p[si1,si2] = np.sum(np.abs(claheS2p[si1,:,:] - claheS2p[si2,:,:]))/pixNumS2p
        
        diffAbsPairsOld[si1,si2] = np.sum(np.abs(regOld0[si1,:,:] - regOld0[si2,:,:]))/pixNumOld
        diffRelPairsOld[si1,si2] = np.sum(np.abs(claheOld[si1,:,:] - claheOld[si2,:,:]))/pixNumOld
        
        for yi in range(patchNumY):
            for xi in range(patchNumX):
                y1 = startY[yi]
                y2 = min(startY[yi]+patchLength, Ly)
                x1 = startX[xi]
                x2 = min(startX[xi]+patchLength, Lx)
                tempPnBi = patchLength**2 - len(np.where(regBi0[0,y1:y2, x1:x2].flatten()==0)[0])
                tempPnAff = patchLength**2 - len(np.where(regAffine0[0,y1:y2, x1:x2].flatten()==0)[0])
                tempPnS2p = patchLength**2 - len(np.where(regS2p0[0,y1:y2, x1:x2].flatten()==0)[0])
                tempPnOld = patchLength**2 - len(np.where(regOld0[0,y1:y2, x1:x2].flatten()==0)[0])

                diffAbsPairsAllBi[si1, si2, yi, xi] = np.sum(np.abs(regBi0[si1,y1:y2,x1:x2] - regBi0[si2,y1:y2,x1:x2])) / tempPnBi
                diffRelPairsAllBi[si1, si2, yi, xi] = np.sum(np.abs(claheBi[si1,y1:y2,x1:x2] - claheBi[si2,y1:y2,x1:x2])) / tempPnBi
                
                diffAbsPairsAllAff[si1, si2, yi, xi] = np.sum(np.abs(regAffine0[si1,y1:y2,x1:x2] - regAffine0[si2,y1:y2,x1:x2])) / tempPnAff
                diffRelPairsAllAff[si1, si2, yi, xi] = np.sum(np.abs(claheAff[si1,y1:y2,x1:x2] - claheAff[si2,y1:y2,x1:x2])) / tempPnAff
                
                diffAbsPairsAllS2p[si1, si2, yi, xi] = np.sum(np.abs(regS2p0[si1,y1:y2,x1:x2] - regS2p0[si2,y1:y2,x1:x2])) / tempPnS2p
                diffRelPairsAllS2p[si1, si2, yi, xi] = np.sum(np.abs(claheS2p[si1,y1:y2,x1:x2] - claheS2p[si2,y1:y2,x1:x2])) / tempPnS2p
                
                diffAbsPairsAllOld[si1, si2, yi, xi] = np.sum(np.abs(regOld0[si1,y1:y2,x1:x2] - regOld0[si2,y1:y2,x1:x2])) / tempPnOld
                diffRelPairsAllOld[si1, si2, yi, xi] = np.sum(np.abs(claheOld[si1,y1:y2,x1:x2] - claheOld[si2,y1:y2,x1:x2])) / tempPnOld
                
#% Display difference - Pairwise full field 
minVal = np.percentile(diffAbsPairsOld.flatten(), 1)
maxVal = np.percentile(diffAbsPairsOld.flatten(), 99)
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(diffAbsPairsBi, vmin=minVal, vmax=maxVal)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].set_title('Bilinear')
ax[0,1].imshow(diffAbsPairsAff, vmin=minVal, vmax=maxVal)
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)
ax[0,1].set_title('Affine')
ax[1,0].imshow(diffAbsPairsS2p, vmin=minVal, vmax=maxVal)
ax[1,0].set_title('Serial s2p')
im=ax[1,1].imshow(diffAbsPairsOld, vmin=minVal, vmax=maxVal)
ax[1,1].get_yaxis().set_visible(False)
ax[1,1].set_title('Old s2p')
plt.colorbar(im, ax=ax)
fig.suptitle('Full FOV session pairwise difference (Mean Absolute)')


minVal = np.percentile(diffRelPairsOld.flatten(), 1)
maxVal = np.percentile(diffRelPairsOld.flatten(), 99)
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(diffRelPairsBi, vmin=minVal, vmax=maxVal)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].set_title('Bilinear')
ax[0,1].imshow(diffRelPairsAff, vmin=minVal, vmax=maxVal)
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)
ax[0,1].set_title('Affine')
ax[1,0].imshow(diffRelPairsS2p, vmin=minVal, vmax=maxVal)
ax[1,0].set_title('Serial s2p')
im=ax[1,1].imshow(diffRelPairsOld, vmin=minVal, vmax=maxVal)
ax[1,1].get_yaxis().set_visible(False)
ax[1,1].set_title('Old s2p')
plt.colorbar(im, ax=ax)
fig.suptitle('Full FOV session pairwise difference (Mean Relative value)')

#%% Display difference of difference to the old s2p nr - Pairwise full field 
minVal = np.percentile((diffAbsPairsBi-diffAbsPairsOld).flatten(), 1)
maxVal = np.percentile((diffAbsPairsBi-diffAbsPairsOld).flatten(), 99)
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(diffAbsPairsBi-diffAbsPairsOld, vmin=minVal, vmax=maxVal)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].set_title('Bilinear')
ax[0,1].imshow(diffAbsPairsAff-diffAbsPairsOld, vmin=minVal, vmax=maxVal)
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)
ax[0,1].set_title('Affine')
ax[1,0].imshow(diffAbsPairsS2p-diffAbsPairsOld, vmin=minVal, vmax=maxVal)
ax[1,0].set_title('Serial s2p')
im=ax[1,1].imshow(diffAbsPairsOld-diffAbsPairsOld, vmin=minVal, vmax=maxVal)
ax[1,1].get_yaxis().set_visible(False)
ax[1,1].set_title('Old s2p')
plt.colorbar(im, ax=ax)
fig.suptitle('Full FOV session pairwise difference (Mean Absolute)')


minVal = np.percentile((diffRelPairsBi-diffRelPairsOld).flatten(), 1)
maxVal = np.percentile((diffRelPairsBi-diffRelPairsOld).flatten(), 99)
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(diffRelPairsBi-diffRelPairsOld, vmin=minVal, vmax=maxVal)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].set_title('Bilinear')
ax[0,1].imshow(diffRelPairsAff-diffRelPairsOld, vmin=minVal, vmax=maxVal)
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)
ax[0,1].set_title('Affine')
ax[1,0].imshow(diffRelPairsS2p-diffRelPairsOld, vmin=minVal, vmax=maxVal)
ax[1,0].set_title('Serial s2p')
im=ax[1,1].imshow(diffRelPairsOld-diffRelPairsOld, vmin=minVal, vmax=maxVal)
ax[1,1].get_yaxis().set_visible(False)
ax[1,1].set_title('Old s2p')
plt.colorbar(im, ax=ax)
fig.suptitle('Full FOV session pairwise difference (Mean Relative value)')

#%% Plot serial differences
fig, ax = plt.subplots(1,2)
ax[0].plot(np.diag(diffAbsPairsOld, k=1), label='Old s2p')
ax[0].plot(np.diag(diffAbsPairsS2p, k=1), label='Serial s2p')
ax[0].plot(np.diag(diffAbsPairsAff, k=1), label='Serial Affine')
ax[0].plot(np.diag(diffAbsPairsBi, k=1), label='Serial Bilinear')
ax[0].set_title('Serial difference in full FOV pixel values')
ax[0].legend()

ax[1].plot(np.diag(diffRelPairsOld, k=1), label='Old s2p')
ax[1].plot(np.diag(diffRelPairsS2p, k=1), label='Serial s2p')
ax[1].plot(np.diag(diffRelPairsAff, k=1), label='Serial Affine')
ax[1].plot(np.diag(diffRelPairsBi, k=1), label='Serial Bilinear')
ax[1].set_title('Serial difference in full FOV pixel values (CLAHE)')
# ax[1].legend()





#%% Compare values
valDiffAbsBiPair = np.sum(np.triu(diffAbsPairsBi,k=1))/(numSelected*(numSelected-1)/2)
valDiffAbsAffPair = np.sum(np.triu(diffAbsPairsAff,k=1))/(numSelected*(numSelected-1)/2)
valDiffAbsS2pPair = np.sum(np.triu(diffAbsPairsS2p,k=1))/(numSelected*(numSelected-1)/2)
valDiffAbsOldPair = np.sum(np.triu(diffAbsPairsOld,k=1))/(numSelected*(numSelected-1)/2)

valDiffAbsBiPairDiag = np.mean(np.diag(diffAbsPairsBi,k=1))
valDiffAbsAffPairDiag = np.mean(np.diag(diffAbsPairsAff,k=1))
valDiffAbsS2pPairDiag = np.mean(np.diag(diffAbsPairsS2p,k=1))
valDiffAbsOldPairDiag = np.mean(np.diag(diffAbsPairsOld,k=1))






#%% Display correlation - Patch correlation
# minVal = min(np.percentile(corrPairsAllBi.flatten(), 1), np.percentile(corrPairsAllAff.flatten(), 1))
tempFlat = corrPairsAllOld.flatten()
minVal = np.percentile(tempFlat[np.where(tempFlat==0)[0]], 5)
fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        ax[yi,xi].imshow(corrPairsAllBi[:,:,yi,xi], vmin=minVal, vmax=1)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
fig.suptitle('Bilinear')
fig.tight_layout()

fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        ax[yi,xi].imshow(corrPairsAllAff[:,:,yi,xi], vmin=minVal, vmax=1)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
fig.suptitle('Affine')
fig.tight_layout()

fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        ax[yi,xi].imshow(corrPairsAllS2p[:,:,yi,xi], vmin=minVal, vmax=1)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
fig.suptitle('Suite2p')
fig.tight_layout()

# #%%
# fig, ax = plt.subplots(patchNumY, patchNumX)
# for yi in range(patchNumY):
#     for xi in range(patchNumX): 
#         im = ax[yi,xi].imshow(corrPairsAllOld[:,:,yi,xi], vmin=minVal, vmax=1)
#         if yi != patchNumY-1:
#             ax[yi,xi].get_xaxis().set_visible(False)
#         if xi != 0:
#             ax[yi,xi].get_yaxis().set_visible(False)
# fig.suptitle('Old s2p')
# plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# # plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
# # fig.tight_layout()


#%% Display differences in the correlation matrix.
minVal = -0.03
maxVal = 0.03
fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        tempMat = corrPairsAllBi[:,:,yi,xi]-corrPairsAllS2p[:,:,yi,xi]
        ax[yi,xi].imshow(tempMat, vmin = minVal, vmax = maxVal)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
        sumSerial = np.sum(np.diag(tempMat,k=1))
        sumAll = np.sum(np.triu(tempMat,k=1))
        ax[yi,xi].set_title(f'sum serial: {sumSerial:.2f}\nsum all: {sumAll:.2f}')
fig.suptitle('Pairwise correlation: Bilinear - Suite2p')
fig.tight_layout()

fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        tempMat = corrPairsAllAff[:,:,yi,xi]-corrPairsAllS2p[:,:,yi,xi]
        ax[yi,xi].imshow(tempMat, vmin = minVal, vmax = maxVal)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
        sumSerial = np.sum(np.diag(tempMat,k=1))
        sumAll = np.sum(np.triu(tempMat,k=1))
        ax[yi,xi].set_title(f'sum serial: {sumSerial:.2f}\nsum all: {sumAll:.2f}')
fig.suptitle('Pairwise correlation: Affine - Suite2p')
fig.tight_layout()

fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        tempMat = corrPairsAllBi[:,:,yi,xi]-corrPairsAllAff[:,:,yi,xi]
        ax[yi,xi].imshow(tempMat, vmin = minVal, vmax = maxVal)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
        sumSerial = np.sum(np.diag(tempMat,k=1))
        sumAll = np.sum(np.triu(tempMat,k=1))
        ax[yi,xi].set_title(f'sum serial: {sumSerial:.2f}\nsum all: {sumAll:.2f}')
fig.suptitle('Pairwise correlation: Bilinear - Affine')
fig.tight_layout()

#%% Patch pairwise correlation - clahe
minVal = -0.03
maxVal = 0.03
fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        tempMat = corrPairsBiAllClahe[:,:,yi,xi]-corrPairsS2pAllClahe[:,:,yi,xi]
        ax[yi,xi].imshow(tempMat, vmin = minVal, vmax = maxVal)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
        sumSerial = np.sum(np.diag(tempMat,k=1))
        sumAll = np.sum(np.triu(tempMat,k=1))
        ax[yi,xi].set_title(f'sum serial: {sumSerial:.2f}\nsum all: {sumAll:.2f}')
fig.suptitle('Pairwise correlation (CLAHE): Bilinear - Suite2p')
fig.tight_layout()

fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        tempMat = corrPairsAffAllClahe[:,:,yi,xi]-corrPairsS2pAllClahe[:,:,yi,xi]
        ax[yi,xi].imshow(tempMat, vmin = minVal, vmax = maxVal)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
        sumSerial = np.sum(np.diag(tempMat,k=1))
        sumAll = np.sum(np.triu(tempMat,k=1))
        ax[yi,xi].set_title(f'sum serial: {sumSerial:.2f}\nsum all: {sumAll:.2f}')
fig.suptitle('Pairwise correlation (CLAHE): Affine - Suite2p')
fig.tight_layout()

fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        tempMat = corrPairsBiAllClahe[:,:,yi,xi]-corrPairsAffAllClahe[:,:,yi,xi]
        ax[yi,xi].imshow(tempMat, vmin = minVal, vmax = maxVal)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
        sumSerial = np.sum(np.diag(tempMat,k=1))
        sumAll = np.sum(np.triu(tempMat,k=1))
        ax[yi,xi].set_title(f'sum serial: {sumSerial:.2f}\nsum all: {sumAll:.2f}')
fig.suptitle('Pairwise correlation (CLAHE): Bilinear - Affine')
fig.tight_layout()

#%% Display pairwise patch pixel value difference
minVal = -0.01
maxVal = 0.01
fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        tempMat = diffRelPairsAllBi[:,:,yi,xi]-diffRelPairsAllS2p[:,:,yi,xi]
        ax[yi,xi].imshow(tempMat, vmin = minVal, vmax = maxVal)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
        sumSerial = np.sum(np.diag(tempMat,k=1))
        sumAll = np.sum(np.triu(tempMat,k=1))
        ax[yi,xi].set_title(f'sum serial: {sumSerial:.2f}\nsum all: {sumAll:.2f}')
fig.suptitle('Pairwise difference (CLAHE): Bilinear - Suite2p')
fig.tight_layout()

fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        tempMat = diffRelPairsAllAff[:,:,yi,xi]-diffRelPairsAllS2p[:,:,yi,xi]
        ax[yi,xi].imshow(tempMat, vmin = minVal, vmax = maxVal)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
        sumSerial = np.sum(np.diag(tempMat,k=1))
        sumAll = np.sum(np.triu(tempMat,k=1))
        ax[yi,xi].set_title(f'sum serial: {sumSerial:.2f}\nsum all: {sumAll:.2f}')
fig.suptitle('Pairwise difference (CLAHE): Affine - Suite2p')
fig.tight_layout()

fig, ax = plt.subplots(patchNumY, patchNumX)
for yi in range(patchNumY):
    for xi in range(patchNumX): 
        tempMat = diffRelPairsAllBi[:,:,yi,xi]-diffRelPairsAllAff[:,:,yi,xi]
        ax[yi,xi].imshow(tempMat, vmin = minVal, vmax = maxVal)
        if yi != patchNumY-1:
            ax[yi,xi].get_xaxis().set_visible(False)
        if xi != 0:
            ax[yi,xi].get_yaxis().set_visible(False)
        sumSerial = np.sum(np.diag(tempMat,k=1))
        sumAll = np.sum(np.triu(tempMat,k=1))
        ax[yi,xi].set_title(f'sum serial: {sumSerial:.2f}\nsum all: {sumAll:.2f}')
fig.suptitle('Pairwise difference (CLAHE): Bilinear - Affine')
fig.tight_layout()


#%% View
# yxlist = [[0,3], [1,3], [2,3], [2,2], [3,3]]
yxlist = [[2,2]]
viewerBi = napari.Viewer()
viewerAff = napari.Viewer()
for li in range(len(yxlist)):
    yi = yxlist[li][0]
    xi = yxlist[li][1]
    tempBi = regBi0[:, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)]
    tempAff = regAffine0[:, startY[yi]:min(startY[yi]+patchLength, Ly), startX[xi]:min(startX[xi]+patchLength, Lx)]
    
    viewerBi.add_image(tempBi, name=f'{yi}, {xi} Bilinear')
    viewerAff.add_image(tempAff, name=f'{yi}, {xi} Affine')


