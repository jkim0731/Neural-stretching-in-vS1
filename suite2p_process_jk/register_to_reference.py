# -*- coding: utf-8 -*-
"""
Register mimg of each session to that of reference session.
    Results: s2p_nr_reg.npy, saved in {h5Dir}{mouse:03}/plane_{pn}
For quality check, also run within-session same-plane and across-plane registration and calculate correlation coefficients.
    Results: mean_img_reg_{sn}_upper.npy, mean_img_reg_{sn}_lower.npy, saved in {h5Dir}{mouse:03} 
            same_session_regCorrVals_JK{mouse:03}, saved in {h5Dir}{mouse:03} 
Should have run 'register_all_sessions.py' first.
Use nonrigid registration with parameters confirmed from '210802_nonrigid_registration.py'

2021/08/25 JK
"""

#%% Basic imports and settings
import numpy as np
from matplotlib import pyplot as plt
from suite2p.registration import rigid, nonrigid, utils
import os, glob
import napari
from suite2p.io.binary import BinaryFile
import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'

mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]

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
        sname = f'{mouse:03}_{sn:03}_'
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

def same_session_planes_reg(mouse, sessionName, startPlane, edgeRem = 0.1, transLim = 20):
    mimgList = []
    regimgList = []
    corrVals = np.zeros((4,4))   
    for i, pi in enumerate(range(startPlane,startPlane+4)):
        piDir = f'{h5Dir}{mouse:03}/plane_{pi}/{sessionName}/plane0/'
        piBinFn = f'{piDir}data.bin'
        piOpsFn = f'{piDir}ops.npy'
        opsi = np.load(piOpsFn, allow_pickle=True).item()
        Lx = opsi['Lx']
        Ly = opsi['Ly']
        piImg = opsi['meanImg'].copy().astype(np.float32)
        if mouse > 50:
            piImg = piImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-70] # 052 has blank edge on the right end for ~70 pix
        else:
            piImg = piImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1] # Ly is always shorter than Lx
        rmin, rmax = np.percentile(piImg,1), np.percentile(piImg,99)
        piImg = np.clip(piImg, rmin, rmax)
        mimgList.append(piImg)
        for j, pj in enumerate(range(startPlane,startPlane+4)):
            if pi == pj: # same plane correlation
                nframes = opsi['nframes']
                with BinaryFile(Ly = Ly, Lx = Lx, read_filename = piBinFn) as f:
                    inds1 = range(int(nframes/3))
                    frames = f.ix(indices=inds1).astype(np.float32)
                    mimg1 = frames.mean(axis=0)
                    if mouse > 50:
                        mimg1 = mimg1[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-70] # 052 has blank edge on the right end for ~70 pix
                    else:
                        mimg1 = mimg1[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1] # Ly is always shorter than Lx
                    rmin, rmax = np.percentile(mimg1,1), np.percentile(mimg1,99)
                    mimg1 = np.clip(mimg1, rmin, rmax)
    
                    inds2 = range(int(nframes/3*2), nframes)
                    frames = f.ix(indices=inds2).astype(np.float32)
                    mimg2 = frames.mean(axis=0)
                    if mouse > 50:
                        mimg2 = mimg2[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-70] # 052 has blank edge on the right end for ~70 pix
                    else:
                        mimg2 = mimg2[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1] # Ly is always shorter than Lx
                    rmin, rmax = np.percentile(mimg2,1), np.percentile(mimg2,99)
                    mimg2 = np.clip(mimg2, rmin, rmax)
                    corrVals[i,i] = np.corrcoef(mimg1.flatten(), mimg2.flatten())[0,1]
            elif pj > pi: # different plane correlation, after rigid registration
                pjDir = f'{h5Dir}{mouse:03}/plane_{pj}/{sn}/plane0/'
                pjOpsFn = f'{pjDir}ops.npy'
                opsj = np.load(pjOpsFn, allow_pickle=True).item()
                pjImg = opsj['meanImg'].copy().astype(np.float32)
                if mouse > 50:
                    pjImg = pjImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-70] # 052 has blank edge on the right end for ~70 pix
                else:
                    pjImg = pjImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1] # Ly is always shorter than Lx
                rmin, rmax = np.percentile(pjImg,1), np.percentile(pjImg,99)
                pjImg = np.clip(pjImg, rmin, rmax)
                    
                # rigid registration
                ymax, xmax,_,_,_ = phase_corr(piImg, pjImg, transLim)
                pjImgReg = rigid.shift_frame(frame=pjImg, dy=ymax, dx=xmax)
                # corrVals[i,j] = np.corrcoef(piImg.flatten(), pjImgReg.flatten())[0,1] # This is wrong. Need to clip transLim
                corrVals[i,j] = np.corrcoef(piImg[transLim:-transLim, transLim:-transLim].flatten(), 
                                            pjImgReg[transLim:-transLim, transLim:-transLim].flatten())[0,1] # 2021/08/13.
                regimgList.append(pjImgReg)
    return mimgList, regimgList, corrVals

#%%
#%% Run within-session same-plane and between-plane registration 
# using edge removal and limit on the translation pixels.
# 
''' mean_img_reg_{sn}_upper.npy, mean_img_reg_{sn}_lower.npy, saved in {h5Dir}{mouse:03} '''
edgeRem = 0.1 # Proportion of image in each axis to remove before registration
transLim = 20 # Number of pixels to limit translation for registration

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0

# for mi in [0,3,8]:
for mi in [0]:    
    mouse = mice[mi]
    refSession = refSessions[mi]
    
    # upper volume first
    # Make a list of session names and corresponding files
    pn = 1
    planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
    sessionNames = get_session_names(planeDir, mouse, pn)
    nSessions = len(sessionNames)
    upperCorr = np.zeros((4,4,nSessions))
    sessionFolderNames = [x.split('_')[1] if len(x.split('_')[1])==3 else x[4:] for x in sessionNames]
    for i, sn in enumerate(sessionFolderNames):
        print(f'Processing JK{mouse:03} upper volume, session {sn}.')
        upperMimgList, upperRegimgList, upperCorr[:,:,i] = same_session_planes_reg(mouse, sn, pn)
        upper = {}
        upper['upperMimgList'] = upperMimgList
        upper['upperRegimgList'] = upperRegimgList
        savefn = f'mean_img_reg_{sn}_upper'
        np.save(f'{h5Dir}{mouse:03}/{savefn}', upper)
    # Then lower volume
    pn = 5
    planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
    sessionNames = get_session_names(planeDir, mouse, pn)
    nSessions = len(sessionNames)
    lowerCorr = np.zeros((4,4,nSessions))
    sessionFolderNames = [x.split('_')[1] if len(x.split('_')[1])==3 else x[4:] for x in sessionNames]
    for i, sn in enumerate(sessionFolderNames):
        print(f'Processing JK{mouse:03} lower volume, session {sn}.')
        lowerMimgList, lowerRegimgList, lowerCorr[:,:,i] = same_session_planes_reg(mouse, sn, pn)
        lower = {}
        lower['lowerMimgList'] = lowerMimgList
        lower['lowerRegimgList'] = lowerRegimgList
        savefn = f'mean_img_reg_{sn}_lower'
        np.save(f'{h5Dir}{mouse:03}/{savefn}', lower)
    corrSaveFn = f'same_session_regCorrVals_JK{mouse:03}'
    np.savez(f'{h5Dir}{mouse:03}/{corrSaveFn}', upperCorr = upperCorr, lowerCorr = lowerCorr)




#%% Run nonrigid across sessions
# with block size 128, maxregshiftNR 15
# Calculate pixel correlation between each mimg to ref mimg

edgeRem = 0.2 # Proportion of image in each axis to remove before calculating correlations
blankEdge = 70 # For mouse > 50, where there is a blank edge on the right side

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['maxregshiftNR'] = 15
op['snr_thresh'] = 1.2
op['block_size'] = [128, 128]
# for mi in [0,3,8]:
for mi in [0]:    
    mouse = mice[mi]
    refSession = refSessions[mi]
    # for pn in range(1,9):
    for pn in range(5,9):○
        planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
        refDir = f'{planeDir}{refSession:03}/plane0/'
        ops = np.load(f'{refDir}ops.npy', allow_pickle=True).item()
        refImg = ops['meanImg'].astype(np.float32)
        # rmin, rmax = np.int16(np.percentile(refImg,1)), np.int16(np.percentile(refImg,99))
        rmin, rmax = np.percentile(refImg,1), np.percentile(refImg,99)
        refImg = np.clip(refImg, rmin, rmax)
        
        # Make a list of session names and corresponding files
        sessionNames = get_session_names(planeDir, mouse, pn)
        nSessions = len(sessionNames)
            
## Load mean images from each session and make a list of them
        mimgList = []
        for sntemp in sessionNames:
            if len(sntemp.split('_')[2]) > 0:
                sn1 = sntemp.split('_')[1]
                sn2 = sntemp.split('_')[2]
                sn = f'{sn1}_{sn2}'
            else:
                sn = sntemp.split('_')[1]
            tempDir = os.path.join(planeDir, f'{sn}/plane0/')    
            ops = np.load(f'{tempDir}ops.npy', allow_pickle=True).item()
            tempImg = ops['meanImg']
            # rmin, rmax = np.int16(np.percentile(tempImg,1)), np.int16(np.percentile(tempImg,99))
            rmin, rmax = np.percentile(tempImg,1), np.percentile(tempImg,99)
            tempImg = np.clip(tempImg, rmin, rmax)
            mimgList.append(tempImg)
        
        # Test x and y length match across images
        ydiff = [x.shape[0]-mimgList[0].shape[0] for x in mimgList]
        xdiff = [x.shape[1]-mimgList[0].shape[1] for x in mimgList]
        if any(ydiff) or any(xdiff):
            raise(f'x or y length mismatch across sessions in JK{mouse:03} plane {pn}')
            
## Perform suite2p nonrigid registration.

        frames, rigid_offsets, nonrigid_offsets = s2p_nonrigid_registration(mimgList, refImg, op)

## Calculate pixel correlation
        if mouse > 50:
            tempFrame = frames[:,int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-blankEdge]
            tempRefImg = refImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-blankEdge]
        else:
            tempFrame = frames[:,int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1]
            tempRefImg = refImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1]
        corrVals = [np.corrcoef(tempRefImg.flatten(), tempImg.flatten())[0,1] 
                    for tempImg in tempFrame]
            
## Save the resulting mimg.
        op['sessionNames'] = sessionNames
        op['regImgs'] = frames
        op['rigid_offsets'] = rigid_offsets
        op['nonrigid_offsets'] = nonrigid_offsets
        op['corrVals'] = corrVals
        
        saveFn = f'{planeDir}s2p_nr_reg'
        np.save(saveFn, op)
















#%% For visual insepction of same-session reg
#%% Some within-correlation values are as low as <0.3
# Check those numbers and see what happend on those sessions
# mi = 8
# mouse = mice[mi]
# corrs = np.load(f'{h5Dir}{mouse:03}/same_session_regCorrVals_JK{mouse:03}.npz')
# upper = corrs['upperCorr']
# lower = corrs['lowerCorr']

# f, ax = plt.subplots(2,4, figsize=(13,7), sharey=True)
# ylimlow = 0.2
# ytickstep = 0.2
# for i in range(4):
#     ax[0,i].plot(upper[i,i,:])
#     ax[0,i].set_ylim(ylimlow,1)
#     ax[0,i].set_yticks(np.arange(ylimlow,1,ytickstep))
#     ax[0,i].set_title(f'Plane {i+1}', fontsize=15)
# for i in range(4):
#     ax[1,i].plot(lower[i,i,:])
#     ax[1,i].set_ylim(ylimlow,1)
#     ax[1,i].set_yticks(np.arange(ylimlow,1,ytickstep))
#     ax[1,i].set_title(f'Plane {i+5}', fontsize=15)
# ax[0,0].set_ylabel('Pixelwise correlation', fontsize=15)
# ax[1,0].set_ylabel('Pixelwise correlation', fontsize=15)
# ax[1,0].set_xlabel('Sessions', fontsize=15)

# f.suptitle(f'JK{mouse:03}', fontsize=20)
# f.tight_layout()

# #%% Check session names
# pn = 1
# planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
# sessionNames = get_session_names(planeDir, mouse, pn)
# nSessions = len(sessionNames)
# upperCorr = np.zeros((4,4,nSessions))
# sessionFolderNames = [x.split('_')[1] if len(x.split('_')[1])==3 else x[4:] for x in sessionNames]

# #%% Check images in the specified session
# pn = 1
# si = 16
# sn = sessionFolderNames[si]

# ops = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/{sn}/plane0/ops.npy', allow_pickle=True).item()
# Ly, Lx, nframes = ops['Ly'], ops['Lx'], ops['nframes']
# binFn = f'{h5Dir}{mouse:03}/plane_{pn}/{sn}/plane0/data.bin'
# with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binFn) as f:
#     inds1 = range(int(nframes/3))
#     frames = f.ix(indices=inds1).astype(np.float32)
#     mimg1 = frames.mean(axis=0)
#     rmin, rmax = np.percentile(mimg1,1), np.percentile(mimg1,99)
#     # mimg1 = np.clip(mimg1, rmin, rmax)
#     mimg1 = (np.clip(mimg1, rmin, rmax) - rmin) / (rmax-rmin)
    
#     inds2 = range(int(nframes/3*2), nframes)
#     frames = f.ix(indices=inds2).astype(np.float32)
#     mimg2 = frames.mean(axis=0)
#     rmin, rmax = np.percentile(mimg2,1), np.percentile(mimg2,99)
#     # mimg2 = np.clip(mimg2, rmin, rmax)
#     mimg2 = (np.clip(mimg2, rmin, rmax) - rmin) / (rmax-rmin)
    
# viewer = napari.Viewer()
# mixImg = np.zeros((Ly,Lx,3), dtype=np.float32)
# mixImg[:,:,0] = mimg1
# mixImg[:,:,2] = mimg1
# mixImg[:,:,1] = mimg2
# viewer.add_image(mixImg, rgb=True)
# viewer.add_image(np.stack([mimg1,mimg2],axis=0), name=f'JK{mouse:03} si {si} plane{pn}')





