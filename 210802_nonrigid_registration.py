"""
Test nonrigid registration methods on mean images of each sessions
(1) Suite2p nonrigid registration method (intensity-based optical flow method)
- s2p_nr_test_0x.npy saved in h5Dir{mouse:03}/plane_y
- mean_img_reg_{sn}_upper.npy, mean_img_reg_{sn}_lower.npy, across_planes_corr.npy
saved in {h5Dir}{mouse:03}
(2) Scikit-image registration using optical flow (test both tvl1 and ilk methods)
(3) Elastix (or just simpleITK for free form deformation (bspline) and demon's method)
             
Use visual inspection and pixelwise correlation as the evaluation metrics
For pixelwise correlation, compare the values with within session comparison (first 1/3 VS last 1/3)
and between planes (just 1 plane apart, within each volume)
2021/08/02 JK

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

def get_session_names(baseDir, mouse, planeNum):
    tempFnList = glob.glob(f'{baseDir}{mouse:03}_*_plane_{planeNum}.h5')    
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

def phase_corr(a,b, transLim = 0):
    if a.shape != b.shape:
        raise('Dimensions must match')
    R = np.fft.fft2(a) * np.fft.fft2(b).conj()
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
#%% (1) Using suite2p nonrigid registration
## Load reference mimg.
## Load mean images from each session and make a list of them.
## Perform suite2p nonrigid registration.
## Save the resulting mimg.
''' s2p_nr_test_0x.npy saved in h5Dir{mouse:03}/plane_y '''


## These will be used across all registration methods:
## Calculate correlation value between mimg of first 1/3 and the last 1/3
## Calculate correaltion values between 1 planes apart within each volume. (in case of middle planes, calculate mean of 2 values)

## Load reference mimg.

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['maxregshiftNR'] = 10
op['snr_thresh'] = 1.2
bsList = [128,64,32]
for mi in [0,3,8]:
# for mi in [0]:    
    mouse = mice[mi]
    refSession = refSessions[mi]
    for pn in range(1,9):
    # for pn in range(1,2):
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
        else:
            Ly = mimgList[0].shape[0]
            Lx = mimgList[0].shape[1]
            
## Perform suite2p nonrigid registration.
# Test 3 different block sizes.
        for bs in bsList:
            print(f'JK{mouse:03} plane {pn} block size [{bs}, {bs}]')
            op['block_size'] = [bs,bs]
            ### ------------- compute registration masks ----------------- ###
            
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
            
            # #%% visual inspection
            # viewer = napari.Viewer()
            # for i in range(nSessions):
            #     data = np.array([refImg, frames[i]])
            #     viewer.add_image(data, name=sessionNames[i], visible=False)
            #%%
    ## Save the resulting mimg.
            op['sessionNames'] = sessionNames
            op['regImgs'] = frames
            op['rigid_offsets'] = rigid_offsets
            op['nonrigid_offsets'] = nonrigid_offsets
            savefnBase = 's2p_nr_test'
            tempFnList = glob.glob(f'{planeDir}{savefnBase}_*.npy')
            if len(tempFnList):
                fnames = [fn.split('\\')[1].split('.npy')[0] for fn in tempFnList]
                fnInds = np.sort([int(fn.split('_')[-1]) for fn in fnames])
                fnInd = fnInds[-1]+1
            else:
                fnInd = 0
            saveFn = f'{planeDir}{savefnBase}_{fnInd:02}'
            np.save(saveFn, op)


#%%
## These will be used across all registration methods:
## Calculate correlation value between mimg of first 1/3 and the last 1/3
## Calculate correaltion values between 1 planes apart within each volume. (in case of middle planes, calculate mean of 2 values)

#%% Register across planes. Using rigid registration because same-session imaging has same rotation
# Also add same session correlation (first 1/3 and last 1/3)
# Collect and save each images and their registered images to later check the quality of registration
# Save correlation values too
''' mean_img_reg_{sn}_upper.npy, mean_img_reg_{sn}_lower.npy, saved in {h5Dir}{mouse:03} '''
op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
for mi in [0,3,8]:
# for mi in [0]:    
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
        upperMimgList = []
        upperRegimgList = []
        for pi in range(1,5):
            piDir = f'{h5Dir}{mouse:03}/plane_{pi}/{sn}/plane0/'
            piBinFn = f'{piDir}data.bin'
            piOpsFn = f'{piDir}ops.npy'
            opsi = np.load(piOpsFn, allow_pickle=True).item()
            Lx = opsi['Lx']
            Ly = opsi['Ly']
            piImg = opsi['meanImg'].copy().astype(np.float32)
            rmin, rmax = np.percentile(piImg,1), np.percentile(piImg,99)
            piImg = np.clip(piImg, rmin, rmax)
            upperMimgList.append(piImg)
            for pj in range(pi,5):
                if pi == pj: # same plane correlation
                    nframes = opsi['nframes']
                    with BinaryFile(Ly = Ly, Lx = Lx, read_filename = piBinFn) as f:
                        inds1 = range(int(nframes/3))
                        frames = f.ix(indices=inds1).astype(np.float32)
                        mimg1 = frames.mean(axis=0)
                        rmin, rmax = np.percentile(mimg1,1), np.percentile(mimg1,99)
                        mimg1 = np.clip(mimg1, rmin, rmax)
        
                        inds2 = range(int(nframes/3*2), nframes)
                        frames = f.ix(indices=inds2).astype(np.float32)
                        mimg2 = frames.mean(axis=0)
                        rmin, rmax = np.percentile(mimg2,1), np.percentile(mimg2,99)
                        mimg2 = np.clip(mimg2, rmin, rmax)
                        upperCorr[pi-1,pi-1,i] = np.corrcoeff(mimg1.flatten(), mimg2.flatten())[0,1]
                else: # different plane correlation, after rigid registration
                    pjDir = f'{h5Dir}{mouse:03}/plane_{pj}/{sn}/plane0/'
                    pjOpsFn = f'{pjDir}ops.npy'
                    opsj = np.load(pjOpsFn, allow_pickle=True).item()
                    pjImg = opsj['meanImg'].copy().astype(np.float32)
                    rmin, rmax = np.percentile(pjImg,1), np.percentile(pjImg,99)
                    pjImg = np.clip(pjImg, rmin, rmax)
                    
                    # rigid registration
                    ymax, xmax,_,_,_ = phase_corr(piImg, pjImg)
                    pjImgReg = rigid.shift_frame(frame=pjImg, dy=ymax, dx=xmax)
                    upperCorr[pi-1,pj-1,i] = np.corrcoef(piImg.flatten(), pjImg.flatten())[0,1]
                    upperRegimgList.append(pjImgReg)
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
        lowerMimgList = []
        lowerRegimgList = []
        for pi in range(5,8):
            piDir = f'{h5Dir}{mouse:03}/plane_{pi}/{sn}/plane0/'
            piBinFn = f'{piDir}data.bin'
            piOpsFn = f'{piDir}ops.npy'
            opsi = np.load(piOpsFn, allow_pickle=True).item()
            Lx = opsi['Lx']
            Ly = opsi['Ly']
            piImg = opsi['meanImg'].copy().astype(np.float32)
            rmin, rmax = np.percentile(piImg,1), np.percentile(piImg,99)
            piImg = np.clip(piImg, rmin, rmax)
            lowerMimgList.append(piImg)
            for pj in range(pi,8):
                if pi == pj: # same plane correlation
                    nframes = opsi['nframes']
                    with BinaryFile(Ly = Ly, Lx = Lx, read_filename = piBinFn) as f:
                        inds1 = range(int(nframes/3))
                        frames = f.ix(indices=inds1).astype(np.float32)
                        mimg1 = frames.mean(axis=0)
                        rmin, rmax = np.percentile(mimg1,1), np.percentile(mimg1,99)
                        mimg1 = np.clip(mimg1, rmin, rmax)
        
                        inds2 = range(int(nframes/3*2), nframes)
                        frames = f.ix(indices=inds2).astype(np.float32)
                        mimg2 = frames.mean(axis=0)
                        rmin, rmax = np.percentile(mimg2,1), np.percentile(mimg2,99)
                        mimg2 = np.clip(mimg2, rmin, rmax)
                        lowerCorr[pi-5,pi-5,i] = np.corrcoef(mimg1.flatten(), mimg2.flatten())[0,1]
                else: # different plane correlation, after rigid registration
                    pjDir = f'{h5Dir}{mouse:03}/plane_{pj}/{sn}/plane0/'
                    pjOpsFn = f'{pjDir}ops.npy'
                    opsj = np.load(pjOpsFn, allow_pickle=True).item()
                    pjImg = opsj['meanImg'].copy().astype(np.float32)
                    rmin, rmax = np.percentile(pjImg,1), np.percentile(pjImg,99)
                    pjImg = np.clip(pjImg, rmin, rmax)
                    
                    # rigid registration
                    ymax, xmax,_,_,_ = phase_corr(piImg, pjImg)
                    pjImgReg = rigid.shift_frame(frame=pjImg, dy=ymax, dx=xmax)
                    lowerCorr[pi-5,pj-5,i] = np.corrcoef(piImg.flatten(), pjImg.flatten())[0,1]
                    lowerRegimgList.append(pjImgReg)
        lower = {}
        lower['lowerMimgList'] = lowerMimgList
        lower['lowerRegimgList'] = lowerRegimgList
        savefn = f'mean_img_reg_{sn}_lower'
        np.save(f'{h5Dir}{mouse:03}/{savefn}', lower)
    # result = {}
    # result['upperCorr'] = upperCorr
    # result['lowerCorr'] = lowerCorr
    # savefn = 'across_planes_corr'
    # np.save(f'{h5Dir}{mouse:03}/{savefn}', result)

#%% Plot the result (same-plane and across-plane correlation value)
# When calculating correlation, edges should be removed.
# So, first go through all session mean images and define edges to discard.

#%% Visual inspection of across-plane registration
# First, see how phase correlation worked for across-plane registration
mi = 0
mouse = mice[mi]
viewer = napari.Viewer()
pn = 1
mouseDir = f'{h5Dir}{mouse:03}/'
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
sessionNames = get_session_names(planeDir, mouse, pn)
nSessions = len(sessionNames)
upperCorr = np.zeros((4,4,nSessions))
sessionFolderNames = [x.split('_')[1] if len(x.split('_')[1])==3 else x[4:] for x in sessionNames]
for sn in sessionFolderNames:
    imgfn = f'mean_img_reg_{sn}_upper.npy'
    imgs = np.load(f'{mouseDir}{imgfn}', allow_pickle=True).item()
    # viewer.add_image(np.array(imgs['upperRegimgList']), name=sn)
    viewer.add_image(np.array(imgs['upperMimgList']), name=sn)


'''
Phase correlation across planes were terrible. Just moved too much.
Test again in one plane with limiting translation (20 pixels?), 
after removing 10% of the image from the edge.
'''
#%% Across plane registration test
edgeRem = 0.1 # Proportion of image in each axis to remove before registration
transLim = 20 # Number of pixels to limit translation for registration

mi = 8
mouse = mice[mi]
pn = 1
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
sessionNames = get_session_names(planeDir, mouse, pn)

viewer = napari.Viewer()
sessionFolderNames = [x.split('_')[1] if len(x.split('_')[1])==3 else x[4:] for x in sessionNames]
for sn in sessionFolderNames:
    upperMimgList = []
    upperRegimgList = []
    for pi in range(1,4):
    # for pi in range(5,8):        
        piDir = f'{h5Dir}{mouse:03}/plane_{pi}/{sn}/plane0/'
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
        upperMimgList.append(piImg)
        # for pj in range(pi+1,5):
        for pj in range(pi+1,9):
            pjDir = f'{h5Dir}{mouse:03}/plane_{pj}/{sn}/plane0/'
            pjOpsFn = f'{pjDir}ops.npy'
            opsj = np.load(pjOpsFn, allow_pickle=True).item()
            pjImg = opsj['meanImg'].copy().astype(np.float32)
            if mouse > 50:
                pjImgTemp = pjImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-70] # 052 has blank edge on the right end for ~70 pix
            else:
                pjImgTemp = pjImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1] # Ly is always shorter than Lx
            rmin, rmax = np.percentile(pjImgTemp,1), np.percentile(pjImgTemp,99)
            pjImg = np.clip(pjImg, rmin, rmax)
            
            # rigid registration
            ymax, xmax,_,_,_ = phase_corr(piImg, pjImgTemp, transLim)
            pjImgReg = rigid.shift_frame(frame=pjImg, dy=ymax, dx=xmax)
            upperRegimgList.append(pjImgReg)
    viewer.add_image(np.array(upperRegimgList), name=sn)
    




