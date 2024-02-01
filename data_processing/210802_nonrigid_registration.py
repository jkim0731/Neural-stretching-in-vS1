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


Results update 2021/08/26
Suite2p nonrigid registration might not be the best registration method.
It shows good match, but not perfect. 

Image contrast adjustment is necessary for intensity correlation. Otherwise,
brightness gradient across FOV dominates correlation values.
How will it affect same-session correlation values?
    
"""



#%% Basic imports and settings
import numpy as np
from matplotlib import pyplot as plt
from suite2p.registration import rigid, nonrigid, utils
import suite2p.registration as registration
import os, glob
import napari
from suite2p.io.binary import BinaryFile
from skimage import exposure
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate, warp_polar
from skimage.filters import difference_of_gaussians
from scipy.fftpack import fft2, fftshift

import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'

mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]


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

# CLAHE each mean images
def clahe_each(img: np.float64, kernel_size = None, clip_limit = 0.01, nbins = 2**8):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    newimg = exposure.equalize_adapthist(newimg, kernel_size = kernel_size, clip_limit = clip_limit, nbins=nbins)    
    return newimg

def s2p_register(mimgList, op, rigid_offsets, nonrigid_offsets = None):
    # Register using rigid (and nonrigid, optionally) offsets
    
    # Rigid registration
    frames = np.array(mimgList).astype(np.float32)
    ymax, xmax = rigid_offsets[0][0][0], rigid_offsets[0][1][0]
    for frame in frames:
        frame[:] = rigid.shift_frame(frame=frame, dy=ymax, dx=xmax)
        
    # Nonrigid registration
    if nonrigid_offsets is not None:
        Ly, Lx = op['Ly'], op['Lx']
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
# ''' mean_img_reg_{sn}_upper.npy, mean_img_reg_{sn}_lower.npy, saved in {h5Dir}{mouse:03} '''
# op = {}
# op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
# op['maxregshift'] = 0.3
# op['smooth_sigma_time'] = 0
# # for mi in [0,3,8]:
# for mi in [0]:    
#     mouse = mice[mi]
#     refSession = refSessions[mi]
    
#     # upper volume first
#     # Make a list of session names and corresponding files
#     pn = 1
#     planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
#     sessionNames = get_session_names(planeDir, mouse, pn)
#     nSessions = len(sessionNames)
#     upperCorr = np.zeros((4,4,nSessions))
#     sessionFolderNames = [x.split('_')[1] if len(x.split('_')[1])==3 else x[4:] for x in sessionNames]
#     for i, sn in enumerate(sessionFolderNames):
#         print(f'Processing JK{mouse:03} upper volume, session {sn}.')
#         upperMimgList = []
#         upperRegimgList = []
#         for pi in range(1,5):
#             piDir = f'{h5Dir}{mouse:03}/plane_{pi}/{sn}/plane0/'
#             piBinFn = f'{piDir}data.bin'
#             piOpsFn = f'{piDir}ops.npy'
#             opsi = np.load(piOpsFn, allow_pickle=True).item()
#             Lx = opsi['Lx']
#             Ly = opsi['Ly']
#             piImg = opsi['meanImg'].copy().astype(np.float32)
#             rmin, rmax = np.percentile(piImg,1), np.percentile(piImg,99)
#             piImg = np.clip(piImg, rmin, rmax)
#             upperMimgList.append(piImg)
#             for pj in range(pi,5):
#                 if pi == pj: # same plane correlation
#                     nframes = opsi['nframes']
#                     with BinaryFile(Ly = Ly, Lx = Lx, read_filename = piBinFn) as f:
#                         inds1 = range(int(nframes/3))
#                         frames = f.ix(indices=inds1).astype(np.float32)
#                         mimg1 = frames.mean(axis=0)
#                         rmin, rmax = np.percentile(mimg1,1), np.percentile(mimg1,99)
#                         mimg1 = np.clip(mimg1, rmin, rmax)
        
#                         inds2 = range(int(nframes/3*2), nframes)
#                         frames = f.ix(indices=inds2).astype(np.float32)
#                         mimg2 = frames.mean(axis=0)
#                         rmin, rmax = np.percentile(mimg2,1), np.percentile(mimg2,99)
#                         mimg2 = np.clip(mimg2, rmin, rmax)
#                         upperCorr[pi-1,pi-1,i] = np.corrcoeff(mimg1.flatten(), mimg2.flatten())[0,1]
#                 else: # different plane correlation, after rigid registration
#                     pjDir = f'{h5Dir}{mouse:03}/plane_{pj}/{sn}/plane0/'
#                     pjOpsFn = f'{pjDir}ops.npy'
#                     opsj = np.load(pjOpsFn, allow_pickle=True).item()
#                     pjImg = opsj['meanImg'].copy().astype(np.float32)
#                     rmin, rmax = np.percentile(pjImg,1), np.percentile(pjImg,99)
#                     pjImg = np.clip(pjImg, rmin, rmax)
                    
#                     # rigid registration
#                     ymax, xmax,_,_,_ = phase_corr(piImg, pjImg)
#                     pjImgReg = rigid.shift_frame(frame=pjImg, dy=ymax, dx=xmax)
#                     upperCorr[pi-1,pj-1,i] = np.corrcoef(piImg.flatten(), pjImg.flatten())[0,1]
#                     upperRegimgList.append(pjImgReg)
#         upper = {}
#         upper['upperMimgList'] = upperMimgList
#         upper['upperRegimgList'] = upperRegimgList
#         savefn = f'mean_img_reg_{sn}_upper'
#         np.save(f'{h5Dir}{mouse:03}/{savefn}', upper)
#     # Then lower volume
#     pn = 5
#     planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
#     sessionNames = get_session_names(planeDir, mouse, pn)
#     nSessions = len(sessionNames)
#     lowerCorr = np.zeros((4,4,nSessions))
#     sessionFolderNames = [x.split('_')[1] if len(x.split('_')[1])==3 else x[4:] for x in sessionNames]
#     for i, sn in enumerate(sessionFolderNames):
#         print(f'Processing JK{mouse:03} lower volume, session {sn}.')
#         lowerMimgList = []
#         lowerRegimgList = []
#         for pi in range(5,8):
#             piDir = f'{h5Dir}{mouse:03}/plane_{pi}/{sn}/plane0/'
#             piBinFn = f'{piDir}data.bin'
#             piOpsFn = f'{piDir}ops.npy'
#             opsi = np.load(piOpsFn, allow_pickle=True).item()
#             Lx = opsi['Lx']
#             Ly = opsi['Ly']
#             piImg = opsi['meanImg'].copy().astype(np.float32)
#             rmin, rmax = np.percentile(piImg,1), np.percentile(piImg,99)
#             piImg = np.clip(piImg, rmin, rmax)
#             lowerMimgList.append(piImg)
#             for pj in range(pi,8):
#                 if pi == pj: # same plane correlation
#                     nframes = opsi['nframes']
#                     with BinaryFile(Ly = Ly, Lx = Lx, read_filename = piBinFn) as f:
#                         inds1 = range(int(nframes/3))
#                         frames = f.ix(indices=inds1).astype(np.float32)
#                         mimg1 = frames.mean(axis=0)
#                         rmin, rmax = np.percentile(mimg1,1), np.percentile(mimg1,99)
#                         mimg1 = np.clip(mimg1, rmin, rmax)
        
#                         inds2 = range(int(nframes/3*2), nframes)
#                         frames = f.ix(indices=inds2).astype(np.float32)
#                         mimg2 = frames.mean(axis=0)
#                         rmin, rmax = np.percentile(mimg2,1), np.percentile(mimg2,99)
#                         mimg2 = np.clip(mimg2, rmin, rmax)
#                         lowerCorr[pi-5,pi-5,i] = np.corrcoef(mimg1.flatten(), mimg2.flatten())[0,1]
#                 else: # different plane correlation, after rigid registration
#                     pjDir = f'{h5Dir}{mouse:03}/plane_{pj}/{sn}/plane0/'
#                     pjOpsFn = f'{pjDir}ops.npy'
#                     opsj = np.load(pjOpsFn, allow_pickle=True).item()
#                     pjImg = opsj['meanImg'].copy().astype(np.float32)
#                     rmin, rmax = np.percentile(pjImg,1), np.percentile(pjImg,99)
#                     pjImg = np.clip(pjImg, rmin, rmax)
                    
#                     # rigid registration
#                     ymax, xmax,_,_,_ = phase_corr(piImg, pjImg)
#                     pjImgReg = rigid.shift_frame(frame=pjImg, dy=ymax, dx=xmax)
#                     lowerCorr[pi-5,pj-5,i] = np.corrcoef(piImg.flatten(), pjImg.flatten())[0,1]
#                     lowerRegimgList.append(pjImgReg)
#         lower = {}
#         lower['lowerMimgList'] = lowerMimgList
#         lower['lowerRegimgList'] = lowerRegimgList
#         savefn = f'mean_img_reg_{sn}_lower'
#         np.save(f'{h5Dir}{mouse:03}/{savefn}', lower)
#     # result = {}
#     # result['upperCorr'] = upperCorr
#     # result['lowerCorr'] = lowerCorr
#     # savefn = 'across_planes_corr'
#     # np.save(f'{h5Dir}{mouse:03}/{savefn}', result)

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
        for pj in range(pi+1,5):
        # for pj in range(pi+1,9):
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
            # pjImgReg = rigid.shift_frame(frame=pjImg, dy=ymax, dx=xmax)
            pjImgReg = np.roll(pjImgTemp, (ymax,xmax), axis=(0,1)) # 2021/09/06 JK
            upperRegimgList.append(pjImgReg)
    viewer.add_image(np.array(upperRegimgList), name=sn)
    

'''
The results look so much better than before.
Very few actually hit the maximum limited translation (20 pixels)
'''

#%%
#%%
#%%
#%% Run within-session and between-plane registration again, 
# using edge removal and limit on the translation pixels.
# Save with the same filenames (override previous results)
# 
''' mean_img_reg_{sn}_upper.npy, mean_img_reg_{sn}_lower.npy, saved in {h5Dir}{mouse:03} '''
edgeRem = 0.1 # Proportion of image in each axis to remove before registration
transLim = 20 # Number of pixels to limit translation for registration
# CLAHE parameters
kernel_size = [128,128] # default 1/8 of each dim
nbins = 2**8 # default 2**8
clip_limit = 0 # default 0.01

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0

def same_session_planes_reg(mouse, sessionName, startPlane, edgeRem = 0.1, transLim = 20):
    mimgList = []
    regimgList = []
    corrVals = np.zeros((4,4))
    corrVals_clahe = np.zeros((4,4))
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
                    tempIm1 = clahe_each(mimg1, kernel_size = kernel_size, clip_limit = clip_limit, nbins = nbins)
                    tempIm2 = clahe_each(mimg2, kernel_size = kernel_size, clip_limit = clip_limit, nbins = nbins)
                    corrVals_clahe[i,i] = np.corrcoef(tempIm1.flatten(), tempIm2.flatten())[0,1]
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
                
                tempIm1 = clahe_each(piImg[transLim:-transLim, transLim:-transLim], kernel_size = kernel_size, clip_limit = clip_limit, nbins = nbins)
                tempIm2 = clahe_each(pjImgReg[transLim:-transLim, transLim:-transLim], kernel_size = kernel_size, clip_limit = clip_limit, nbins = nbins)
                corrVals_clahe[i,j] = np.corrcoef(tempIm1.flatten(), tempIm2.flatten())[0,1]
                regimgList.append(pjImgReg)
    return mimgList, regimgList, corrVals, corrVals_clahe

# for mi in [0,3,8]:
for mi in [8]:    
    mouse = mice[mi]
    refSession = refSessions[mi]
    
    # upper volume first
    # Make a list of session names and corresponding files
    pn = 1
    planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
    sessionNames = get_session_names(planeDir, mouse, pn)
    nSessions = len(sessionNames)
    upperCorr = np.zeros((4,4,nSessions))
    upperCorr_clahe = np.zeros((4,4,nSessions))
    sessionFolderNames = [x.split('_')[1] if len(x.split('_')[1])==3 else x[4:] for x in sessionNames]
    for i, sn in enumerate(sessionFolderNames):
        print(f'Processing JK{mouse:03} upper volume, session {sn}.')
        upperMimgList, upperRegimgList, upperCorr[:,:,i], upperCorr_clahe[:,:,i] = same_session_planes_reg(mouse, sn, pn)
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
    lowerCorr_clahe = np.zeros((4,4,nSessions))
    sessionFolderNames = [x.split('_')[1] if len(x.split('_')[1])==3 else x[4:] for x in sessionNames]
    for i, sn in enumerate(sessionFolderNames):
        print(f'Processing JK{mouse:03} lower volume, session {sn}.')
        lowerMimgList, lowerRegimgList, lowerCorr[:,:,i], lowerCorr_clahe[:,:,i] = same_session_planes_reg(mouse, sn, pn)
        lower = {}
        lower['lowerMimgList'] = lowerMimgList
        lower['lowerRegimgList'] = lowerRegimgList
        savefn = f'mean_img_reg_{sn}_lower'
        np.save(f'{h5Dir}{mouse:03}/{savefn}', lower)
    corrSaveFn = f'same_session_regCorrVals_JK{mouse:03}'
    np.savez(f'{h5Dir}{mouse:03}/{corrSaveFn}', upperCorr = upperCorr, upperCorr_clahe = upperCorr_clahe, 
             lowerCorr = lowerCorr, lowerCorr_clahe = lowerCorr_clahe)


#%% Some within-correlation values are as low as <0.3
# Check those numbers and see what happend on those sessions
mi = 0
mouse = mice[mi]
corrs = np.load(f'{h5Dir}{mouse:03}/same_session_regCorrVals_JK{mouse:03}.npy', allow_pickle=True).item()
upper = corrs['upperCorr']
lower = corrs['lowerCorr']

f, ax = plt.subplots(2,4, figsize=(13,7), sharey=True)
ylimlow = 0.2
ytickstep = 0.2
for i in range(4):
    ax[0,i].plot(upper[i,i,:])
    ax[0,i].set_ylim(ylimlow,1)
    ax[0,i].set_yticks(np.arange(ylimlow,1,ytickstep))
    ax[0,i].set_title(f'Plane {i+1}', fontsize=15)
for i in range(4):
    ax[1,i].plot(lower[i,i,:])
    ax[1,i].set_ylim(ylimlow,1)
    ax[1,i].set_yticks(np.arange(ylimlow,1,ytickstep))
    ax[1,i].set_title(f'Plane {i+5}', fontsize=15)
ax[0,0].set_ylabel('Pixelwise correlation', fontsize=15)
ax[1,0].set_ylabel('Pixelwise correlation', fontsize=15)
ax[1,0].set_xlabel('Sessions', fontsize=15)

f.suptitle(f'JK{mouse:03}', fontsize=20)
f.tight_layout()

#%% CLAHE plot
mi = 8
mouse = mice[mi]
corrs = np.load(f'{h5Dir}{mouse:03}/same_session_regCorrVals_JK{mouse:03}.npz')
upper = corrs['upperCorr']
lower = corrs['lowerCorr']
upper_clahe = corrs['upperCorr_clahe']
lower_clahe = corrs['lowerCorr_clahe']

f, ax = plt.subplots(2,4, figsize=(13,7), sharey=True)
ylimlow = 0.2
ytickstep = 0.2
for i in range(4):
    ax[0,i].plot(upper_clahe[i,i,:])
    ax[0,i].plot(upper[i,i,:])
    
    ax[0,i].set_ylim(ylimlow,1)
    ax[0,i].set_yticks(np.arange(ylimlow,1,ytickstep))
    ax[0,i].set_title(f'Plane {i+1}', fontsize=15)
for i in range(4):
    ax[1,i].plot(lower_clahe[i,i,:])
    ax[1,i].plot(lower[i,i,:])
    
    ax[1,i].set_ylim(ylimlow,1)
    ax[1,i].set_yticks(np.arange(ylimlow,1,ytickstep))
    ax[1,i].set_title(f'Plane {i+5}', fontsize=15)
ax[0,0].set_ylabel('Pixelwise correlation', fontsize=15)
ax[1,0].set_ylabel('Pixelwise correlation', fontsize=15)
ax[1,0].set_xlabel('Sessions', fontsize=15)

f.suptitle(f'JK{mouse:03}', fontsize=20)
f.tight_layout()



#%% CLAHE image example






#%% Check session names
pn = 1
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
sessionNames = get_session_names(planeDir, mouse, pn)
nSessions = len(sessionNames)
upperCorr = np.zeros((4,4,nSessions))
sessionFolderNames = [x.split('_')[1] if len(x.split('_')[1])==3 else x[4:] for x in sessionNames]

#%% Check images in the specified session
# CLAHE parameters
kernel_size = [32,32] # default 1/8 of each dim
nbins = 2**8 # default 2**8
clip_limit = 0 # default 0.01

pn = 1
si = 13
sn = sessionFolderNames[si]

ops = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/{sn}/plane0/ops.npy', allow_pickle=True).item()
Ly, Lx, nframes = ops['Ly'], ops['Lx'], ops['nframes']
binFn = f'{h5Dir}{mouse:03}/plane_{pn}/{sn}/plane0/data.bin'
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binFn) as f:
    inds1 = range(int(nframes/3))
    frames = f.ix(indices=inds1).astype(np.float32)
    mimg1 = frames.mean(axis=0)
    rmin, rmax = np.percentile(mimg1,1), np.percentile(mimg1,99)
    # mimg1 = np.clip(mimg1, rmin, rmax)
    mimg1 = (np.clip(mimg1, rmin, rmax) - rmin) / (rmax-rmin)
    
    inds2 = range(int(nframes/3*2), nframes)
    frames = f.ix(indices=inds2).astype(np.float32)
    mimg2 = frames.mean(axis=0)
    rmin, rmax = np.percentile(mimg2,1), np.percentile(mimg2,99)
    # mimg2 = np.clip(mimg2, rmin, rmax)
    mimg2 = (np.clip(mimg2, rmin, rmax) - rmin) / (rmax-rmin)
    
if mouse > 50:
    mimg1 = mimg1[:,:-70]
    mimg2 = mimg2[:,:-70]
    Lx = Lx-70
viewer = napari.Viewer()
mixImg = np.zeros((Ly,Lx,3), dtype=np.float32)
mixImg[:,:,0] = mimg1
mixImg[:,:,2] = mimg1
mixImg[:,:,1] = mimg2
viewer.add_image(mixImg, rgb=True)
viewer.add_image(np.stack([mimg1,mimg2],axis=0), name=f'JK{mouse:03} si {si} plane{pn}')

claheImg1 = clahe_each(mimg1, kernel_size = kernel_size, clip_limit = clip_limit, nbins = nbins)
claheImg2 = clahe_each(mimg2, kernel_size = kernel_size, clip_limit = clip_limit, nbins = nbins)
mixClahe = np.zeros((Ly,Lx,3), dtype=np.float32)
mixClahe[:,:,0] = claheImg1
mixClahe[:,:,2] = claheImg1
mixClahe[:,:,1] = claheImg2
viewer.add_image(mixClahe, rgb=True)
viewer.add_image(np.stack([claheImg1,claheImg2],axis=0), name=f'JK{mouse:03} si {si} plane{pn}')

'''
There were some shadow appearing in some of the sessions in JK052.
It can be either mouse-specific or because of using US gel (bubble inside due to heating?).
Test in other mice in the same batch (JK053, 054, 056)
(Do this on 12-core WS)
'''



'''
For now, let's assume that it's just some sessions of JK052.
Proceed the analysis.
'''
#%% JK052 same-plane correlation values

si2rem052 = [13,16,17,18,20,21,22,24,25,26]
# sn2check052 = [16,19,20,21,23,24,25,27,28,29]
mi = 8
mouse = mice[mi]
corrs = np.load(f'{h5Dir}{mouse:03}/same_session_regCorrVals_JK{mouse:03}.npy', allow_pickle=True).item()
upper = corrs['upperCorr']
lower = corrs['lowerCorr']
if mouse == 52:
    # si2rem052 = np.where(upper[0,0,:]<0.9) # same as manually selected session indice
    showinds = [i for i in range(upper.shape[2]) if i not in si2rem052]
else:
    showinds = [i for i in range(upper.shape[2])]

f, ax = plt.subplots(2,4, figsize=(13,7), sharey=True)
ylimlow = 0.8
ytickstep = 0.1
for i in range(4):
    ax[0,i].plot(upper[i,i,showinds])
    ax[0,i].set_ylim(ylimlow,1)
    ax[0,i].set_yticks(np.arange(ylimlow,1,ytickstep))
    ax[0,i].set_title(f'Plane {i+1}', fontsize=15)
for i in range(4):
    ax[1,i].plot(lower[i,i,showinds])
    ax[1,i].set_ylim(ylimlow,1)
    ax[1,i].set_yticks(np.arange(ylimlow,1,ytickstep))
    ax[1,i].set_title(f'Plane {i+5}', fontsize=15)
ax[0,0].set_ylabel('Pixelwise correlation', fontsize=15)
ax[1,0].set_ylabel('Pixelwise correlation', fontsize=15)
ax[1,0].set_xlabel('Sessions', fontsize=15)

f.suptitle(f'JK{mouse:03}', fontsize=20)
f.tight_layout()

#%% Look at between-plane correlation values
mi = 0
mouse = mice[mi]
corrs = np.load(f'{h5Dir}{mouse:03}/same_session_regCorrVals_JK{mouse:03}.npy', allow_pickle=True).item()
upper = corrs['upperCorr']
lower = corrs['lowerCorr']
if mouse == 52:
    # si2rem052 = np.where(upper[0,0,:]<0.9) # same as manually selected session indice
    # showinds = [i for i in range(upper.shape[2]) if i not in si2rem052]
    showinds = [i for i in range(upper.shape[2])]
else:
    showinds = [i for i in range(upper.shape[2])]

f, ax = plt.subplots(2,4, figsize=(13,7), sharey=True)
ylimlow = 0
ytickstep = 0.1
for i in range(4):
    for j in range(i,4):
        ax[0,i].plot(upper[i,j,showinds], color=((j-i)*0.33, 0, (j-i)*0.33))
    if i ==0:
        ax[0,i].legend(['0 diff', '1 diff', '2 diff', '3 diff'])
    ax[0,i].set_ylim(ylimlow,1)
    ax[0,i].set_yticks(np.arange(ylimlow,1,ytickstep))
    ax[0,i].set_title(f'Plane {i+1}', fontsize=15)
for i in range(4):
    for j in range(i,4):
        ax[1,i].plot(lower[i,j,showinds], color=((j-i)*0.33, 0, (j-i)*0.33))
        
    ax[1,i].set_ylim(ylimlow,1)
    ax[1,i].set_yticks(np.arange(ylimlow,1,ytickstep))
    ax[1,i].set_title(f'Plane {i+5}', fontsize=15)
ax[0,0].set_ylabel('Pixelwise correlation', fontsize=15)
ax[1,0].set_ylabel('Pixelwise correlation', fontsize=15)
ax[1,0].set_xlabel('Sessions', fontsize=15)

f.suptitle(f'JK{mouse:03}', fontsize=20)
f.tight_layout()

f, ax = plt.subplots(1,4, figsize = (13,4), sharey = True)
ylimlow = 0.2
ytickstep = 0.1

for i in range(4):
    for j in range(i,4):
        ax[j-i].plot(upper[i,j,showinds], color=(0, 1-i/8, 1-i/8))
for i in range(4):
    for j in range(i,4):
        ax[j-i].plot(lower[i,j,showinds], color=(0, 1-(i+4)/8, 1-(i+4)/8))
for pdiff in range(4):
    ax[pdiff].set_title(f'{pdiff} plane diff', fontsize=15)
ax[0].legend([f'Plane {i}' for i in range(1,9)])
ax[0].set_ylabel('Pixelwise correlation', fontsize=15)
ax[0].set_xlabel('Sessions', fontsize=15)
f.suptitle(f'JK{mouse:03}', fontsize=20)
f.tight_layout()

'''
For same-plane, all planes have similarly very high correlation (of course).
Between-plane correlation values are different between volume.
For comparison, upper and lower volume should be divided.
'''
#%% Compare between different nonrigid registration parameters
edgeRem = 0.2 # Proportion of image in each axis to remove before calculating correlations

mi = 8
mouse = mice[mi]
f, ax = plt.subplots(2,4, figsize=(13,7))
corrMouse = []
for pn in range(1,9):
    planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
    sessionNames = get_session_names(planeDir, mouse, pn)    
    corrPlane = np.zeros((len(sessionNames),3))
    refSn = refSessions[mi]
    refSi = [i for i,sn in enumerate(sessionNames) if f'{mouse:03}_{refSn:03}_' in sn][0]
    # refSi = np.where(np.array(sessionNames) == f'{mouse:03}_{refSn:03}_')[0][0] # I don't like np.where...
    ai = np.unravel_index(pn-1, (2,4))
    for param in range(3):
        op = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/s2p_nr_test_{param:02}.npy',
                     allow_pickle = True).item()
        refImg = op['regImgs'][refSi]
        Ly, Lx = refImg.shape
        if mouse > 50:
            refImg = refImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-70] # 052 has blank edge on the right end for ~70 pix
        else:
            refImg = refImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1] # Ly is always shorter than Lx
        rmin, rmax = np.percentile(refImg,1), np.percentile(refImg,99)
        refImg = np.clip(refImg, rmin, rmax)
        
        for si in range(len(op['regImgs'])):
            testImg = op['regImgs'][si]
            if mouse > 50:
                testImg = testImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-70] # 052 has blank edge on the right end for ~70 pix
            else:
                testImg = testImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1] # Ly is always shorter than Lx
            rmin, rmax = np.percentile(testImg,1), np.percentile(testImg,99)
            testImg = np.clip(testImg, rmin, rmax)
            corrPlane[si,param] = np.corrcoef(refImg.flatten(), testImg.flatten())[0,1]
        ax[ai[0],ai[1]].plot(corrPlane[:,param])
        
    corrMouse.append(corrPlane)
ax[0,0].legend(['block size 128', '64', '32'])
ax[1,0].set_xlabel('Session', fontsize=15)
ax[1,0].set_ylabel('Correlation to the ref img', fontsize=15)
f.suptitle(f'JK{mouse:03}', fontsize=20)
f.tight_layout()

'''
Block size 128 looks the best.
Too small block sizes could hurt.
'''

#%% Test maxregshiftNR parameter
# with block size fixed to [128,128]
mnrList = [5,10,15,20]

edgeRem = 0.2 # Proportion of image in each axis to remove before calculating correlations
blankEdge = 70 # For mouse > 50, where there is a blank edge on the right side

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['snr_thresh'] = 1.2
op['block_size'] = [128, 128]

for mi in [0,3,8]:
    mouse = mice[mi]
    refSession = refSessions[mi]
    for pn in range(1,9):
        print(f'Processing JK{mouse:03} plane {pn}')
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
            Ly, Lx = refImg.shape
## Perform suite2p nonrigid registration.
        for mnr in mnrList:
            op['maxregshiftNR'] = mnr
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
            
            saveFn = f'{planeDir}s2p_nr_test_mnr{mnr:02}'
            np.save(saveFn, op)

#%% Plot the result
mnrList = [5,10,15,20]
for mi in [0,3,8]:
    mouse = mice[mi]
    f, ax = plt.subplots(2,4,figsize = (13,7), sharey = True, sharex = True)
    for pn in range(1,9):
        planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
        axind = np.unravel_index(pn-1, (2,4))
        for mnr in mnrList:
            loadFn =  f'{planeDir}s2p_nr_test_mnr{mnr:02}.npy'
            op = np.load(loadFn, allow_pickle=True).item()
            corrVals = op['corrVals']
            ax[axind[0],axind[1]].plot(corrVals)
        ax[axind[0],axind[1]].legend([mnr for mnr in mnrList])
        ax[axind[0],axind[1]].set_title(f'Plane {pn}', fontsize = 15)
    ax[1,0].set_ylabel('Correlation', fontsize=15)
    ax[1,0].set_xlabel('Sessions', fontsize=15)
    f.suptitle(f'JK{mouse:03}', fontsize=20)
    f.tight_layout()
    



    
'''
15=10>=20>>5
Choose 15.
Too large maxregshiftNR could hurt.
'''





#%%
#%%
#%%
#%% Run nonrigid across sessions
# with block size 128, maxregshiftNR 15
# Calculate pixel correlation between each mimg to ref mimg

edgeRem = 0.1 # Proportion of image in each axis to remove before calculating correlations
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
    for pn in range(1,9):
        planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
        refDir = f'{planeDir}{refSession:03}/plane0/'
        ops = np.load(f'{refDir}ops.npy', allow_pickle=True).item()
        refImg = ops['meanImg'].astype(np.float32)
        Ly = ops['Ly']
        # rmin, rmax = np.int16(np.percentile(refImg,1)), np.int16(np.percentile(refImg,99))
        # rmin, rmax = np.percentile(refImg,1), np.percentile(refImg,99)
        # refImg = np.clip(refImg, rmin, rmax)
        
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
            # rmin, rmax = np.percentile(tempImg,1), np.percentile(tempImg,99)
            # tempImg = np.clip(tempImg, rmin, rmax)
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

#%%
#%%
#%%
#%% Compare within-session correlation with session-to-session pixel correlation values
# Can be used for session selection

# Divide into upper volume and lower volume to calculate 0-, 1-, 2-, and 3-plane diff pixel correlation values (mean +/- SD)
# This value will be applied to all 4 planes
# In each plane, calculate the correlation between mimgs of each session with that of the reference session

edgeRem = 0.2 # Proportion of image in each axis to remove before calculating correlations

mi = 0
mouse = mice[mi]
f, ax = plt.subplots(2,4, figsize=(13,7))
corrMouse = []
for pn in range(1,9):
    planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
    sessionNames = get_session_names(planeDir, mouse, pn)    
    corrPlane = np.zeros((len(sessionNames),3))
    refSn = refSessions[mi]
    refSi = [i for i,sn in enumerate(sessionNames) if f'{mouse:03}_{refSn:03}_' in sn][0]
    # refSi = np.where(np.array(sessionNames) == f'{mouse:03}_{refSn:03}_')[0][0] # I don't like np.where...
    ai = np.unravel_index(pn-1, (2,4))
    for param in range(3):
        op = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/s2p_nr_test_{param:02}.npy',
                     allow_pickle = True).item()
        refImg = op['regImgs'][refSi]
        Ly, Lx = refImg.shape
        if mouse > 50:
            refImg = refImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-70] # 052 has blank edge on the right end for ~70 pix
        else:
            refImg = refImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1] # Ly is always shorter than Lx
        rmin, rmax = np.percentile(refImg,1), np.percentile(refImg,99)
        refImg = np.clip(refImg, rmin, rmax)
        
        for si in range(len(op['regImgs'])):
            testImg = op['regImgs'][si]
            if mouse > 50:
                testImg = testImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1-70] # 052 has blank edge on the right end for ~70 pix
            else:
                testImg = testImg[int(Ly*edgeRem):-int(Ly*edgeRem)-1, int(Ly*edgeRem):-int(Ly*edgeRem)-1] # Ly is always shorter than Lx
            rmin, rmax = np.percentile(testImg,1), np.percentile(testImg,99)
            testImg = np.clip(testImg, rmin, rmax)
            corrPlane[si,param] = np.corrcoef(refImg.flatten(), testImg.flatten())[0,1]
        ax[ai[0],ai[1]].plot(corrPlane[:,param])
        
    corrMouse.append(corrPlane)
ax[0,0].legend(['block size 128', '64', '32'])
ax[1,0].set_xlabel('Session', fontsize=15)
ax[1,0].set_ylabel('Correlation to the ref img', fontsize=15)
f.suptitle(f'JK{mouse:03}', fontsize=20)
f.tight_layout()











'''
Great, but not that perfect registration.
Try other nonrigid methods!
'''

#%% Optical flow iLK and TV-L1 from skimage
# Test images from JK025 plane 2 session 004 (ref, or target) and 019 (moving)
# Compare with suite2p nonrigid registration
from skimage.registration import optical_flow_ilk as iLK
from skimage.registration import optical_flow_tvl1 as TVL1
from skimage.transform import warp

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

mi = 0
pn = 2
refSi = 3
testSi = 12
testSn = 19

mouse = mice[mi]
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'

# Load reference image and s2p registered image
s2preg = np.load(f'{planeDir}s2p_nr_reg.npy', allow_pickle = True).item()
s2pRefImg = s2preg['regImgs'][refSi,:,:]
s2pRegImg = s2preg['regImgs'][testSi,:,:]
s2pBlend = imblend(s2pRefImg, s2pRegImg)

#%%
# Load reference image
sessionNames = get_session_names(planeDir, mouse, pn)
refSn = sessionNames[refSi].split('_')[1]
ops = np.load(f'{planeDir}{refSn}/plane0/ops.npy', allow_pickle=True).item()
refImg = ops['meanImg']

# Load test image
ops = np.load(f'{planeDir}{testSn:03}/plane0/ops.npy', allow_pickle=True).item()
testImg = ops['meanImg']


# iLK
v, u = iLK(refImg, testImg)
Ly, Lx = testImg.shape
row_coords, col_coords = np.meshgrid(np.arange(Ly), np.arange(Lx), indexing='ij')
ilkRegImg = warp(testImg, np.array([row_coords + v, col_coords + u]), mode='edge')
ilkBlend = imblend(refImg, ilkRegImg)

# TV-L1
v, u = TVL1(refImg, testImg)
tvlRegImg = warp(testImg, np.array([row_coords + v, col_coords + u]), mode='edge')
tvlBlend = imblend(refImg, tvlRegImg)

#%% Show registered images
viewer = napari.Viewer()
viewer.add_image(np.array([np.moveaxis(np.tile(s2pRefImg/np.amax(s2pRefImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(s2pRegImg/np.amax(s2pRegImg), (3,1,1)),0,-1), 
                           s2pBlend/np.amax(s2pBlend)]), name='Suite2p')
viewer.add_image(np.array([np.moveaxis(np.tile(refImg/np.amax(refImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(ilkRegImg/np.amax(ilkRegImg), (3,1,1)),0,-1), 
                           ilkBlend/np.amax(ilkBlend)]), name='iLK')
viewer.add_image(np.array([np.moveaxis(np.tile(refImg/np.amax(refImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(tvlRegImg/np.amax(tvlRegImg), (3,1,1)),0,-1), 
                           tvlBlend/np.amax(tvlBlend)]), name='TV-L1')

#%% Show optical flows






mi = 0
pn = 2
refSi = 3
testSi = 12
testSn = 19

mouse = mice[mi]
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'

# Load s2p reference image and s2p registered image
s2preg = np.load(f'{planeDir}s2p_nr_reg.npy', allow_pickle = True).item()
s2pRefImg = s2preg['regImgs'][refSi,:,:]
s2pRegImg = s2preg['regImgs'][testSi,:,:]
s2pBlend = imblend(s2pRefImg, s2pRegImg)

# Load reference image
sessionNames = get_session_names(planeDir, mouse, pn)
refSn = sessionNames[refSi].split('_')[1]
ops = np.load(f'{planeDir}{refSn}/plane0/ops.npy', allow_pickle=True).item()
refImg = ops['meanImg'].astype(np.float32)
# refImg = ((refImg-np.amax(refImg))/np.amax(refImg)*255).astype(np.float32)
# Load test image
ops = np.load(f'{planeDir}{testSn:03}/plane0/ops.npy', allow_pickle=True).item()
testImg = ops['meanImg'].astype(np.float32)
# testImg = ((testImg-np.amax(testImg))/np.amax(testImg)*255).astype(np.float32)
f, ax = plt.subplots(1,2)
ax[0].imshow(refImg)
ax[1].imshow(testImg)
#%%
import itk
# from itkwidgets import compare
# compare(refImg, testImg)

#%%

param_obj = itk.ParameterObject.New()
def_affine_param_map= param_obj.GetDefaultParameterMap('affine',4)
def_affine_param_map['FinalBSplineInterpolationOrder'] = ['3']
param_obj.AddParameterMap(def_affine_param_map)

def_bspline_param_map= param_obj.GetDefaultParameterMap('bspline',4)
def_bspline_param_map['FinalBSplineInterpolationOrder'] = ['3']
param_obj.AddParameterMap(def_bspline_param_map)

result_img, transform_param = itk.elastix_registration_method(
    refImg, testImg,
    parameter_object = param_obj,
    log_to_console=False)
#%%
plt.imshow(result_img, cmap='gray')

#%%
import SimpleITK as sitk
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
#%%

viewer = napari.Viewer()
viewer.add_image(np.array([np.moveaxis(np.tile(s2pRefImg/np.amax(s2pRefImg), (3,1,1)),0,-1), 
                            np.moveaxis(np.tile(s2pRegImg/np.amax(s2pRegImg), (3,1,1)),0,-1), 
                            np.moveaxis(np.tile(testImg/np.amax(testImg), (3,1,1)),0,-1), 
                            s2pBlend/np.amax(s2pBlend)]), name='Suite2p')

# viewer.add_image(np.array([np.moveaxis(np.tile(s2pRefImg, (3,1,1)),0,-1), 
#                            np.moveaxis(np.tile(s2pRegImg, (3,1,1)),0,-1), 
#                            np.moveaxis(np.tile(testImg/np.amax(testImg), (3,1,1)),0,-1), 
#                            s2pBlend/np.amax(s2pBlend)]), name='Suite2p')


viewer.add_image(np.array([np.moveaxis(np.tile(demonRefImg/np.amax(demonRefImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(demonRegImg/np.amax(demonRegImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(testImg/np.amax(testImg), (3,1,1)),0,-1), 
                           demonBlend/np.amax(demonBlend)]), name='demon')

viewer.add_image(np.array([np.moveaxis(np.tile(bsplineRefImg/np.amax(bsplineRefImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(bsplineRegImg/np.amax(bsplineRegImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(testImg/np.amax(testImg), (3,1,1)),0,-1), 
                           bsplineBlend/np.amax(bsplineBlend)]), name='bspline')



'''
Generally bspline and demons work better than suite2p piece-wise flow,
but dim regions show worse matching.
What if I apply clahe first?
'''
#%%
fixed_clahe = sitk.GetImageFromArray(clahe_each(refImg))
moving_clahe = sitk.GetImageFromArray(clahe_each(testImg))
moving = sitk.GetImageFromArray(testImg)


#%% bspline registration
transformDomainMeshSize = [8] * moving.GetDimension()
tx = sitk.BSplineTransformInitializer(fixed_clahe,
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

outTx = R.Execute(fixed_clahe, moving_clahe)
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(100)
resampler.SetTransform(outTx)

out = resampler.Execute(moving)
resultImg = sitk.GetArrayFromImage(out)

bsplineRefImg = (refImg)
bsplineRegImg = (resultImg)
bsplineBlend = imblend(bsplineRefImg, bsplineRegImg)

#%% demons registration
demons = sitk.DemonsRegistrationFilter()
demons.SetNumberOfIterations(1000)
demons.SetStandardDeviations(10.0)

displacementField = demons.Execute(fixed_clahe, moving_clahe)

outTx = sitk.DisplacementFieldTransform(displacementField)

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed_clahe)
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
demonRefImg = (refImg)
demonRegImg = (resultImg)
demonBlend = imblend(demonRefImg, demonRegImg)

#%% Show results

viewer = napari.Viewer()
viewer.add_image(np.array([np.moveaxis(np.tile(s2pRefImg/np.amax(s2pRefImg), (3,1,1)),0,-1), 
                            np.moveaxis(np.tile(s2pRegImg/np.amax(s2pRegImg), (3,1,1)),0,-1), 
                            np.moveaxis(np.tile(testImg/np.amax(testImg), (3,1,1)),0,-1), 
                            s2pBlend/np.amax(s2pBlend)]), name='Suite2p')

viewer.add_image(np.array([np.moveaxis(np.tile(demonRefImg/np.amax(demonRefImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(demonRegImg/np.amax(demonRegImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(testImg/np.amax(testImg), (3,1,1)),0,-1), 
                           demonBlend/np.amax(demonBlend)]), name='demon')

viewer.add_image(np.array([np.moveaxis(np.tile(bsplineRefImg/np.amax(bsplineRefImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(bsplineRegImg/np.amax(bsplineRegImg), (3,1,1)),0,-1), 
                           np.moveaxis(np.tile(testImg/np.amax(testImg), (3,1,1)),0,-1), 
                           bsplineBlend/np.amax(bsplineBlend)]), name='bspline')







'''
No, the results look almost the same.
CLAHE does not help dim area registration.
Suite2p registration seems the best.
'''









#%% The last test about best nonrigid registraion test
# Using meanImgE (high-frequency enhanced image) for registration
# 2-step registration
# Adjusting block sizes

# Test with visual inspection, correlation values, and sum of squared errors (normalized by # of pixels)
# Use JK025 plane 3, session 004 and 019. Then expand to other sessions

mi = 0
mouse = mice[mi]
pn = 3
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
sn1 = 4
sn2 = 19

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['maxregshiftNR'] = 15
op['snr_thresh'] = 1.2
# op['block_size'] = [128, 128]

ops1 = np.load(f'{planeDir}{sn1:03}/plane0/ops.npy', allow_pickle=True).item()
meanImg1 = ops1['meanImg']
meanImgE1 = ops1['meanImgE']

op['Ly'] = ops1['Ly']
op['Lx'] = ops1['Lx']

ops2 = np.load(f'{planeDir}{sn2:03}/plane0/ops.npy', allow_pickle=True).item()
meanImg2 = ops2['meanImg']
meanImgE2 = ops2['meanImgE']

# (0) Previous method (1-step, bs[128, 128])
op['block_size'] = [128, 128]
t0Result,_,_ = s2p_nonrigid_registration([meanImg2], meanImg1, op)

# (1) Use meanImg, 2-step registration, bs [128,128]
op['block_size'] = [128, 128]
t1ResultTemp,_,_ = s2p_nonrigid_registration([meanImg2], meanImg1, op)
t1Result,_,_ = s2p_nonrigid_registration(t1ResultTemp, meanImg1, op)

# (2) Use meanImg, 2-step registration, bs [128,128] and then [64, 64]
op['block_size'] = [128, 128]
t2ResultTemp,_,_ = s2p_nonrigid_registration([meanImg2], meanImg1, op)
op['block_size'] = [64, 64]
op['maxregshiftNR'] = 5
t2Result,_,_ = s2p_nonrigid_registration(t2ResultTemp, meanImg1, op)

# (3) Use meanImgE, 1-step registration, bs [128,128]
op['block_size'] = [128, 128]
op['maxregshiftNR'] = 15
_, rigid_offsets, nonrigid_offsets = s2p_nonrigid_registration([meanImgE2], meanImgE1, op)
t3Result = s2p_register([meanImg2], op, rigid_offsets, nonrigid_offsets)

# (4) Use meanImgE, 1-step registration, bs [64,64]
op['block_size'] = [64, 64]
op['maxregshiftNR'] = 5
_, rigid_offsets, nonrigid_offsets = s2p_nonrigid_registration([meanImgE2], meanImgE1, op)
t4Result = s2p_register([meanImg2], op, rigid_offsets, nonrigid_offsets)

# (5) Use meanImgE, 2-step registration, bs [128,128] twice
op['block_size'] = [128, 128]
op['maxregshiftNR'] = 15
_, rigid_offsets, nonrigid_offsets = s2p_nonrigid_registration([meanImgE2], meanImgE1, op)
t5ResultTemp = s2p_register([meanImgE2], op, rigid_offsets, nonrigid_offsets)
t5ResultTempMoving = s2p_register([meanImg2], op, rigid_offsets, nonrigid_offsets)
_, rigid_offsets, nonrigid_offsets = s2p_nonrigid_registration(t5ResultTemp, meanImgE1, op)
t5Result = s2p_register(t5ResultTempMoving, op, rigid_offsets, nonrigid_offsets)

# (6) Use meanImgE, 2-step registration, bs [128,128] and then [64, 64]
op['block_size'] = [128, 128]
op['maxregshiftNR'] = 15
_, rigid_offsets, nonrigid_offsets = s2p_nonrigid_registration([meanImgE2], meanImgE1, op)
t6ResultTemp = s2p_register([meanImgE2], op, rigid_offsets, nonrigid_offsets)
t6ResultTempMoving = s2p_register([meanImg2], op, rigid_offsets, nonrigid_offsets)
op['block_size'] = [64, 64]
op['maxregshiftNR'] = 5
_, rigid_offsets, nonrigid_offsets = s2p_nonrigid_registration(t6ResultTemp, meanImgE1, op)
t6Result = s2p_register(t6ResultTempMoving, op, rigid_offsets, nonrigid_offsets)


#%% Show the result
def minmaxnorm(mat):
    return (mat-np.amin(mat))/(np.amax(mat) - np.amin(mat))

viewer = napari.Viewer()


results = [t0Result, t1Result, t2Result, t3Result, t4Result, t5Result, t6Result]
refrgb = np.moveaxis(np.tile(minmaxnorm(meanImg1), (3,1,1)),0,-1)
for result in results:
    resultrgb = np.moveaxis(np.tile(minmaxnorm(result), (3,1,1)),0,-1)
    blended = imblend(meanImg1, result[0,:,:])
    viewer.add_image(np.array([refrgb, resultrgb, minmaxnorm(blended)]))
    


''' 
Using meanImgE does not help much.
But, 2-step with mean image definitely helps. Seems like smaller blocksize helps.
Try with smaller block sizes
'''

#%%
# (0) Previous method (1-step, bs[128, 128])
op['block_size'] = [128, 128]
t0Result,_,_ = s2p_nonrigid_registration([meanImg2], meanImg1, op)

# (1) Use meanImg, 2-step registration, bs [128,128]
op['block_size'] = [128, 128]
op['maxregshiftNR'] = 5
# t1ResultTemp,_,_ = s2p_nonrigid_registration([meanImg2], meanImg1, op)
t1Result,_,_ = s2p_nonrigid_registration(t0Result, meanImg1, op)

# (2) Use meanImg, 2-step registration, bs [128,128] and then [64, 64]
# op['block_size'] = [128, 128]
# t2ResultTemp,_,_ = s2p_nonrigid_registration([meanImg2], meanImg1, op)
op['block_size'] = [64, 64]
op['maxregshiftNR'] = 5
t2Result,_,_ = s2p_nonrigid_registration(t0Result, meanImg1, op)

# (3) Use meanImg, 2-step registration, bs [128,128] and then [32, 32]
# op['block_size'] = [128, 128]
# t3ResultTemp,_,_ = s2p_nonrigid_registration([meanImg2], meanImg1, op)
op['block_size'] = [32, 32]
op['maxregshiftNR'] = 3
t3Result,_,_ = s2p_nonrigid_registration(t0Result, meanImg1, op)

# (3) Use meanImg, 2-step registration, bs [128,128] and then [16, 16]
# op['block_size'] = [128, 128]
# t3ResultTemp,_,_ = s2p_nonrigid_registration([meanImg2], meanImg1, op)
op['block_size'] = [16, 16]
op['maxregshiftNR'] = 2
t4Result,_,_ = s2p_nonrigid_registration(t0Result, meanImg1, op)

viewer = napari.Viewer()
results = [t0Result, t1Result, t2Result, t3Result, t4Result]
refrgb = np.moveaxis(np.tile(minmaxnorm(meanImg1), (3,1,1)),0,-1)
movingrgb = np.moveaxis(np.tile(minmaxnorm(meanImg2), (3,1,1)),0,-1)
for result in results:
    resultrgb = np.moveaxis(np.tile(minmaxnorm(result), (3,1,1)),0,-1)
    # blended = imblend(meanImg1, result[0,:,:])
    # viewer.add_image(np.array([refrgb, resultrgb, minmaxnorm(blended)]))
    viewer.add_image(np.array([refrgb, resultrgb, movingrgb]))
    
    
'''
Smaller block sizes (<64) makes highly distorted local flows.
No help at all (even with low maxregshiftNR)
Mostly rotation. Need to fix this. For both session-stitched or individual suite2p
'''


#%% Rigid body registration (translation and rotation)
# (1) using skimage
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate, warp_polar
from skimage.filters import difference_of_gaussians
from scipy.fftpack import fft2, fftshift

#%%
fixed = difference_of_gaussians(meanImg1, 2, 100)
moving = difference_of_gaussians(meanImg2, 2, 100)

fixedfs = np.abs(fftshift(fft2(fixed)))
movingfs = np.abs(fftshift(fft2(moving)))

shape = fixedfs.shape
radius = shape[1]
warped_fixedfs = warp_polar(fixedfs, radius = radius, scaling='log', order=0)
warped_movingfs = warp_polar(movingfs, radius = radius, scaling='log', order=0)

warped_fixedfs = warped_fixedfs[:shape[0]//2, :]
warped_movingfs = warped_movingfs[:shape[0]//2, :]

shifts, _,_ = phase_cross_correlation(warped_fixedfs, warped_movingfs, upsample_factor=10)
shiftr, shiftc = shifts[:2]
recovered_angle = (360 / shape[0]) * shiftr

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(fixed, cmap='gray')
ax[0,1].imshow(moving, cmap='gray')

ax[1,0].imshow(warped_fixedfs)
ax[1,1].imshow(warped_movingfs)




#%%
radius = ops1['Ly']//2
# fixed = difference_of_gaussians(meanImg1, 0, 20)
# moving = difference_of_gaussians(meanImg2, 0, 20)
fixed = meanImgE1
moving = meanImgE2

warp_fixed = warp_polar(fixed, radius = radius)
warp_moving = warp_polar(moving, radius = radius)

shifts, _, _ = phase_cross_correlation(warp_fixed[:,:180], warp_moving[:,:180], upsample_factor=100)


fig, ax = plt.subplots(2,2)
ax[0,0].imshow(fixed, cmap='gray')
ax[0,1].imshow(moving, cmap='gray')

ax[1,0].imshow(warp_fixed)
ax[1,1].imshow(warp_moving)
#%%
regImgRot = rotate(meanImg2, shifts[0])

napari.view_image(np.array([meanImg1, regImgRot, meanImg2]))

#%%
op['block_size'] = [128, 128]
op['maxregshiftNR'] = 5
regImgS2p,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
op['block_size'] = [32, 32]
op['maxregshiftNR'] = 3
regImgFin,_,_ = s2p_nonrigid_registration(regImgS2p, meanImg1, op)
napari.view_image(np.array([meanImg1, regImgFin[0], regImgRot, meanImg2]))
#%%

radius = ops1['Ly']//2
# fixed = difference_of_gaussians(meanImg1, 0, 20)
# moving = difference_of_gaussians(meanImg2, 0, 20)
fixed = meanImg1
moving = meanImg2

warp_fixed = warp_polar(fixed, radius = radius)
warp_moving = warp_polar(moving, radius = radius)

# shifts, _, _ = phase_cross_correlation(warp_fixed[:,:180], warp_moving[:,:180], upsample_factor=100)
shifts, _, _ = phase_cross_correlation(warp_fixed, warp_moving, upsample_factor=100)

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(fixed, cmap='gray')
ax[0,1].imshow(moving, cmap='gray')

ax[1,0].imshow(warp_fixed)
ax[1,1].imshow(warp_moving)

regImgRot = rotate(meanImg2, shifts[0])

op['block_size'] = [128, 128]
op['maxregshiftNR'] = 5
regImgS2p,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
op['block_size'] = [32, 32]
op['maxregshiftNR'] = 3
regImgFin,_,_ = s2p_nonrigid_registration(regImgS2p, meanImg1, op)
napari.view_image(np.array([minmaxnorm(meanImg1), minmaxnorm(regImgFin[0,:,:]), minmaxnorm(regImgRot), minmaxnorm(meanImg2)]))






'''
Rotation matching before suite2p nonrigid greatly improves. Almost perfect by eyes, no need to search more.
Just using mean image was good enough (no need to use enhanced image)
    There was some difference in best rotation, but <1 degree seems to be well handled by suite2p nonrigid
Now, is 2-step helping? What is the best parameter for 1-step after rotation matching?
'''
#%%
op['block_size'] = [128, 128]
op['maxregshiftNR'] = 10
regImg0,_,_ = s2p_nonrigid_registration([meanImg2], meanImg1, op)
# regImgRot = rotate(meanImg2, shifts[0])

op['block_size'] = [128, 128]
op['maxregshiftNR'] = 10
regImg1,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)

op['block_size'] = [64, 64]
op['maxregshiftNR'] = 5
regImg2,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)

op['block_size'] = [32, 32]
op['maxregshiftNR'] = 5
regImg3,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)


# 2-step with [128,128] first
op['block_size'] = [128, 128]
op['maxregshiftNR'] = 5
regImg4,_,_ = s2p_nonrigid_registration(regImg1, meanImg1, op)

op['block_size'] = [64, 64]
op['maxregshiftNR'] = 3
regImg5,_,_ = s2p_nonrigid_registration(regImg1, meanImg1, op)

op['block_size'] = [32, 32]
op['maxregshiftNR'] = 3
regImg6,_,_ = s2p_nonrigid_registration(regImg1, meanImg1, op)

# 2-step with [64,64] first
op['block_size'] = [64, 64]
op['maxregshiftNR'] = 3
regImg7,_,_ = s2p_nonrigid_registration(regImg2, meanImg1, op)

op['block_size'] = [32, 32]
op['maxregshiftNR'] = 3
regImg8,_,_ = s2p_nonrigid_registration(regImg2, meanImg1, op)

#%%
viewer = napari.Viewer()
ref = clahe_each(meanImg1)
viewer.add_image(np.array([ref, clahe_each(regImg0[0,:,:])]), name='nonrigid 1-step')
viewer.add_image(np.array([ref, clahe_each(regImg1[0,:,:])]), name='rotate & nonrigid 1-step 128')
viewer.add_image(np.array([ref, clahe_each(regImg2[0,:,:])]), name='rotate & nonrigid 1-step 64')
viewer.add_image(np.array([ref, clahe_each(regImg3[0,:,:])]), name='rotate & nonrigid 1-step 32')
viewer.add_image(np.array([ref, clahe_each(regImg4[0,:,:])]), name='rotate & nonrigid 2-step 128->128')
viewer.add_image(np.array([ref, clahe_each(regImg5[0,:,:])]), name='rotate & nonrigid 2-step 128->64')
viewer.add_image(np.array([ref, clahe_each(regImg6[0,:,:])]), name='rotate & nonrigid 2-step 128->32')
viewer.add_image(np.array([ref, clahe_each(regImg7[0,:,:])]), name='rotate & nonrigid 2-step 64->64')
viewer.add_image(np.array([ref, clahe_each(regImg8[0,:,:])]), name='rotate & nonrigid 2-step 64->32')

'''
2-step does not seem to help that much.
Too small block size hurts.
Hard to compare with naked eyes now. 
Quantify
'''
#%% Quantify the match using (1) correlation and (2) mean of squared difference (SSD)
ystart = 35
corrvals = []
msd = []
results = np.vstack([regImg0, regImg1, regImg2, regImg3, regImg4, regImg5, regImg6, regImg7, regImg8])
for i in range(9):
    cv = np.corrcoef(clahe_each(meanImg1[ystart:,:]).flatten(), clahe_each(results[i,ystart:,:]).flatten())[0][1]
    corrvals.append(cv)
    msdTemp = np.mean((clahe_each(meanImg1[ystart:,:]).flatten() - clahe_each(results[i,ystart:,:]).flatten())**2)
    msd.append(msdTemp)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(corrvals, 'g')
ax1.set_ylabel('Correlation', color='g')
ax1.tick_params(axis='y', colors='g')
ax2.plot(msd, 'b')
ax2.set_ylabel('Mean of squared difference', color='b')
ax2.tick_params(axis='y', colors='b')
ax1.set_xticklabels(['','s2p only', 'rotate & s2p 1-step 128', 'rotate & s2p 1-step 64', 'rotate & s2p 1-step 32',
                     'rotate & s2p 2-step 128 -> 128', 'rotate & s2p 2-step 128 -> 64', 'rotate & s2p 2-step 128 -> 32',
                     'rotate & s2p 2-step 64 -> 64', 'rotate & s2p 2-step 64 -> 32'], rotation=45, ha='right')
ax1.set_title(f'JK{mouse:03} plane{pn} session {sn2}')
fig.tight_layout()



''' 
Rough test in one plane one session says 
rotate -> s2p 128 -> s2p 32 is the best. (not thorough search)
Is this true for all other sessions, all other planes, and all other mice?
'''

#%%
mi = 0
mouse = mice[mi]
pn = 3
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
sn1 = 4

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['snr_thresh'] = 1.2
# op['block_size'] = [128, 128]
# op['maxregshiftNR'] = 15

ops1 = np.load(f'{planeDir}{sn1:03}/plane0/ops.npy', allow_pickle=True).item()
refMeanImg = ops1['meanImg']

Ly = ops1['Ly']
Lx = ops1['Lx']
radius = Ly//2
sns = get_session_names(planeDir, mouse, pn)
warp_fixed = warp_polar(refMeanImg, radius = radius)
corrPlane = []
msdPlane = []

top = 50
bottom = 20
left = 20
right = 20

claheRef = clahe_each(refMeanImg[top:-bottom, left:-right], nbins = 2**16)

for sn2 in sns:
    sn2 = sn2[4:]
    ops2 = np.load(f'{planeDir}{sn2}/plane0/ops.npy', allow_pickle=True).item()
    tempMeanImg = ops2['meanImg']
    
    # Rotation registration
    warp_moving = warp_polar(tempMeanImg, radius = radius)
    shifts, _, _ = phase_cross_correlation(warp_fixed, warp_moving, upsample_factor=100)
    regImgRot = rotate(tempMeanImg, shifts[0])
    
    regImgList = []
    # (0) s2p only 1-step 128
    op['block_size'] = [128,128]
    op['maxregshiftNR'] = 12
    regImg,_,_ = s2p_nonrigid_registration([tempMeanImg], meanImg1, op)
    regImgList.append(regImg)
    # (1) s2p only 2-step 128 -> 32
    op['block_size'] = [32,32]
    op['maxregshiftNR'] = 3
    regImg,_,_ = s2p_nonrigid_registration(regImg, meanImg1, op)
    regImgList.append(regImg)
    # (2) rotate -> s2p 128
    op['block_size'] = [128,128]
    op['maxregshiftNR'] = 12
    regImg,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
    regImgList.append(regImg)
    # (3) rotate -> s2p 64
    op['block_size'] = [64,64]
    op['maxregshiftNR'] = 6
    regImg,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
    regImgList.append(regImg)
    # (4) rotate -> s2p 32
    op['block_size'] = [32,32]
    op['maxregshiftNR'] = 3
    regImg,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
    regImgList.append(regImg)
    # (5) rotate -> s2p 128 -> s2p 128
    op['block_size'] = [128,128]
    op['maxregshiftNR'] = 12
    regImgTemp,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
    op['block_size'] = [128,128]
    op['maxregshiftNR'] = 12
    regImg,_,_ = s2p_nonrigid_registration(regImgTemp, meanImg1, op)
    regImgList.append(regImg)
    # (6) rotate -> s2p 128 -> s2p 64
    op['block_size'] = [128,128]
    op['maxregshiftNR'] = 12
    regImgTemp,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
    op['block_size'] = [64,64]
    op['maxregshiftNR'] = 6
    regImg,_,_ = s2p_nonrigid_registration(regImgTemp, meanImg1, op)
    regImgList.append(regImg)
    # (7) rotate -> s2p 128 -> s2p 32
    op['block_size'] = [128,128]
    op['maxregshiftNR'] = 12
    regImgTemp,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
    op['block_size'] = [32,32]
    op['maxregshiftNR'] = 3
    regImg,_,_ = s2p_nonrigid_registration(regImgTemp, meanImg1, op)
    regImgList.append(regImg)
    # (8) rotate -> s2p 64 -> s2p 64
    op['block_size'] = [64,64]
    op['maxregshiftNR'] = 6
    regImgTemp,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
    op['block_size'] = [64,64]
    op['maxregshiftNR'] = 6
    regImg,_,_ = s2p_nonrigid_registration(regImgTemp, meanImg1, op)
    regImgList.append(regImg)
    # (8) rotate -> s2p 64 -> s2p 32
    op['block_size'] = [64,64]
    op['maxregshiftNR'] = 6
    regImgTemp,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
    op['block_size'] = [32,32]
    op['maxregshiftNR'] = 3
    regImg,_,_ = s2p_nonrigid_registration(regImgTemp, meanImg1, op)
    regImgList.append(regImg)

    # correlation and mean squared distance after 
    # (1) trim edges
    #   For now, arbitrary values
    # (2) histogram adjustment (clahe)
    corrSession = []
    msdSession= []
    for regImg in regImgList:
        claheReg = clahe_each(regImg[0,top:-bottom,left:-right], nbins=2**16)
        corr = np.corrcoef(claheRef.flatten(), claheReg.flatten())[0][1]
        corrSession.append(corr)
        msd = np.mean((claheRef.flatten() - claheReg.flatten()) **2)
        msdSession.append(msd)
    corrPlane.append(corrSession)
    msdPlane.append(msdSession)

corrPlane = np.array(corrPlane)
msdPlane = np.array(msdPlane)

#%% Plot the result
numSession = len(sns)
fig, ax1 = plt.subplots(figsize=(13,7))
# Plot mean +/- sd of correlation values
ax1.plot(corrPlane.mean(axis=0), 'g-', label=None)
ax1.fill_between(range(10), corrPlane.mean(axis=0)-corrPlane.std(axis=0), 
                 corrPlane.mean(axis=0)+corrPlane.std(axis=0), alpha=0.2, facecolor='g')
ax1.set_ylabel('Correlation', color='g')
ax1.tick_params(axis='y', colors='g')
# Plot mean +/- sd of msd values
ax2 = ax1.twinx()
ax2.plot(msdPlane.mean(axis=0), 'b-', label=None)
ax2.fill_between(range(10), msdPlane.mean(axis=0)-msdPlane.std(axis=0), 
                 msdPlane.mean(axis=0)+msdPlane.std(axis=0), alpha=0.2, facecolor='b')
ax2.set_ylabel('Mean of squared difference', color='b')
ax2.tick_params(axis='y', colors='b')
ax1.set_xticks(range(10))
ax1.set_xticklabels(['s2p only 128', 's2p 128 -> 128', 
                     'rotate & s2p 128', 'rotate & s2p 64', 'rotate & s2p 32',
                     'rotate & s2p 128 -> 128', 'rotate & s2p 128 -> 64', 'rotate & s2p 128 -> 32',
                     'rotate & s2p 64 -> 64', 'rotate & s2p 64 -> 32'], rotation=45, ha='right')
ax1.set_title(f'JK{mouse:03} plane{pn}')
fig.tight_layout()
    
#%%
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots(1,2, figsize=(10,10))
imCorr = ax[0].imshow(corrPlane)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(imCorr, cax=cax)

ax[0].set_ylabel('Sessions')
ax[0].set_title('Correlation')
ax[0].set_xticks(range(10))
ax[0].set_xticklabels(['s2p only 128', 's2p 128 -> 128', 
                     'rotate & s2p 128', 'rotate & s2p 64', 'rotate & s2p 32',
                     'rotate & s2p 128 -> 128', 'rotate & s2p 128 -> 64', 'rotate & s2p 128 -> 32',
                     'rotate & s2p 64 -> 64', 'rotate & s2p 64 -> 32'], rotation=45, ha='right')

imMsd = ax[1].imshow(msdPlane)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(imMsd, cax=cax)

ax[1].set_ylabel('Sessions')
ax[1].set_title('MSD')

ax[1].set_xticks(range(10))
ax[1].set_xticklabels(['s2p only 128', 's2p 128 -> 128', 
                     'rotate & s2p 128', 'rotate & s2p 64', 'rotate & s2p 32',
                     'rotate & s2p 128 -> 128', 'rotate & s2p 128 -> 64', 'rotate & s2p 128 -> 32',
                     'rotate & s2p 64 -> 64', 'rotate & s2p 64 -> 32'], rotation=45, ha='right')
fig.suptitle(f'JK{mouse:03} Plane {pn}')
fig.tight_layout()




'''
It makes not-so-great matching (or almost matching) sessions pretty bad.
Rotation makes it bad, so let's check what went wrong and fix it.
2021/09/05
'''


#%% Check rotation
mi = 0
mouse = mice[mi]
pn = 3
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
sn1 = 4

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['snr_thresh'] = 1.2
op['block_size'] = [128, 128]
op['maxregshiftNR'] = op['block_size'][0]//10

ops1 = np.load(f'{planeDir}{sn1:03}/plane0/ops.npy', allow_pickle=True).item()
refMeanImg = ops1['meanImg']
refMeanImgE = ops1['meanImgE']

Ly = ops1['Ly']
Lx = ops1['Lx']
radius = Ly//2
sns = get_session_names(planeDir, mouse, pn)
warp_fixed = warp_polar(refMeanImg, radius = radius)
warp_fixedE = warp_polar(refMeanImgE, radius = radius)
corrPlane = []
msdPlane = []

top = 50
bottom = 20
left = 20
right = 20

claheRef = clahe_each(refMeanImg[top:-bottom, left:-right], nbins = 2**16)

# for sn2 in sns:
#     sn2 = sn2[4:]

#%%
si = 6
sn2 = sns[si][4:]
ops2 = np.load(f'{planeDir}{sn2}/plane0/ops.npy', allow_pickle=True).item()
tempMeanImg = ops2['meanImg']
tempMeanImgE = ops2['meanImgE']
# Rotation registration
warp_moving = warp_polar(tempMeanImg, radius = radius)
shifts, _, _ = phase_cross_correlation(warp_fixed, warp_moving, upsample_factor=100)
regImgRot = rotate(tempMeanImg, shifts[0])

# Translation
ymax, xmax, _, _, _ = phase_corr(refMeanImg, regImgRot)
regImg = rigid.shift_frame(frame=regImgRot, dy=-ymax, dx=-xmax)

napari.view_image(np.array([regImgRot, tempMeanImg, refMeanImg, regImg]))


#%% Using enhanced image
si = 6
sn2 = sns[si][4:]
ops2 = np.load(f'{planeDir}{sn2}/plane0/ops.npy', allow_pickle=True).item()
tempMeanImg = ops2['meanImg']
tempMeanImgE = ops2['meanImgE']


# Rotation registration
warp_movingE = warp_polar(tempMeanImgE, radius = radius)
shifts, _, _ = phase_cross_correlation(warp_fixedE, warp_movingE, upsample_factor=100)
regImgRot = rotate(tempMeanImg, shifts[0])

# Translation
ymax, xmax, _, _, _ = phase_corr(refMeanImg, regImgRot)
regImg = rigid.shift_frame(frame=regImgRot, dy=-ymax, dx=-xmax)

napari.view_image(np.array([regImgRot, tempMeanImg, refMeanImg, regImg]))

'''
Enhanced image does not help at all
'''

#%% Translate first, and then rotate
# Translation
ymax, xmax, _, _, _ = phase_corr(refMeanImg, tempMeanImg, transLim = 90)
transImg = rigid.shift_frame(frame=tempMeanImg, dy=-ymax, dx=-xmax)
# Rotate
warp_moving = warp_polar(transImg, radius = radius)
shifts, _, _ = phase_cross_correlation(warp_fixed, warp_moving, upsample_factor=100)
regImgRot = rotate(transImg, shifts[0])
# Translate again
ymax, xmax, _, _, _ = phase_corr(refMeanImg, regImgRot)
regImg = rigid.shift_frame(frame=regImgRot, dy=-ymax, dx=-xmax)


# napari.view_image(np.array([regImgRot, tempMeanImg, refMeanImg, regImg]))
napari.view_image(np.array([regImgRot, tempMeanImg, refMeanImg, regImg]))





#%% Compare with s2p 2-step
op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['snr_thresh'] = 1.2
# op['block_size'] = [128, 128]
# op['maxregshiftNR'] = op['block_size'][0]//10

ops1 = np.load(f'{planeDir}{sn1:03}/plane0/ops.npy', allow_pickle=True).item()
refMeanImg = ops1['meanImg']
refMeanImgE = ops1['meanImgE']

Ly = ops1['Ly']
Lx = ops1['Lx']
radius = Ly//2
sns = get_session_names(planeDir, mouse, pn)
warp_fixed = warp_polar(refMeanImg, radius = radius)
warp_fixedE = warp_polar(refMeanImgE, radius = radius)
corrPlane = []
msdPlane = []

top = 50
bottom = 20
left = 20
right = 20

claheRef = clahe_each(refMeanImg[top:-bottom, left:-right], nbins = 2**16)
#%%
si = 12
sn2 = sns[si][4:]
ops2 = np.load(f'{planeDir}{sn2}/plane0/ops.npy', allow_pickle=True).item()
tempMeanImg = ops2['meanImg']
tempMeanImgE = ops2['meanImgE']

# s2p 2-step
op['block_size'] = [128,128]
op['maxregshiftNR'] = op['block_size'][0]//10
regImg,_,_ = s2p_nonrigid_registration([tempMeanImg], meanImg1, op)
regImgList.append(regImg)
op['block_size'] = [32,32]
op['maxregshiftNR'] = op['block_size'][0]//10
s2p2stepRegImg,_,_ = s2p_nonrigid_registration(regImg, meanImg1, op)

# Rotate first and then s2p 2-step
# Translate
ymax, xmax, _, _, _ = phase_corr(refMeanImg, tempMeanImg, transLim = 50)
transImg = rigid.shift_frame(frame=tempMeanImg, dy=-ymax, dx=-xmax)
# Rotate
warp_moving = warp_polar(transImg, radius = radius)
shifts, _, _ = phase_cross_correlation(warp_fixed, warp_moving, upsample_factor=100)
regImgRot = rotate(transImg, shifts[0])
# s2p 2-step
op['block_size'] = [128,128]
op['maxregshiftNR'] = 5
regImg,_,_ = s2p_nonrigid_registration([regImgRot], meanImg1, op)
op['block_size'] = [32,32]
op['maxregshiftNR'] = 3
rotateFirstRegImg,_,_ = s2p_nonrigid_registration(regImg, meanImg1, op)

napari.view_image(np.array([s2p2stepRegImg[0,:,:], refMeanImg, rotateFirstRegImg[0,:,:], s2p2stepRegImg[0,:,:]]))




'''
this seems to be working.
the problem was that the moving images were translated too much, 
and other blood vessels matched better in these cases.
'''


#%% Test translate first approach (which s2p nonrigid actually does)
# First, test with s2p 2-step and trt s2p 2-step
mi = 0
mouse = mice[mi]
pn = 3
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
sn1 = 4

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['snr_thresh'] = 1.2
# op['block_size'] = [128, 128]
# op['maxregshiftNR'] = 15

ops1 = np.load(f'{planeDir}{sn1:03}/plane0/ops.npy', allow_pickle=True).item()
refMeanImg = ops1['meanImg']

Ly = ops1['Ly']
Lx = ops1['Lx']
radius = Ly//2
sns = get_session_names(planeDir, mouse, pn)
warp_fixed = warp_polar(refMeanImg, radius = radius)
rotAngles = []
corrPlane = []
msdPlane = []
s2pRegPlane = []
rotRegPlane = []
top = 50
bottom = 50
left = 50
right = 50

claheRef = clahe_each(refMeanImg[top:-bottom, left:-right], nbins = 2**16)

for sn2 in sns:
    sn2 = sn2[4:]
    ops2 = np.load(f'{planeDir}{sn2}/plane0/ops.npy', allow_pickle=True).item()
    tempMeanImg = ops2['meanImg']
    
    # Rotation registration 
    # (with translation first)
    # ymax, xmax, _, _, _ = phase_corr(refMeanImg[top:-bottom,left:-right], tempMeanImg[top:-bottom,left:-right], transLim = 30)
    # transMeanImg = rigid.shift_frame(frame=tempMeanImg, dy=-ymax, dx=-xmax)
    # warp_moving = warp_polar(transMeanImg, radius = radius)
    warp_moving = warp_polar(tempMeanImg, radius = radius)
    shifts, _, _ = phase_cross_correlation(warp_fixed, warp_moving, upsample_factor=100)
    regImgRot = rotate(tempMeanImg, shifts[0])
    rotAngles.append(shifts[0])
    
    regImgList = []
    
    # (0) s2p 128 -> 32
    op['block_size'] = [128,128]
    op['maxregshiftNR'] = 12
    regImg1st,_,_ = s2p_nonrigid_registration([tempMeanImg], refMeanImg, op)
    op['block_size'] = [64,64]
    op['maxregshiftNR'] = 6
    regImg,_,_ = s2p_nonrigid_registration(regImg1st, refMeanImg, op)
    regImgList.append(regImg)
    s2pRegPlane.append(regImg)
    
    # (1) rotate -> s2p 128
    op['block_size'] = [128,128]
    op['maxregshiftNR'] = 12
    regImg1st,_,_ = s2p_nonrigid_registration([regImgRot], refMeanImg, op)
    op['block_size'] = [64,64]
    op['maxregshiftNR'] = 6
    regImg,_,_ = s2p_nonrigid_registration(regImg1st, refMeanImg, op)    
    regImgList.append(regImg)
    rotRegPlane.append(regImg)

    # correlation and mean squared distance after 
    # (1) trim edges
    #   For now, arbitrary values
    # (2) histogram adjustment (clahe)
    corrSession = []
    msdSession= []
    for regImg in regImgList:
        claheReg = clahe_each(regImg[0,top:-bottom,left:-right], nbins=2**16)
        corr = np.corrcoef(claheRef.flatten(), claheReg.flatten())[0][1]
        corrSession.append(corr)
        msd = np.mean((claheRef.flatten() - claheReg.flatten()) **2)
        msdSession.append(msd)
    corrPlane.append(corrSession)
    msdPlane.append(msdSession)

corrPlane = np.array(corrPlane)
msdPlane = np.array(msdPlane)

#%% Plot the test result
fig, ax = plt.subplots(2,1, figsize=(13,7))
ax[0].plot(corrPlane[:,0], 'c', label='s2p')
ax[0].plot(corrPlane[:,1], 'm', label='Rotate first')
ax[0].legend()
ax[0].set_ylabel('Correlation')
ax[0].set_xlabel('Session index')
ax[1].plot(msdPlane[:,0], 'c', label='s2p')
ax[1].plot(msdPlane[:,1], 'm', label='Rotate first')
ax[1].set_ylabel('MSD')
ax[1].set_xlabel('Session index')
fig.suptitle(f'JK{mouse:03} plane {pn}')
fig.tight_layout()


'''
s2p 2-step worked better.... -_-
'''

#%% Visual inspection
viewer = napari.Viewer()
for i in range(len(s2pRegPlane)):
    viewer.add_image(np.array([s2pRegPlane[i][0,:,:], refMeanImg, rotRegPlane[i][0,:,:], s2pRegPlane[i][0,:,:]]), 
                     name=sns[i][4:], visible = False)
    
    
    
#%% One last test with suite2p nonrigid registration parameters
# Combination of block sizes and maxregshiftNR
# Block sizes: 128->128 / 128->64 / 128->32 / 64->64 / 64->32 / 32->32
# maxregshiftNR: 10% floor / 10->10 / 10->5 / 10->3 / 5->5 / 5->3 / 3->3
bsList = [[128,128], [128,64], [128,32], [64,64], [64,32], [32,32]]
mrsList = [[10,10],[10,5],[10,3],[5,5],[5,3],[3,3]]

siTest = [1,4,9,12,16]
# siTest = [4,12]
s2pRegPlane = []
rotRegPlane = []



mi = 0
mouse = mice[mi]
pn = 3
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
snRef = 4

op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['snr_thresh'] = 1.2
# op['block_size'] = [128, 128]
# op['maxregshiftNR'] = 15

opsRef = np.load(f'{planeDir}{snRef:03}/plane0/ops.npy', allow_pickle=True).item()
refMeanImg = opsRef['meanImg']

Ly = opsRef['Ly']
Lx = opsRef['Lx']
radius = Ly//2
sns = get_session_names(planeDir, mouse, pn)
warp_fixed = warp_polar(refMeanImg, radius = radius)
rotAngles = []
rotAnglesTr = []
corrPlane = []
msdPlane = []
s2pRegPlane = []
rotRegPlane = []
trRotRegPlane = []
top = 50
bottom = 50
left = 50
right = 50

claheRef = clahe_each(refMeanImg[top:-bottom, left:-right], nbins = 2**16)

for si in siTest:
    snTest = sns[si][4:]
    print(f'Testing with session {snTest}')
    opsTest = np.load(f'{planeDir}{snTest}/plane0/ops.npy', allow_pickle=True).item()
    tempMeanImg = opsTest['meanImg']
    
    warp_moving = warp_polar(tempMeanImg, radius = radius)
    shifts, _, _ = phase_cross_correlation(warp_fixed, warp_moving, upsample_factor=100)
    regImgRot = rotate(tempMeanImg, shifts[0])
    rotAngles.append(shifts[0])
    
    ymax, xmax, _, _, _ = phase_corr(refMeanImg[top:-bottom,left:-right], tempMeanImg[top:-bottom,left:-right], transLim=50)
    trImg = rigid.shift_frame(frame=tempMeanImg, dy=-ymax, dx=-xmax)
    warp_moving = warp_polar(trImg, radius = radius)
    shifts, _, _ = phase_cross_correlation(warp_fixed, warp_moving, upsample_factor=100)
    regImgTrRot = rotate(trImg, shifts[0])
    rotAnglesTr.append(shifts[0])
    
    s2pRegSession = []
    rotRegSession = []
    trRotRegSession = []
    for bs in bsList:
        bsTest1 = bs[0]
        bsTest2 = bs[1]
        print(f'Block sizes [{bsTest1}, {bsTest1}] -> [{bsTest2}, {bsTest2}]')
        op['block_size'] = [bs[0],bs[0]]
        for mrsi in range(len(mrsList)+1):
            if mrsi == 0:
                # 1st nonrigid
                op['maxregshiftNR'] = bs[0]//10
                s2pTemp1,_,_ = s2p_nonrigid_registration([tempMeanImg], refMeanImg, op)
                # Rotated image
                rotTemp1,_,_ = s2p_nonrigid_registration([regImgRot], refMeanImg, op)
                # TransRotated imge
                trRotTemp1,_,_ = s2p_nonrigid_registration([regImgTrRot], refMeanImg, op)
                
                # 2nd nonrigid
                op['block_size'] = [bs[1],bs[1]]
                op['maxregshiftNR'] = bs[1]//10
                s2pReg,_,_ = s2p_nonrigid_registration(s2pTemp1, refMeanImg, op)
                s2pRegSession.append(s2pReg)
                # Rotated image
                rotReg,_,_ = s2p_nonrigid_registration(rotTemp1, refMeanImg, op)
                rotRegSession.append(rotReg)
                # TransRotated imge
                trRotReg,_,_ = s2p_nonrigid_registration(trRotTemp1, refMeanImg, op)
                trRotRegSession.append(trRotReg)
            else:
                # 1st nonrigid
                op['maxregshiftNR'] = mrsList[mrsi-1][0]
                s2pTemp1,_,_ = s2p_nonrigid_registration([tempMeanImg], refMeanImg, op)
                # Rotated image
                rotTemp1,_,_ = s2p_nonrigid_registration([regImgRot], refMeanImg, op)
                # TransRotated imge
                trRotTemp1,_,_ = s2p_nonrigid_registration([regImgTrRot], refMeanImg, op)
                
                # 2nd nonrigid
                op['block_size'] = [bs[1],bs[1]]
                op['maxregshiftNR'] = mrsList[mrsi-1][1]
                s2pReg,_,_ = s2p_nonrigid_registration(s2pTemp1, refMeanImg, op)
                s2pRegSession.append(s2pReg)
                # Rotated image
                rotReg,_,_ = s2p_nonrigid_registration(rotTemp1, refMeanImg, op)
                rotRegSession.append(rotReg)
                # TransRotated imge
                trRotReg,_,_ = s2p_nonrigid_registration(trRotTemp1, refMeanImg, op)
                trRotRegSession.append(trRotReg)
    
    s2pRegPlane.append(s2pRegSession)
    rotRegPlane.append(rotRegSession)
    trRotRegPlane.append(trRotRegSession)

#%% Show quantification
top = 50
bottom = 50
left = 50
right = 50
fig, ax = plt.subplots(2,len(siTest), figsize=(13,7))
for i in range(len(siTest)):
    corrS2p = []
    msdS2p = []
    for s2pImg in s2pRegPlane[i]:
        claheTest = clahe_each(s2pImg[0,top:-bottom, left:-right], nbins = 2**16)
        corrS2p.append(np.corrcoef(claheRef.flatten(), claheTest.flatten())[0][1])
        msdS2p.append(np.mean((claheRef.flatten()-claheTest.flatten())**2))
        
    corrRot = []
    msdRot = []
    for rotImg in rotRegPlane[i]:
        claheTest = clahe_each(rotImg[0,top:-bottom, left:-right], nbins = 2**16)
        corrRot.append(np.corrcoef(claheRef.flatten(), claheTest.flatten())[0][1])
        msdRot.append(np.mean((claheRef.flatten()-claheTest.flatten())**2))
        
    corrTr = []
    msdTr = []
    for trImg in trRotRegPlane[i]:
        claheTest = clahe_each(trImg[0,top:-bottom, left:-right], nbins = 2**16)
        corrTr.append(np.corrcoef(claheRef.flatten(), claheTest.flatten())[0][1])
        msdTr.append(np.mean((claheRef.flatten()-claheTest.flatten())**2))
        
    ax[0,i].plot(corrS2p, 'c', label='s2p')
    ax[0,i].plot(corrRot, 'm', label='rotate')
    ax[0,i].plot(corrTr, 'g', label='translate')
    ax[0,i].legend()
    ax[0,i].set_ylabel('Correlation')
    si = siTest[i]    
    ax[0,i].set_title(f'Session index {si}')
    
    ax[1,i].plot(msdS2p, 'c', label='s2p')
    ax[1,i].plot(msdRot, 'm', label='rotate')
    ax[1,i].plot(msdTr, 'g', label='translate')
    ax[1,i].set_ylabel('MSD')
    ax[1,i].set_xlabel('Combination index')
    
fig.suptitle(f'JK{mouse:03} plane {pn}')    
fig.tight_layout()
    
'''
128 -> 32 with 10% floor works the best in all cases.
Now, is this true in other planes and other mice?
'''


#%% Test in other planes and mice
# with some reduced combination of parameters
bsList = [[128,128], [128,64], [128,32], [64,64]]
mrsList = [[10,10],[10,3],[5,5]]

top = 50
bottom = 50
left = 50
# right = 50
op = {}
op['smooth_sigma'] = 1.15 # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
op['maxregshift'] = 0.3
op['smooth_sigma_time'] = 0
op['snr_thresh'] = 1.2

miList = [0,3,8]
pnList = [2,5,8]
siList = [2,7,10,13,18]
s2pRegAll = []
rotRegAll = []
trRegAll = []

rotAnglesAll = []
rotAnglesTrAll = []

for mi in miList:
    mouse = mice[mi]
    snRef = refSessions[mi]
    
    # Right edge different for mouse 052
    right = 50 if mi < 8 else 120
        
    s2pRegMouse = []
    rotRegMouse = []
    trRegMouse = []
    
    rotAnglesMouse = []
    rotAnglesTrMouse = []
    
    for pn in pnList:
        print(f'JK{mouse:03} plane {pn}')
        planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
        opsRef = np.load(f'{planeDir}{snRef:03}/plane0/ops.npy', allow_pickle=True).item()
        refMeanImg = opsRef['meanImg']
        
        Ly = opsRef['Ly']
        radius = Ly//2
        sns = get_session_names(planeDir, mouse, pn)
        warp_fixed = warp_polar(refMeanImg, radius = radius)
        
        corrPlane = []
        msdPlane = []
        
        s2pRegPlane = []
        rotRegPlane = []
        trRotRegPlane = []
        
        rotAngles = []
        rotAnglesTr = []
        
        claheRef = clahe_each(refMeanImg[top:-bottom, left:-right], nbins = 2**16)
        
        for si in siList:
            snTest = sns[si][4:]
            print(f'Testing with session {snTest}')
            opsTest = np.load(f'{planeDir}{snTest}/plane0/ops.npy', allow_pickle=True).item()
            tempMeanImg = opsTest['meanImg']
            
            warp_moving = warp_polar(tempMeanImg, radius = radius)
            shifts, _, _ = phase_cross_correlation(warp_fixed, warp_moving, upsample_factor=100)
            regImgRot = rotate(tempMeanImg, shifts[0])
            rotAngles.append(shifts[0])
            
            ymax, xmax, _, _, _ = phase_corr(refMeanImg[top:-bottom,left:-right], tempMeanImg[top:-bottom,left:-right], transLim=50)
            trImg = rigid.shift_frame(frame=tempMeanImg, dy=-ymax, dx=-xmax)
            warp_moving = warp_polar(trImg, radius = radius)
            shifts, _, _ = phase_cross_correlation(warp_fixed, warp_moving, upsample_factor=100)
            regImgTrRot = rotate(trImg, shifts[0])
            rotAnglesTr.append(shifts[0])
            
            s2pRegSession = []
            rotRegSession = []
            trRotRegSession = []
            for bs in bsList:
                bsTest1 = bs[0]
                bsTest2 = bs[1]
                print(f'Block sizes [{bsTest1}, {bsTest1}] -> [{bsTest2}, {bsTest2}]')
                op['block_size'] = [bs[0],bs[0]]
                for mrsi in range(len(mrsList)+1):
                    if mrsi == 0:
                        # 1st nonrigid
                        op['maxregshiftNR'] = bs[0]//10
                        s2pTemp1,_,_ = s2p_nonrigid_registration([tempMeanImg], refMeanImg, op)
                        # Rotated image
                        rotTemp1,_,_ = s2p_nonrigid_registration([regImgRot], refMeanImg, op)
                        # TransRotated imge
                        trRotTemp1,_,_ = s2p_nonrigid_registration([regImgTrRot], refMeanImg, op)
                        
                        # 2nd nonrigid
                        op['block_size'] = [bs[1],bs[1]]
                        op['maxregshiftNR'] = bs[1]//10
                        s2pReg,_,_ = s2p_nonrigid_registration(s2pTemp1, refMeanImg, op)
                        s2pRegSession.append(s2pReg)
                        # Rotated image
                        rotReg,_,_ = s2p_nonrigid_registration(rotTemp1, refMeanImg, op)
                        rotRegSession.append(rotReg)
                        # TransRotated imge
                        trRotReg,_,_ = s2p_nonrigid_registration(trRotTemp1, refMeanImg, op)
                        trRotRegSession.append(trRotReg)
                    else:
                        # 1st nonrigid
                        op['maxregshiftNR'] = mrsList[mrsi-1][0]
                        s2pTemp1,_,_ = s2p_nonrigid_registration([tempMeanImg], refMeanImg, op)
                        # Rotated image
                        rotTemp1,_,_ = s2p_nonrigid_registration([regImgRot], refMeanImg, op)
                        # TransRotated imge
                        trRotTemp1,_,_ = s2p_nonrigid_registration([regImgTrRot], refMeanImg, op)
                        
                        # 2nd nonrigid
                        op['block_size'] = [bs[1],bs[1]]
                        op['maxregshiftNR'] = mrsList[mrsi-1][1]
                        s2pReg,_,_ = s2p_nonrigid_registration(s2pTemp1, refMeanImg, op)
                        s2pRegSession.append(s2pReg)
                        # Rotated image
                        rotReg,_,_ = s2p_nonrigid_registration(rotTemp1, refMeanImg, op)
                        rotRegSession.append(rotReg)
                        # TransRotated imge
                        trRotReg,_,_ = s2p_nonrigid_registration(trRotTemp1, refMeanImg, op)
                        trRotRegSession.append(trRotReg)
            
            s2pRegPlane.append(s2pRegSession)
            rotRegPlane.append(rotRegSession)
            trRotRegPlane.append(trRotRegSession)
            
        # Show plots
        fig, ax = plt.subplots(2,len(siList), figsize=(13,7))
        for i in range(len(siList)):
            corrS2p = []
            msdS2p = []
            for s2pImg in s2pRegPlane[i]:
                claheTest = clahe_each(s2pImg[0,top:-bottom, left:-right], nbins = 2**16)
                corrS2p.append(np.corrcoef(claheRef.flatten(), claheTest.flatten())[0][1])
                msdS2p.append(np.mean((claheRef.flatten()-claheTest.flatten())**2))
                
            corrRot = []
            msdRot = []
            for rotImg in rotRegPlane[i]:
                claheTest = clahe_each(rotImg[0,top:-bottom, left:-right], nbins = 2**16)
                corrRot.append(np.corrcoef(claheRef.flatten(), claheTest.flatten())[0][1])
                msdRot.append(np.mean((claheRef.flatten()-claheTest.flatten())**2))
                
            corrTr = []
            msdTr = []
            for trImg in trRotRegPlane[i]:
                claheTest = clahe_each(trImg[0,top:-bottom, left:-right], nbins = 2**16)
                corrTr.append(np.corrcoef(claheRef.flatten(), claheTest.flatten())[0][1])
                msdTr.append(np.mean((claheRef.flatten()-claheTest.flatten())**2))
                
            ax[0,i].plot(corrS2p, 'c', label='s2p')
            ax[0,i].plot(corrRot, 'm', label='rotate')
            ax[0,i].plot(corrTr, 'g', label='translate')
            ax[0,i].legend()
            ax[0,i].set_ylabel('Correlation')
            si = siList[i]    
            ax[0,i].set_title(f'Session index {si}')
            
            ax[1,i].plot(msdS2p, 'c', label='s2p')
            ax[1,i].plot(msdRot, 'm', label='rotate')
            ax[1,i].plot(msdTr, 'g', label='translate')
            ax[1,i].set_ylabel('MSD')
            ax[1,i].set_xlabel('Combination index')
        fig.suptitle(f'JK{mouse:03} Plane {pn}')
        fig.tight_layout()
        
        # Gather mouse data
        s2pRegMouse.append(s2pRegPlane)
        rotRegMouse.append(rotRegPlane)
        trRegMouse.append(trRotRegPlane)
        
        rotAnglesMouse.append(rotAngles)
        rotAnglesTrMouse.append(rotAnglesTr)
    # Gather all data
    s2pRegAll.append(s2pRegMouse)
    rotRegAll.append(rotRegMouse)
    trRegAll.append(trRegMouse)
    
    rotAnglesAll.append(rotAnglesMouse)
    rotAnglesTrAll.append(rotAnglesTrMouse)


'''
In 9 planes (from 3 mice) tested,
either 128->64 or 128->32 with floor 10% (//10) max shift works the best.
Sometimes when translate->rotate works better, the correlation value difference is negligible. 

'''
#%% Visual inspection
# Look for those that rotation improved matching

miList = [0,3,8]
pnList = [2,5,8]
siList = [2,7,10,13,18]

# (mi, pn, si) = (8, 8, 7)
# (mi, pn, si) = (8, 5, 7)
# (mi, pn, si) = (8, 2, 2)
# (mi, pn, si) = (3, 8, 2)
# (mi, pn, si) = (0, 8, 18)
# (mi, pn, si) = (0, 8, 2)
# (mi, pn, si) = (0, 5, 10)
(mi, pn, si) = (0, 2, 18)
imi = np.where(np.array(miList)==mi)[0][0]
ipn = np.where(np.array(pnList)==pn)[0][0]
isi = np.where(np.array(siList)==si)[0][0]

mouse = mice[mi]
snRef = refSessions[mi]
    
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
opsRef = np.load(f'{planeDir}{snRef:03}/plane0/ops.npy', allow_pickle=True).item()
refMeanImg = opsRef['meanImg']

besti = 8

# rotImg = rotRegAll[imi][ipn][isi][besti][0,:,:]
rotImg = trRegAll[imi][ipn][isi][besti][0,:,:]
s2pImg = s2pRegAll[imi][ipn][isi][besti][0,:,:]
napari.view_image(np.array([rotImg, refMeanImg, s2pImg, rotImg]))    


#%% And Compare between 128->64 vs 128->32 (is there really any difference?)
# 64 > 32
# (mi, pn, si) = (0, 8, 2)
# (mi, pn, si) = (0, 8, 7)
# (mi, pn, si) = (3, 8, 2)
# (mi, pn, si) = (8, 8, 2)

# 64 < 32
# (mi, pn, si) = (3, 5, 7)
# (mi, pn, si) = (3, 5, 18)
# (mi, pn, si) = (8, 2, 7)
imi = np.where(np.array(miList)==mi)[0][0]
ipn = np.where(np.array(pnList)==pn)[0][0]
isi = np.where(np.array(siList)==si)[0][0]

mouse = mice[mi]
snRef = refSessions[mi]
    
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
opsRef = np.load(f'{planeDir}{snRef:03}/plane0/ops.npy', allow_pickle=True).item()
refMeanImg = opsRef['meanImg']

besti1 = 4
besti2 = 8
s2pImg1 = s2pRegAll[imi][ipn][isi][besti1][0,:,:]
s2pImg2 = s2pRegAll[imi][ipn][isi][besti2][0,:,:]
napari.view_image(np.array([s2pImg1, refMeanImg, s2pImg2, s2pImg1]))    



'''
All these comparisons were indistinguishable.
Paramters fixed. 2-step suite2p nonrigid registration with 128/12->32/3
'''























