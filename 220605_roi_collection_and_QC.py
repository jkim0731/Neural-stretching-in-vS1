'''
ROI collection and QC

Collect ROIs based on plane-specific optimal session-to-session registration methods
From the last slide of “220303 session-to-session registration”.
Extract signals from each ROIs from the “master ROI map”.
For each ROI map, log if the ROI was present at each session.
For QC
Log each session's depth span.
For flexible depth difference threshold.
QC
(1) Compare between the signal from the master ROI and the one from that session. 
To check if using unified master ROI is OK.
(2) Compare between the signal from the master ROI that was NOT in the session and other ROIs present in that session.
To check if assigning an ROI to sessions without that ROI is OK. But how?
SNR distribution compared to the session-curated neurons.
Signal comparison between adjacent sessions (without learning).

======
Copy registration from 220302_another_registration_test.py
Copy ROI collection from 220117_roi_collection.py
Use z-drift data from 220117_roi_collection.py
Save the registration parameters and results as JK{mouse:03}plane{pn}_session_to_session_registration.npy
Save the master map as JK{mouse:03}plane{n}_roi_collection.npy

======

2022/06/05 JK
'''

#%% BS
import pandas
import numpy as np
import matplotlib.pyplot as plt
import napari
import os, glob, shutil
from pystackreg import StackReg
from skimage import exposure
from suite2p.registration import rigid, nonrigid
from tqdm import tqdm

import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'
# h5Dir = 'D:/'

# mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
# refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
# expSessions =   [19,    10,   21,   17,     0,      0,      23,     0,      21,     0,      0,      0]
# zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
# freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]

mice =          [25,    27,   30,   36,     39,     52]
refSessions =   [4,     3,    3,    1,      1,      3]
expSessions =   [19,    10,   21,   17,     23,     21]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    7.7]

optimal_reg_methods = {'025': ['bilinear', 'bilinear', 'suite2p', 'bilinear', 'old', 'old', 'old', 'old'],
                        '027': ['bilinear', 'bilinear', 'bilinear', 'bilinear', 'suite2p', 'suite2p', 'affine', 'bilinear'],
                        '030': ['old', 'old', 'old', 'bilinear', 'affine', 'bilinear', 'bilinear', 'bilinear'],
                        '036': ['old', 'old', 'old', 'suite2p', 'suite2p', 'bilinear', 'old', 'bilinear'],
                        '039': ['suite2p', 'suite2p', 'suite2p', 'suite2p', 'affine', 'affine', 'affine', 'affine'],
                        '052': ['bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'affine']}
relative_depths_selected = {'025': [[7,17], [18,28]],
                            '027': [[20,30], [25,35]],
                            '030': [[17,27], [22,32]],
                            '036': [[16,26], [12,22]],
                            '039': [[22,32], [17,27]],
                            '052': [[27,37], [17,27]]} # per imaged volume
manual_removal_Session_i = {'025': [[7,14], []],
                            '027': [range(7,12), []],
                            '030': [[], [19]],
                            '036': [[17], [19]],
                            '039': [[], []],
                            '052': [[], []]} # per imaged volume
prevN = 3
op = {'smooth_sigma': 1.15, 'maxregshift': 0.3, 'smooth_sigma_time': 0, 'snr_thresh': 1.2, 'block_size_list': [128,32]}
roi_overlap_threshold = 0.5
# When two ROIs have an overlap, if this overlap/area is larger than the threshold
# for EITHER of the ROIs, then these two ROIs are defined as matching
# This is the same approach as in suite2p, and stricter than that of CaImAn (where intersection/union is used instead)

#%% Helper functions

# Functions for registration
def clahe_each(img: np.float64, kernel_size = None, clip_limit = 0.01, nbins = 2**16):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    newimg = exposure.equalize_adapthist(newimg, kernel_size = kernel_size, clip_limit = clip_limit, nbins=nbins)    
    return newimg

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

# Functions for ROI collection
def calculate_regCell_threshold(cellMap, numPix, thresholdResolution = 0.01):
    trPrecision = len(str(thresholdResolution).split('.')[1])
    thresholdRange = np.around(np.arange(0.3,1+thresholdResolution/10,thresholdResolution), trPrecision)
    threshold = thresholdRange[np.argmin([np.abs(numPix - np.sum(cellMap>=threshold)) for threshold in thresholdRange])]
    cutMap = (cellMap >= threshold).astype(bool)
    return cutMap, threshold

def perimeter_area_ratio(img: bool):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    if len(img.shape) != 3:
        raise('Input image dimension should be either 2 or 3.')
    img = img.astype('bool')
    numCell = img.shape[0]
    par = np.zeros(numCell)
    for ci in range(numCell):
        tempImg = img[ci,:,:]
        inside = tempImg*np.roll(tempImg,1,axis=0)*np.roll(tempImg,-1,axis=0)*np.roll(tempImg,1,axis=1)*np.roll(tempImg,-1,axis=1)
        perimeter = np.logical_xor(tempImg, inside)
        par[ci] = np.sum(perimeter)/np.sum(tempImg) # tempImg instead of inside just to prevent dividing by 0 for some scattered rois
    return par

def check_multi_match_pair(multiMatchMasterInd, multiMatchNewInd, masterPar, newPar, 
                           overlapMatrix, overlaps, delFromMasterInd, delFromNewInd):
    tempDfMasterInd = np.zeros(0,'int')
    tempDfNewInd = np.zeros(0,'int')
    remFromMMmaster = np.zeros(0,'int') # Collect multiMatchMasterInd that is already processed (in the for loop)
    remFromMMnew = np.zeros(0,'int') # Collect multiMatchNewInd that is already processed (in the for loop)
    # Remove remFromMM* at the end.
    
    # First, deal with delFromMasterInd
    if len(multiMatchMasterInd)>0:
        for mci in range(len(multiMatchMasterInd)):
            masterCi = multiMatchMasterInd[mci]
            if masterCi in remFromMMmaster: # For the updates from the loop.
                continue
            else:
                # cell index from the new map, that matches with this master ROI.
                newCis = np.where(overlapMatrix[masterCi,:])[0] # This should be longer than len == 1, by definition of multiMatchMasterInd.
                
                # Calculate any other master ROIs that overlap with this new ROI(s).
                masterBestMatchi = np.zeros(len(newCis), 'int')
                for i, nci in enumerate(newCis):
                    masterBestMatchi[i] = np.argmax(overlaps[:,nci]).astype(int)
                # Check if the best-matched master ROIs for these multiple new ROIs are this master ROI.
                # (I.e., when a large combined master ROI covers more than one new ROIs. 
                # See 220117 ROI collection across sessions with matching depth.pptx)
                # In this case, just remove this master ROI
                if any([len(np.where(masterBestMatchi==mi)[0])>1 for mi in masterBestMatchi]):
                    tempDfMasterInd = np.hstack((tempDfMasterInd, [masterCi]))
                    remFromMMmaster = np.hstack((remFromMMmaster, [masterCi]))
                
                # Else, check if there is a matching pair (or multiples)
                # I.e., multiple overlapping new ROIs with multiple master ROIs (usually couples in both ROI maps)
                # In this case, remove the pair (or multiples) with higher mean PAR 
                else:
                    newBestMatchi = np.zeros(len(masterBestMatchi), 'int')
                    for i, mci in enumerate(masterBestMatchi):
                        newBestMatchi[i] = np.argmax(overlaps[mci,:]).astype(int)

                    if all(newCis == newBestMatchi): # found a matching pair (or a multiple)
                        # Calculate mean perimeter/area ratio
                        masterMeanpar = np.mean(masterPar[masterBestMatchi])
                        newMeanpar = np.mean(newPar[newBestMatchi])
                        
                        # Remove the pair with higher mean par
                        if masterMeanpar <= newMeanpar:
                            tempDfNewInd = np.hstack((tempDfNewInd, newBestMatchi))
                        else:
                            tempDfMasterInd = np.hstack((tempDfMasterInd, masterBestMatchi))
    
                        # Collect indices already processed
                        remFromMMmaster = np.hstack((remFromMMmaster, masterBestMatchi))
                        remFromMMnew = np.hstack((remFromMMnew, newBestMatchi))
                              
    # Then, deal with delFromNewInd
    if len(multiMatchNewInd)>0:
        for nci in range(len(multiMatchNewInd)):
            newCi = multiMatchNewInd[nci]
            if newCi in remFromMMnew:
                continue
            else:
                masterCis = np.where(overlapMatrix[:,newCi])[0]
                    
                newBestMatchi = np.zeros(len(masterCis), 'int')
                for i, mci in enumerate(masterCis):
                    newBestMatchi[i] = np.argmax(overlaps[mci,:]).astype(int)
                    
                # Check if there are multiple same matched IDs 
                # In this case, just remove the new ROI
                if any([len(np.where(newBestMatchi==ni)[0])>1 for ni in newBestMatchi]):
                    tempDfNewInd = np.hstack((tempDfNewInd, [newCi]))
                    remFromMMnew = np.hstack((remFromMMnew, [newCi]))
                
                # Else, check if there is a matching pair (or multiples)
                else:
                    masterBestMatchi = np.zeros(len(newBestMatchi), 'int')
                    for i, nci in enumerate(newBestMatchi):
                        masterBestMatchi[i] = np.argmax(overlaps[:,nci]).astype(int)

                    if all(masterCis == masterBestMatchi): # found a matching pair
                        # Calculate mean perimeter/area ratio
                        masterMeanpar = np.mean(masterPar[masterBestMatchi])
                        newMeanpar = np.mean(newPar[newBestMatchi])
                        
                        # Remove the pair with higher mean par
                        if masterMeanpar <= newMeanpar:
                            tempDfNewInd = np.hstack((tempDfNewInd, newBestMatchi))
                        else:
                            tempDfMasterInd = np.hstack((tempDfMasterInd, masterBestMatchi))
    
                        # Collect indices already processed
                        remFromMMmaster = np.hstack((remFromMMmaster, masterBestMatchi))
                        remFromMMnew = np.hstack((remFromMMnew, newBestMatchi))
    
    # Remove collected indices
    if len(tempDfMasterInd)>0:
        delFromMasterInd.extend(tempDfMasterInd)
    if len(tempDfNewInd)>0:
        delFromNewInd.extend(tempDfNewInd)

    # Ideally, these multi indices should be empty.    
    multiMatchMasterInd = np.array([mi for mi in multiMatchMasterInd if mi not in remFromMMmaster])
    multiMatchNewInd = np.array([ni for ni in multiMatchNewInd if ni not in remFromMMnew])
    
    return delFromMasterInd, delFromNewInd, multiMatchMasterInd, multiMatchNewInd

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

def img_norm(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

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

# %% Run registration in each plane


#%% Test with one plane
# Regstration test
for mi in range(6):
    for pn in range(1,9):

        mouse=mice[mi]
        mouse_str = f'{mouse:03}'
        reg_meth = optimal_reg_methods[mouse_str][pn-1]
        refSn = refSessions[mi] # for 'old' method

        if pn < 5:
            vi = 0 # volume index, either 1 or 5
            vn = 1
        else:
            vi = 1
            vn = 5

        selDepthsRV = relative_depths_selected[mouse_str][vi]
        manRmvSi = manual_removal_Session_i[mouse_str][vi]

        # Load z-drift data
        zdrift = np.load(f"{h5Dir}JK{mouse:03}_zdrift_plane{vn}.npy", allow_pickle=True).item()

        # Select training sessions only
        # Re-order sessions if necessary
        siArr = np.where([len(sn.split('_'))==2 for sn in zdrift['info']['sessionNames']])[0]
        snums = np.array([int(sn.split('_')[1]) for sn in zdrift['info']['sessionNames'] if len(sn.split('_'))==2])
        siSorted = siArr[np.argsort(snums)]

        # Select sessions
        selectedSi = np.array([si for si in siSorted if \
                    sum(np.logical_and(zdrift['zdriftList'][si]>=selDepthsRV[0], zdrift['zdriftList'][si]<=selDepthsRV[1])) >=3 ])
        selectedSnums = [int(sname.split('_')[1]) for sname in np.array(zdrift['info']['sessionNames'])[selectedSi]]

        if len(manRmvSi)>0:
            selectedSi = np.delete(selectedSi, manRmvSi)
            selectedSnums = np.delete(selectedSnums, manRmvSi)

        zdrift_list = [zdrift['zdriftList'][si] for si in selectedSi]

        # To deal with the rolling effect of suite2p within-session registration
        leftBuffer = 30
        rightBuffer = 30 if mouse < 50 else 100
        bottomBuffer = 10
        topBuffer = 50

        edge_buffer = {'leftBuffer': leftBuffer,
                    'rightBuffer': rightBuffer,
                    'bottomBuffer': bottomBuffer,
                    'topBuffer': topBuffer}

        planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
        # regFn = f'{planeDir}s2p_nr_reg_bu1.npy' # for JK052. Need to get rid of _bu1.py file
        # Ignore this for now. If the result looks weird 

        numSelected = len(selectedSi)
        mimgs = []
        mimgClahe = []
        refOld = []
        for si, sn in enumerate(selectedSnums):
            opsFn = f'{planeDir}{sn:03}/plane0/ops.npy'
            ops = np.load(opsFn, allow_pickle=True).item()
            mimg = ops['meanImg'][topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
            mimgs.append(mimg)
            mimgClahe.append(clahe_each(mimg))
            if sn == refSn:
                refOld = ops['meanImg'][topBuffer:-bottomBuffer,leftBuffer:-rightBuffer]
        if len(refOld) == 0:
            raise('Reference session image not defined (for old reg method).')
        # Change axes of mimgs and mimgClahe to match with previous coding (suite2p legacy)
        mimgs = np.moveaxis(np.dstack(mimgs), -1, 0)
        mimgClahe = np.moveaxis(np.dstack(mimgClahe), -1, 0)

        srBi = StackReg(StackReg.BILINEAR)
        srAffine = StackReg(StackReg.AFFINE)
        regBi = np.zeros_like(mimgs)
        regAffine = np.zeros_like(mimgs)
        regS2p = np.zeros_like(mimgs)
        regOld = np.zeros_like(mimgs)

        tformsBi = []
        tformsAffine = []

        roff1 = []
        roff2 = []
        nroff1 = []
        nroff2 = []

        roff1Old = []
        roff2Old = []
        nroff1Old = []
        nroff2Old = []

        if reg_meth == 'old':
            regOld, roff1Old, roff2Old, nroff1Old, nroff2Old= s2p_2step_nr(mimgs, refOld, op)
        else:
            for si in range(numSelected):
                if si == 0:
                    if reg_meth == 'bilinear':
                        regBi[si,:,:] = mimgs[si,:,:]
                        tformsBi.append(np.eye(4))
                    elif reg_meth == 'affine':
                        regAffine[si,:,:] = mimgs[si,:,:]
                        tformsAffine.append(np.eye(3))
                    elif reg_meth == 'suite2p':
                        regS2p[si,:,:] = mimgs[si,:,:]
                        _, roff1tmp, roff2tmp, nroff1tmp, nroff2tmp = s2p_2step_nr([mimgs[si,:,:]], mimgs[si,:,:], op)
                    else:
                        raise('Registration method error.')

                else:
                    previStart = max(0,si-prevN)
                    if reg_meth == 'bilinear':
                        refBi = clahe_each(np.mean(regBi[previStart:si,:,:], axis=0))
                        tform = srBi.register(refBi, mimgClahe[si,:,:])
                        regBi[si,:,:] = srBi.transform(mimgs[si,:,:], tmat=tform)
                        tformsBi.append(tform)
                    elif reg_meth == 'affine':
                        refAffine = clahe_each(np.mean(regAffine[previStart:si,:,:], axis=0))
                        tform = srAffine.register(refAffine, mimgClahe[si,:,:])
                        regAffine[si,:,:] = srAffine.transform(mimgs[si,:,:], tmat=tform)
                        tformsAffine.append(tform)
                    elif reg_meth == 'suite2p':
                        refImg = np.mean(regS2p[previStart:si,:,:], axis=0)
                        regS2p[si,:,:], roff1tmp, roff2tmp, nroff1tmp, nroff2tmp = s2p_2step_nr([mimgs[si,:,:]], refImg, op)
                        roff1.append(roff1tmp[0])
                        roff2.append(roff2tmp[0])
                        nroff1.append(nroff1tmp[0])
                        nroff2.append(nroff2tmp[0])
                    else:
                        raise('Registration method error.')
        bilinear_result = {'reg_image': regBi,
                        'tforms': tformsBi,
                        }
        affine_result = {'reg_image': regAffine,
                        'tforms': tformsAffine,
                        }
        suite2p_result = {'reg_image': regS2p,
                        'roff1': roff1,
                        'roff2': roff2,
                        'nroff1': nroff1,
                        'nroff2': nroff2,
                        'block_size1': [op['block_size_list'][0], op['block_size_list'][0]],
                        'block_size2': [op['block_size_list'][1], op['block_size_list'][1]],
                        }
        old_result = {'reg_image': regOld,
                        'roff1': roff1Old,
                        'roff2': roff2Old,
                        'nroff1': nroff1Old,
                        'nroff2': nroff2Old,
                        'block_size1': [op['block_size_list'][0], op['block_size_list'][0]],
                        'block_size2': [op['block_size_list'][1], op['block_size_list'][1]],
                        }
        result = {'mouse': mouse,
                'plane': pn,
                'edge_buffer': edge_buffer,
                'selected_session_i': selectedSi,
                'selected_session_num': selectedSnums,
                'zdrift_list': zdrift_list,
                'registration_method': reg_meth,
                'bilinear_result': bilinear_result, 
                'affine_result': affine_result,
                'suite2p_result': suite2p_result,
                'old_result': old_result
                }

        save_fn = f'{planeDir}JK{mouse:03}_plane{pn}_session_to_session_registartion.npy'

        np.save(save_fn, result)

#%% Visually check the registration
# Visually check the registration
mi = 5
pn = 8

mouse = mice[mi]
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
load_fn = f'{planeDir}JK{mouse:03}_plane{pn}_session_to_session_registartion.npy'
result = np.load(load_fn, allow_pickle=True).item()
reg_meth = result['registration_method']
if reg_meth == 'old':
    reg_img = result['old_result']['reg_image']
elif reg_meth == 'affine':
    reg_img = result['affine_result']['reg_image']
elif reg_meth == 'bilinear':
    reg_img = result['bilinear_result']['reg_image']
elif reg_meth == 'suite2p':
    reg_img = result['suite2p_result']['reg_image']
else:
    raise('Registration method mismatch.')
    
napari.view_image(reg_img)

#%% ROI collection
# ROI collection test
mi = 1
pn = 7
mouse=mice[mi]

planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
registration_fn = f'{planeDir}JK{mouse:03}_plane{pn}_session_to_session_registartion.npy'
reg_result = np.load(registration_fn, allow_pickle=True).item()

num_sessions = len(reg_result['selected_session_num'])
reg_meth = reg_result['registration_method']
if reg_meth == 'old':
    reg_result_ops = reg_result['old_result']
    ybuffer = np.amax(np.abs(reg_result_ops['roff1'][0][0]))
    xbuffer = np.amax(np.abs(reg_result_ops['roff1'][0][1]))
elif reg_meth == 'suite2p':
    reg_result_ops = reg_result['suite2p_result']
    ybuffer = np.amax(np.abs(reg_result_ops['roff1'][0][0]))
    xbuffer = np.amax(np.abs(reg_result_ops['roff1'][0][1]))
elif reg_meth == 'affine':
    reg_result_ops = reg_result['affine_result']
elif reg_meth == 'bilinear':
    reg_result_ops = reg_result['bilinear_result']
else:
    raise('Registration method mismatch.')
reg_img = reg_result_ops['reg_image']
Ly, Lx = reg_img.shape[1:]

# Session-to-session registration creates non-overlaping regions at the edges.
# Set registration boundary to remove ROIs from each session that overlaps with the boundary.
if reg_meth == 'old' or reg_meth == 'suite2p':
    top_edge = np.amax(reg_result_ops['roff1'][0][0])
    bottom_edge = np.amin(reg_result_ops['roff1'][0][0])
    left_edge = np.amax(reg_result_ops['roff1'][0][1])
    right_edge = np.amin(reg_result_ops['roff1'][0][1])
    registration_boundary = np.ones(reg_img.shape[1:], 'uint8')
    registration_boundary[top_edge:bottom_edge, left_edge:right_edge] = 0
else:
    registration_boundary = np.sum(reg_img > 0,axis=0) < reg_img.shape[0]

# Set a master ROI map
masterMap = np.zeros((0,*reg_img.shape[1:]), 'bool')
masterCellThresh = np.zeros(0)
masterPAR = np.zeros(0) # perimeter-area ratio
roiSessionInd = np.zeros(0) # Recording which session (index) the ROIs came from
# Go through sessions and collect ROIs into the master ROI map
# Pre-sessions (901 and 902) should be at the beginning
# for snum in selectedSnums:

srBi = StackReg(StackReg.BILINEAR)
srAffine = StackReg(StackReg.AFFINE)

leftBuffer = reg_result['edge_buffer']['leftBuffer']
topBuffer = reg_result['edge_buffer']['topBuffer']
rightBuffer = reg_result['edge_buffer']['rightBuffer']
bottomBuffer = reg_result['edge_buffer']['bottomBuffer']

master_map_list = []
new_master_map_list = []
session_map_list = []
new_map_list = []
viable_cell_index_list = []

print('ROI collection')
for si in tqdm(range(num_sessions)):
    snum = reg_result['selected_session_num'][si]
    sname = f'{mouse:03}_{snum:03}'
    print(f'Processing {sname} {si}/{num_sessions-1}')
    
    if reg_meth == 'old' or reg_meth == 'suite2p':
        rigid_y1 = reg_result_ops['roff1'][0][0][si]
        rigid_x1 = reg_result_ops['roff1'][0][1][si]
        nonrigid_y1 = reg_result_ops['nroff1'][0][0][si,:]
        nonrigid_x1 = reg_result_ops['nroff1'][0][1][si,:]
        
        rigid_y2 = reg_result_ops['roff2'][0][0][si]
        rigid_x2 = reg_result_ops['roff2'][0][1][si]
        nonrigid_y2 = reg_result_ops['nroff2'][0][0][si,:]
        nonrigid_x2 = reg_result_ops['nroff2'][0][1][si,:]

        block_size1 = reg_result_ops['block_size1']
        block_size2 = reg_result_ops['block_size2']
    
    # Gather cell map and log session cell index for QC
    tempStat = np.load(f'{planeDir}{snum:03}/plane0/stat.npy', allow_pickle=True)
    tempIscell = np.load(f'{planeDir}{snum:03}/plane0/iscell.npy', allow_pickle=True)
    ops = np.load(f'{planeDir}{snum:03}/plane0/ops.npy', allow_pickle=True).item()
    Ly, Lx = ops['Ly'], ops['Lx']
    tempCelli = np.where(tempIscell[:,0])[0]
    numCell = len(tempCelli)
    tempMap = np.zeros((numCell,Ly,Lx), 'bool')
    for n, ci in enumerate(tempCelli):
        for pixi in range(len(tempStat[ci]['ypix'])):
            xi = tempStat[ci]['xpix'] - leftBuffer
            yi = tempStat[ci]['ypix'] - topBuffer
            tempMap[n,yi,xi] = 1
    tempMap = tempMap[:, topBuffer:-bottomBuffer, leftBuffer:-rightBuffer]
    # Transform
    if (reg_meth == 'old') or (reg_meth == 'suite2p'):
        tempRegMap = twostep_register(tempMap, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                        rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2)
    elif reg_meth == 'affine':
        tempRegMap = np.zeros(tempMap.shape)
        for trmi in range(tempRegMap.shape[0]):
            tempRegMap[trmi,:,:] = srAffine.transform(tempMap[trmi,:,:], tmat=reg_result_ops['tforms'][si])
    elif reg_meth == 'bilinear':
        tempRegMap = np.zeros(tempMap.shape)
        for trmi in range(tempRegMap.shape[0]):
            tempRegMap[trmi,:,:] = srBi.transform(tempMap[trmi,:,:], tmat=reg_result_ops['tforms'][si])
    else:
        raise('Registration method mismatch')

    # Transformation makes ROI map float values, not binary. 
    # Select threshold per cell after transformation, to have (roughly) matching # of pixels before the transformation
    # Save this threshold value per cell per session
    cutMap = np.zeros((numCell, *reg_img.shape[1:]), 'bool')
    delFromCut = []
    warpCellThresh = np.zeros(numCell)
    for ci in range(numCell):
        numPix = np.sum(tempMap[ci,:,:])
        cutMap[ci,:,:], warpCellThresh[ci] = calculate_regCell_threshold(tempRegMap[ci,:,:], numPix, thresholdResolution = 0.01)
        # Remove ROIs that have pixels within the edge buffers (:ybuffer, Ly-ybuffer:, :xbuffer, Lx-xbuffer:)
        if (cutMap[ci,:,:] * registration_boundary).flatten().any():
            delFromCut.append(ci)
    cutMap = np.delete(cutMap, np.array(delFromCut), axis=0)
    viable_cell_index = np.setdiff1d(range(numCell), np.array(delFromCut))
    numCell -= len(delFromCut)

    # Chronological matching and addition of ROIs
    # When there are matching ROIs, choose the one that has lower perimeter/area ratio
    
    # if masterMap.shape[0]>0:
    masterArea = np.sum(masterMap, axis=(1,2))
    newArea = np.sum(cutMap, axis=(1,2))
    masterPar = perimeter_area_ratio(masterMap)
    newPar = perimeter_area_ratio(cutMap)
    overlaps = np.zeros((masterMap.shape[0], numCell), 'uint16')
    
    # Find if there is any matched ROI, per new cells
    # Calculate overlap and applying the threshold
    for ci in range(numCell):
        overlaps[:,ci] = np.sum(masterMap*cutMap[ci,:,:], axis=(1,2))
    overlapRatioMaster = overlaps/np.tile(np.expand_dims(masterArea, axis=1), (1,numCell))
    overlapRatioNew = overlaps/np.tile(np.expand_dims(newArea, axis=0), (masterMap.shape[0],1))
    overlapMatrixOld = np.logical_or(overlapRatioMaster>=roi_overlap_threshold, overlapRatioNew>=roi_overlap_threshold)
    # # Added matching calculation: Overlap pix # > roi_overlap_threshold of median ROI pix #
    # # Median ROI calcualted from masterMap. If masterMap does not exist, then cutMap
    if len(masterArea) > 0:
        roiPixThresh = roi_overlap_threshold * np.median(masterArea)
    else:
        roiPixThresh = roi_overlap_threshold * np.median(newArea)
    overlapMatrix = np.logical_or(overlaps > roiPixThresh, overlapMatrixOld)
    
    # Deal with error cases where there can be multiple matching
    multiMatchNewInd = np.where(np.sum(overlapMatrix, axis=0)>1)[0]
    multiMatchMasterInd = np.where(np.sum(overlapMatrix, axis=1)>1)[0]
    
    # Deal with multi-matching pairs
    # First with master ROI, then with new ROIs, because there can be redundancy in multi-matching pairs
    delFromMasterInd = []
    delFromNewInd = []
    delFromMasterInd, delFromNewInd, multiMatchMasterInd, multiMatchNewInd = \
        check_multi_match_pair(multiMatchMasterInd, multiMatchNewInd, masterPar, newPar, 
                                overlapMatrix, overlaps, delFromMasterInd, delFromNewInd)
    
    if len(multiMatchNewInd)>0 or len(multiMatchMasterInd)>0:
        print(f'{len(multiMatchNewInd)} multi-match for new rois')
        print(f'{len(multiMatchMasterInd)} multi-match for master rois')
        raise('Multiple matches found after fixing multi-match pairs.')
    else:
        ################ Select what to remove for matched cells based on PAR
        # For now, treat if there is no multiple matches (because it was dealt in check_multi_match_pair)
        # Do this until I comb through all examples that I have and decide how to treat
        # multiple matches (and update check_multi_match_pair)

        for ci in range(numCell): # for every new roi
            if ci in delFromNewInd:
                continue
            else:
                matchedMasterInd = np.where(overlapMatrix[:,ci]==True)[0]
                matchedMasterInd = np.array([mi for mi in matchedMasterInd if mi not in delFromMasterInd])
                if len(matchedMasterInd)>0: # found a match in the master roi
                    # Compare perimeter-area ratio (par) between the matches
                    # Keep smaller par, remove larger PAR
                    if masterPar[matchedMasterInd] <= newPar[ci]:
                        delFromNewInd.append(ci)
                    else:
                        delFromMasterInd.append(matchedMasterInd[0])
        if len(delFromMasterInd)>0:
            newMasterMap = np.delete(masterMap, np.array(delFromMasterInd), axis=0)
            roiSessionInd = np.delete(roiSessionInd, np.array(delFromMasterInd))
        else:
            newMasterMap = masterMap.copy()
        if len(delFromNewInd)>0:
            newMap = np.delete(cutMap, np.array(delFromNewInd), axis=0)
        else:
            newMap = cutMap.copy()
        roiNewSessionInd = np.ones(newMap.shape[0])*si
        print(f'Delete from Master {delFromMasterInd}')
        print(f'Delete from New {delFromNewInd}')
    
        finalMasterMap = np.vstack((newMasterMap, newMap))
        roiSessionInd = np.concatenate((roiSessionInd, roiNewSessionInd))
        masterMap = finalMasterMap.copy()

        # Collect the result        
        master_map_list.append(masterMap) # Master map after each round. The last one is the final master map to be used.
        session_map_list.append(cutMap) # Transformed ROI map of each session after removing those overlapping with the edge
        viable_cell_index_list.append(viable_cell_index) # index of cells from the session's iscell.npy file
        new_master_map_list.append(newMasterMap) # Map of ROIs from the last master map to be included in the master map in this round
        new_map_list.append(newMap) # Map of ROIs from this session to be included in the master map in this round

        print(f'{sname} done.')
print('Collection done.')

# Now, match between the master ROI and each session ROI for QC
print('Re-matching with master ROI map')
master_map = master_map_list[-1]
matching_master_roi_index_list = []
for si in tqdm(range(num_sessions)):
    snum = reg_result['selected_session_num'][si]
    sname = f'{mouse:03}_{snum:03}'
    print(f'Processing {sname} {si}/{num_sessions-1}')

    session_map = session_map_list[si]
    viable_cell_index = viable_cell_index_list[si] # This has the same order as the session_map
    numCell = len(viable_cell_index)
    overlaps = np.zeros((master_map.shape[0], numCell), 'uint16')
    for ci in range(numCell):
        overlaps[:,ci] = np.sum(masterMap*session_map[ci,:,:], axis=(1,2))
    overlapRatioMaster = overlaps/np.tile(np.expand_dims(masterArea, axis=1), (1,numCell))
    overlapRatioNew = overlaps/np.tile(np.expand_dims(newArea, axis=0), (masterMap.shape[0],1))
    overlapMatrixOld = np.logical_or(overlapRatioMaster>=roi_overlap_threshold, overlapRatioNew>=roi_overlap_threshold)
    # # Added matching calculation: Overlap pix # > roi_overlap_threshold of median ROI pix #
    # # Median ROI calcualted from masterMap. If masterMap does not exist, then session_map
    if len(masterArea) > 0:
        roiPixThresh = roi_overlap_threshold * np.median(masterArea)
    else:
        roiPixThresh = roi_overlap_threshold * np.median(newArea)
    overlapMatrix = np.logical_or(overlaps > roiPixThresh, overlapMatrixOld)
    
    session_matching_master_roi_index = np.nan(numCell)
    for ci in range(numCell):
        matching_master_roi_index = np.where(overlapMatrix[:,ci])[0]
        if len(matching_master_roi_index) == 1:
            session_matching_master_roi_index[ci] == matching_master_roi_index
    matching_master_roi_index_list.append(session_matching_master_roi_index)
#%% save the result
save_fn = f'{planeDir}JK{mouse:03}_plane{pn}_roi_collection.npy'

result = {'master_map_list': master_map_list,
'session_map_list': session_map_list,
'viable_cell_index_list': viable_cell_index_list,
'new_master_map_list': new_master_map_list,
'new_map_list': new_map_list,
'matching_master_roi_index_list': matching_master_roi_index_list,
        }

np.save(save_fn, result)




#%% Checking multiple matches

#%% When there are multi-matches for master rois
mci = 0
masterCi = multiMatchMasterInd[mci]
newCis = np.where(overlapMatrix[masterCi,:])[0]

viewer = napari.Viewer()
viewer.add_image(masterMap[masterCi,:,:], name='from Master')
errorMap = imblend_for_napari(cutMap[newCis[0],:,:].astype(int), cutMap[newCis[1],:,:].astype(int))
viewer.add_image(errorMap, rgb=True, name='new map')

print(f'Master ROI index = {masterCi}')
for nc in newCis:
    # print(f'New ROI index from stat.npy = {tempCelli[nc]}') # to print out ROI index from stat.npy
    print(f'New ROI index from cutMap = {nc}')



#%% When there are multi-matches for new rois
nci = 0
newCi = multiMatchNewInd[nci]
masterCis = np.where(overlapMatrix[:,newCi])[0]

viewer = napari.Viewer()
viewer.add_image(cutMap[newCi,:,:], name='New map')
errorMap = imblend_for_napari(masterMap[masterCis[0],:,:].astype(int), masterMap[masterCis[1],:,:].astype(int))
viewer.add_image(errorMap, rgb=True, name='Master map')

# print(f'New ROI index from stat.npy = {tempCelli[newCi]}') 
print(f'New ROI index from cutMap = {newCi}')
for mc in masterCis:
    print(f'Master ROI index = {mc}')





#%% Get the index of a specific neuron at a specific position
ypix = 141
xpix = 672

positionMap = np.zeros((masterMap.shape[1:]))
positionMap[ypix,xpix]=1
masterCi = np.where(np.sum(masterMap*positionMap, axis=(1,2)))[0]

napari.view_image(imblend_for_napari(masterMap[masterCi[0],:,:].astype(int), masterMap[masterCi[1],:,:].astype(int)))


#%% Find the sessions
fromSession = roiSessionInd[masterCi].astype(int)
print(sname[fromSession[0]])
print(sname[fromSession[1]])
