# Copied from 220605_roi_collection_and_QC.py

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

h5Dir = 'E:/TPM/JK/h5/'

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


# for mi in range(6):
#     for pn in range(1,9):
for mi in [1]:
    for pn in [2,5,7]:

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

        save_fn = f'{planeDir}JK{mouse:03}_plane{pn}_session_to_session_registration.npy'

        np.save(save_fn, result)